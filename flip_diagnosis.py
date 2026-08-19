# -*- coding: utf-8 -*-
"""
Flip 来源诊断脚本 v2（量化误差 -> token 翻转的归因工具）
=========================================================

回答三个问题，决定量化算法往哪优化：

Q1 边际还是整体误差？  flip 是否集中在决策边际 m 极小的位置。
Q2 误差来自 LM head 还是 body？
   对 logit gap 的扰动做精确一阶分解（单次运行、无额外成本）：
     Δgap = direct + propa + cross   （恒等式，见下）
     direct = (Δw_a − Δw_b)·h_fp    -- LM head 自身量化误差
     propa  = (w_a − w_b)·Δh        -- 上游隐状态误差的传播
     cross  = (Δw_a − Δw_b)·Δh      -- 二阶交叉项（应为小量，兼作数值校验）
Q3 噪声是否异方差？    在小边际子集内，按最终隐状态范数 ||h|| 分四分位看 flip 率。

对每个 token 位置取：
  m    = FP16 (top1 − top2) logit 差          -- 决策边际
  e    = m − 量化模型在同一对 token 上的 gap    -- 边际收缩（可为负）
  flip = 量化 argmax ≠ FP16 argmax
  flip_far = flip 且目标不在 FP16 top-k 内      -- 远端翻转（灾难性噪声信号）

用法（建议依次跑 all / head / body 三次做端到端对照）：
  python flip_diagnosis.py --model Qwen/Qwen3-1.7B --bits 3 --groupsize 128 \
      --seqlen 512 --n_samples 32 --scope all
  python flip_diagnosis.py ... --scope head    # 只量化 LM head
  python flip_diagnosis.py ... --scope body    # 只量化 body
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# ----------------------------- 量化器（轻量 RTN，足够做诊断） -----------------------------

def quantize_groupwise(W: torch.Tensor, bits: int, groupsize: int) -> torch.Tensor:
    """组内对称 int 量化，尺度按输出行（per-channel × per-group），返回反量化后的 FP 权重。"""
    rows, cols = W.shape
    Wq = torch.zeros_like(W)
    maxq = 2 ** (bits - 1) - 1
    for g in range(0, cols, groupsize):
        w = W[:, g:g + groupsize]
        scale = w.abs().max(dim=1, keepdim=True).values / maxq
        scale = scale.clamp(min=1e-12)
        q = torch.clamp(torch.round(w / scale), -maxq, maxq)
        Wq[:, g:g + groupsize] = q * scale
    return Wq


def quantize_model(model: nn.Module, bits: int, groupsize: int, scope: str = "all") -> None:
    """就地假量化 nn.Linear。scope: all=全部, head=仅 LM head, body=除 LM head 外全部。"""
    lm_head = model.get_output_embeddings()
    for mod in model.modules():
        if not isinstance(mod, nn.Linear):
            continue
        if scope == "head" and mod is not lm_head:
            continue
        if scope == "body" and mod is lm_head:
            continue
        with torch.no_grad():
            mod.weight.data = quantize_groupwise(mod.weight.data, bits, groupsize)


def compute_gamma(fp_model, q_model):
    """逐模块噪声/信号能量比 γ = ‖Ŵ−W‖²_F / ‖W‖²_F（量化=投影 -> 通路增益收缩率）。
    返回 (γ_lm_head, {block_idx: 块内 Linear 的 γ 均值}, [(其他模块名, γ)])。"""
    import re
    qmap = dict(q_model.named_modules())
    lm_head = fp_model.get_output_embeddings()
    gamma_head, blocks, other = None, {}, []
    for name, mod in fp_model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        qm = qmap.get(name)
        if qm is None:
            continue
        wf = mod.weight.data.float()
        wq = qm.weight.data.float()
        g = ((wq - wf) ** 2).sum().item() / max((wf ** 2).sum().item(), 1e-12)
        if mod is lm_head:
            gamma_head = g
            continue
        mnum = re.search(r"\.(\d+)\.", name)
        if mnum:
            blocks.setdefault(int(mnum.group(1)), []).append(g)
        else:
            other.append((name, g))
    gamma_blocks = {k: float(np.mean(v)) for k, v in blocks.items()}
    return gamma_head, gamma_blocks, other


# ----------------------------- 数据 -----------------------------

def prepare_data(tokenizer, seqlen: int, n_samples: int, dataset: str) -> torch.Tensor:
    """返回 [n_samples, seqlen] 的输入 id 张量，每个样本做 next-token 预测。"""
    if dataset == "wikitext2":
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    else:  # 本地 txt 文件路径
        with open(dataset, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[0]
    n = ids.shape[0]
    data, stride = [], seqlen
    for i in range(0, n - seqlen, stride):
        data.append(ids[i:i + seqlen])
        if len(data) >= n_samples:
            break
    if not data:
        raise ValueError("语料太短，无法切出样本；调小 --seqlen 或换更长语料")
    return torch.stack(data)


# ----------------------------- 统计收集 -----------------------------

@torch.no_grad()
def collect_stats(fp_model, q_model, data, device, topk: int, chunk: int = 2):
    """分块前向（logits 是 [chunk, L, V]，不分块在真实 V 下会 OOM），
    返回逐位置统计量的 dict（每个为一维 numpy 数组，长度 = n_samples*L）。"""
    Wf = fp_model.get_output_embeddings().weight.float()   # [V, d]
    Wq = q_model.get_output_embeddings().weight.float()
    out = {k: [] for k in
           ["m", "e", "flip", "flip_far", "act", "direct", "propa", "cross", "total"]}
    rho_acc, final_energy = None, 0.0   # 每块对残差流的能量贡献份额（用于 γ 加权）

    n, L = data.shape
    pbar = tqdm(total=n, desc="collecting")
    for i in range(0, n, chunk):
        seq = data[i:i + chunk].to(device)
        of = fp_model(seq, output_hidden_states=True)
        oq = q_model(seq, output_hidden_states=True)
        lf = of.logits.float()                              # [c, L, V]
        lq = oq.logits.float()
        hs = of.hidden_states
        hf = hs[-1].float()                                 # [c, L, d] lm_head 的输入
        hq = oq.hidden_states[-1].float()
        dh = hq - hf
        if len(hs) > 1:                                     # 块贡献能量 F_l = h_{l+1} - h_l
            if rho_acc is None:
                rho_acc = [0.0] * (len(hs) - 1)
            for l in range(len(hs) - 1):
                F = hs[l + 1].float() - hs[l].float()
                rho_acc[l] += F.pow(2).sum().item()
            final_energy += hf.pow(2).sum().item()

        ftop_val, ftop_idx = lf.topk(topk, dim=-1)          # [c, L, k]
        ia, ib = ftop_idx[..., 0], ftop_idx[..., 1]         # FP16 top2 索引
        m = ftop_val[..., 0] - ftop_val[..., 1]
        qa = lq.argmax(-1)
        flip = (qa != ia).float()
        flip_far = (flip > 0) & ~((qa.unsqueeze(-1) == ftop_idx).any(-1))

        # gap 分解（恒等式：pert = lq(ia)-lq(ib) - m = direct + propa + cross）
        dv = (Wf[ia] - Wf[ib])                              # [c, L, d]
        dW = (Wq - Wf)[ia] - (Wq - Wf)[ib]
        direct = (dW * hf).sum(-1)
        propa = (dv * dh).sum(-1)
        cross = (dW * dh).sum(-1)
        pert = (lq.gather(-1, ia.unsqueeze(-1)).squeeze(-1)
                - lq.gather(-1, ib.unsqueeze(-1)).squeeze(-1)) - m
        e = -pert                                           # e ≡ 边际收缩 m−qgap（正 = 边际被吃）

        for k, v in [("m", m), ("e", e), ("flip", flip), ("flip_far", flip_far),
                     ("act", hf.norm(dim=-1)), ("direct", direct),
                     ("propa", propa), ("cross", cross), ("total", pert)]:
            out[k].append(v.reshape(-1).cpu().numpy())
        pbar.update(seq.shape[0])
    pbar.close()
    d = {k: np.concatenate(v) for k, v in out.items()}
    d["rho"] = (np.array(rho_acc) / max(final_energy, 1e-12)) if rho_acc else np.array([])
    return d


# ----------------------------- 分析 -----------------------------

M_BINS = [0, 0.1, 0.3, 0.7, 1.5, 3.0, 6.0, np.inf]
E_BINS = [-np.inf, 0, 0.1, 0.3, 0.7, 1.5, 3.0, 6.0, np.inf]


def bin_label(lo, hi):
    f = lambda x: "-∞" if x == -np.inf else ("∞" if x == np.inf else f"{x:g}")
    return f"{f(lo)}~{f(hi)}"


def bucket_table(x, bins, flip, title, xname):
    nb = np.clip(np.digitize(x, bins) - 1, 0, len(bins) - 2)
    print(f"\n=== {title} ===")
    print(f"{xname+' 区间':<12}{'样本数':>10}{'flip数':>8}{'flip率':>10}")
    for i in range(len(bins) - 1):
        mask = nb == i
        if mask.sum() == 0:
            continue
        print(f"{bin_label(bins[i], bins[i+1]):<12}{mask.sum():>10}"
              f"{int(flip[mask].sum()):>8}{flip[mask].mean()*100:>9.2f}%")


def analyze(d, gamma_head=None, gamma_blocks=None):
    m, e, flip, act = d["m"], d["e"], d["flip"], d["act"]
    n_total, n_flip = len(flip), flip.sum()
    far = d["flip_far"].sum()
    print(f"\n=== 总览 ===  位置数 {n_total}   翻转 {int(n_flip)}   flip率 {n_flip/n_total*100:.2f}%")
    if n_flip:
        print(f"远端翻转（目标不在 FP top-k 内）：{int(far)} ({far/n_flip*100:.1f}% of flips)")
        print(f"翻转位置的边际：mean m={m[flip>0].mean():.3f}  median={np.median(m[flip>0]):.3f}"
              f"  (全量 mean={m.mean():.3f})")
    else:
        print("无翻转。")

    # Q1: 边际条件 flip 率
    bucket_table(m, M_BINS, flip, "按边际 m 分桶的 flip 率（越靠近决策边界越危险）", "m")
    # 边际收缩条件 flip 率（e<0 表示量化反而扩大了边际）
    bucket_table(e, E_BINS, flip, "按边际收缩 e 分桶的 flip 率（e 大 = 量化在此方向吃得多）", "e")

    # (m, e) 二维热力图
    nb = np.clip(np.digitize(m, M_BINS) - 1, 0, len(M_BINS) - 2)
    eb = np.clip(np.digitize(e, E_BINS) - 1, 0, len(E_BINS) - 2)
    print("\n=== (m, e) 二维 flip 率热力图（行=m 列=e, %）===")
    print("m\\e     " + "".join(f"{bin_label(E_BINS[j], E_BINS[j+1]):>11}" for j in range(len(E_BINS) - 1)))
    for i in range(len(M_BINS) - 1):
        row = []
        for j in range(len(E_BINS) - 1):
            mask = (nb == i) & (eb == j)
            r = flip[mask].mean() * 100 if mask.sum() > 0 else float("nan")
            row.append(f"{r:>10.1f}%")
        print(f"{bin_label(M_BINS[i], M_BINS[i+1]):<8}" + "".join(row))

    # Q2: 误差来源分解（恒等式校验 + head/body 归因）
    resid = np.abs(d["total"] - (d["direct"] + d["propa"] + d["cross"]))
    print("\n=== 误差来源分解（logit gap 扰动的精确一阶分解）===")
    print(f"恒等式数值残差 |total−(direct+propa+cross)|：median={np.median(resid):.2e}"
          f"  max={resid.max():.2e}  （应接近 0，否则分解有 bug）")
    sel = flip > 0 if n_flip > 50 else np.ones_like(flip, bool)
    if n_flip > 0 or d["e"].__abs__().max() > 1e-9:
        ad, ap = np.abs(d["direct"][sel]), np.abs(d["propa"][sel])
        print(f"在{'翻转' if n_flip > 50 else '全部'}位置上： |direct| median={np.median(ad):.4f}"
              f"   |propa| median={np.median(ap):.4f}"
              f"   |cross| median={np.median(np.abs(d['cross'][sel])):.2e}")
        ratio = np.median(ad) / max(np.median(ap), 1e-12)
        print(f"|direct| / |propa| 中位数之比 = {ratio:.2f}")
        if ratio > 2:
            print("-> LM head 自身量化误差主导：token 对齐问题主要是最后一层的量化问题。")
        elif ratio < 0.5:
            print("-> 上游传播误差主导：需要动 body 的失真度量（margin/Fisher 加权）。")
        else:
            print("-> 两项相当：head 和 body 都要处理。")
    else:
        print("量化前后无差异（scope 未覆盖任何被量化的模块）。")

    # Q3: 异方差检验 —— 小边际内按 ||h|| 四分位
    qs = np.percentile(act, [25, 50, 75])
    aq = np.digitize(act, qs)
    small = m < 0.5
    print("\n=== 异方差检验：flip 率按最终隐状态范数 ||h|| 四分位 ===")
    print(f"{'||h|| 分位':<12}{'全量flip率':>12}{'小边际(m<0.5)flip率':>20}")
    rates = []
    for q in range(4):
        mask = aq == q
        sm = mask & small
        r_all = flip[mask].mean() * 100 if mask.sum() else float("nan")
        r_sm = flip[sm].mean() * 100 if sm.sum() > 20 else float("nan")
        rates.append(r_sm)
        print(f"Q{q+1}{'':<10}{r_all:>11.2f}%{r_sm:>19.2f}%")
    if not np.isnan(rates).any() and rates[3] / max(rates[0], 1e-9) > 2:
        print("-> 大激活位置噪声显著更大：σ-加权（异方差感知）值得做。")
    else:
        print("-> 未见强异方差信号（或小边际样本不足）。")

    # Q4: 乘性收缩检验 + γ 预测（量化=投影 -> 通路增益收缩 -> c = 1 − γ）
    mt = m - e                                            # 量化后 gap（FP16 top-2 索引处）
    A = np.vstack([m, np.ones_like(m)]).T
    coef, *_ = np.linalg.lstsq(A, mt, rcond=None)
    c_fit, b_fit = coef
    resid = mt - A @ coef
    dof = max(len(m) - 2, 1)
    s2 = float((resid ** 2).sum() / dof)
    se_c = float(np.sqrt(s2 / ((m - m.mean()) ** 2).sum()))
    r2 = 1 - float((resid ** 2).sum() / ((mt - mt.mean()) ** 2).sum())
    print("\n=== 乘性收缩检验：m̃ = c·m + b + ε（自然文本，小 margin 制域）===")
    print(f"c = {c_fit:.4f} ± {se_c:.4f}   b = {b_fit:.4f}   R² = {r2:.4f}")
    z = abs(1 - c_fit) / max(se_c, 1e-12)
    if c_fit < 1 and z > 2:
        print(f"-> c 显著 < 1（z={z:.1f}）：乘性收缩在本制域成立，纯加性(c=1)被拒绝。")
    elif c_fit > 1 and z > 2:
        print(f"-> c 显著 > 1（z={z:.1f}）：本制域无收缩甚至放大，与高 margin 制域结论相反。")
    else:
        print(f"-> c 与 1 无显著差异（z={z:.1f}）：本制域表现为加性，乘性定律不迁移。")
    if gamma_head is not None:
        rho = d["rho"]
        g_body = float(sum(rho[l] * gamma_blocks.get(l, 0.0) for l in range(len(rho))))
        c_pred = 1 - (gamma_head or 0.0) - g_body
        print(f"γ 预测：γ_head = {gamma_head:.4f}   γ_body(块能量加权) = {g_body:.4f}"
              f"   -> c_pred = {c_pred:.4f}")
        print(f"对照：c_fit = {c_fit:.4f}   |c_fit − c_pred| = {abs(c_fit - c_pred):.4f}"
              f"   （命中 ⇒ 投影理论成立；偏差大 ⇒ 检查通路假设）")

    # 综合判读
    print("\n=== 综合判读 ===")
    if n_flip:
        frac_small = ((m < 0.5) & (flip > 0)).sum() / n_flip
        print(f"flip 中发生在 m<0.5 的比例：{frac_small*100:.1f}%")
        if frac_small > 0.6:
            print("-> 边际感知失真（在 m 小的位置定向降误差）收益最大。")
        elif frac_small < 0.3:
            print("-> flip 由整体误差驱动：压整体误差（P0/膨胀场）更划算。")
        else:
            print("-> 混合来源：看热力图密度最高的 (m,e) 桶再定。")
        if n_flip > 0 and far / n_flip > 0.3:
            print("-> 远端翻转占比高：噪声过大已越出近邻竞争区，考虑提高位宽/优化分组。")
    if np.std(m) > 0:
        print(f"m 与 flip 的 Pearson 相关：{np.corrcoef(m, flip)[0,1]:.3f}（负 = 边际越小越易翻转，预期）")


# ----------------------------- main -----------------------------

def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--groupsize", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--n_samples", type=int, default=32)
    ap.add_argument("--dataset", default="wikitext2")  # 或本地 txt 路径
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--scope", choices=["all", "head", "body"], default="all")
    ap.add_argument("--chunk", type=int, default=2, help="每次前向的样本数，OOM 则调小")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dev = args.device
    try:   # transformers v5 用 dtype，v4 用 torch_dtype
        fp = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map=dev, trust_remote_code=True)
    except TypeError:
        fp = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map=dev, trust_remote_code=True)
    fp.eval()

    import copy
    q_model = copy.deepcopy(fp)
    q_model.eval()
    quantize_model(q_model, args.bits, args.groupsize, scope=args.scope)

    data = prepare_data(tok, args.seqlen, args.n_samples, args.dataset)
    print(f"模型 {args.model}  位宽 {args.bits}  groupsize {args.groupsize}  "
          f"量化范围 {args.scope}\n样本 {len(data)} 条 × {args.seqlen} tokens")

    d = collect_stats(fp, q_model, data, dev, args.topk, args.chunk)
    gamma_head, gamma_blocks, gamma_other = compute_gamma(fp, q_model)
    if gamma_other:
        gtxt = ", ".join(f"{n}:{g:.4f}" for n, g in gamma_other[:4])
        print(f"块外量化模块 γ：{gtxt}")
    analyze(d, gamma_head, gamma_blocks)


if __name__ == "__main__":
    main()
