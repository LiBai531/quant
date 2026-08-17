# -*- coding: utf-8 -*-
"""
Flip 来源诊断脚本（3-5 天级别的快速判断工具）
=================================================

目标：回答"量化模型相对 FP16 的 token 翻转到底从哪里来"，从而决定
量化失真度量该往哪优化（决策边际感知 vs 整体误差）。

对每个 token 位置，取两个标量：
  m = FP16 模型的 (top1 - top2) logit 差        —— 决策边际
  e = m - (量化模型的 top1-top2 差)            —— 量化把边际吃掉多少
翻转条件：e > m（量化后边际变负，top2 反超 top1）

输出：总体 flip 率、(m,e) 二维桶 flip 率热力图 + 判读规则。

用法：
  python flip_diagnosis.py --model Qwen/Qwen3-1.7B --bits 3 --groupsize 128 \
      --seqlen 512 --n_samples 32 --topk 8
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


def quantize_model(model: nn.Module, bits: int, groupsize: int) -> None:
    """就地假量化所有 nn.Linear（保留 bias，权重替换为反量化值）。"""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                m.weight.data = quantize_groupwise(m.weight.data, bits, groupsize)


# ----------------------------- 数据 -----------------------------

def prepare_data(tokenizer, seqlen: int, n_samples: int, dataset: str) -> torch.Tensor:
    """返回 [n_samples, seqlen] 的输入 id 张量，每个样本做 next-token 预测。"""
    from datasets import load_dataset
    if dataset == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    else:  # 本地 txt 文件路径
        with open(dataset, "r", encoding="utf-8") as f:
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
def collect_stats(fp_model, q_model, loader, device, topk: int):
    m_list, e_list, flip_list = [], [], []
    for seq in tqdm(loader, desc="collecting"):
        seq = seq.to(device)
        lf = fp_model(seq).logits.float()          # [L, V]
        lq = q_model(seq).logits.float()
        ftop_val, ftop_idx = lf.topk(topk, dim=-1)  # [L, topk]
        qtop = lq.gather(-1, ftop_idx[:, :2])       # 量化模型在 FP16 top2 处的值
        fa = lf.argmax(-1)
        qa = lq.argmax(-1)
        m = ftop_val[:, 0] - ftop_val[:, 1]          # 决策边际
        qm = qtop[:, 0] - qtop[:, 1]                 # 量化后的边际
        e = m - qm                                   # 边际收缩
        flip = (qa != fa).float()                    # 翻转 = argmax 不一致
        m_list.append(m.cpu().numpy())
        e_list.append(e.cpu().numpy())
        flip_list.append(flip.cpu().numpy())
    return (np.concatenate(m_list), np.concatenate(e_list), np.concatenate(flip_list))


# ----------------------------- 分析 -----------------------------

M_BINS = [0, 0.1, 0.3, 0.7, 1.5, 3.0, 6.0, np.inf]
E_BINS = [0, 0.1, 0.3, 0.7, 1.5, 3.0, 6.0, np.inf]
BIN_LABEL = lambda b: f"{b[0]:.1f}~{b[1] if b[1] != np.inf else '∞'}"


def analyze(m, e, flip):
    nb = np.digitize(m, M_BINS) - 1
    eb = np.digitize(e, E_BINS) - 1
    nb = np.clip(nb, 0, len(M_BINS) - 2)
    eb = np.clip(eb, 0, len(E_BINS) - 2)

    n_total = len(flip)
    n_flip = flip.sum()
    print(f"\n=== 总览 ===  位置数 {n_total}   翻转 {int(n_flip)}   flip率 {n_flip/n_total*100:.2f}%")
    print(f"翻转位置的边际分布：mean m={m[flip>0].mean():.3f} (全量 mean={m.mean():.3f})")

    # 边际条件 flip 率
    print("\n=== 按边际 m 分桶的 flip 率（越靠近决策边界越危险）===")
    print(f"{'m 区间':<12}{'样本数':>10}{'flip数':>8}{'flip率':>10}")
    for i in range(len(M_BINS) - 1):
        mask = nb == i
        if mask.sum() == 0:
            continue
        r = flip[mask].mean()
        print(f"{BIN_LABEL((M_BINS[i], M_BINS[i+1])):<12}{mask.sum():>10}{int(flip[mask].sum()):>8}{r*100:>9.2f}%")

    # 边际收缩条件 flip 率
    print("\n=== 按边际收缩 e 分桶的 flip 率（e 大 = 量化在此方向吃得多）===")
    print(f"{'e 区间':<12}{'样本数':>10}{'flip数':>8}{'flip率':>10}")
    for i in range(len(E_BINS) - 1):
        mask = eb == i
        if mask.sum() == 0:
            continue
        r = flip[mask].mean()
        print(f"{BIN_LABEL((E_BINS[i], E_BINS[i+1])):<12}{mask.sum():>10}{int(flip[mask].sum()):>8}{r*100:>9.2f}%")

    # 2D 热力图
    print("\n=== (m, e) 二维 flip 率热力图（行=m 列=e）===")
    hdr = "m\\e    " + "".join(f"{BIN_LABEL((E_BINS[j], E_BINS[j+1])):>11}" for j in range(len(E_BINS) - 1))
    print(hdr)
    for i in range(len(M_BINS) - 1):
        row = []
        for j in range(len(E_BINS) - 1):
            mask = (nb == i) & (eb == j)
            r = flip[mask].mean() * 100 if mask.sum() > 0 else float("nan")
            row.append(f"{r:>10.1f}%")
        print(f"{BIN_LABEL((M_BINS[i], M_BINS[i+1])):<8}" + "".join(row))

    # 决策判读
    small_m = m < 0.5
    flip_small_m = (small_m & (flip > 0)).sum() / max(n_flip, 1)
    print(f"\n=== 判读 ===\nflip 中发生在 m<0.5 的比例：{flip_small_m*100:.1f}%")
    if flip_small_m > 0.6:
        print("→ 判读1：flip 集中在极小边际。决策边际感知的失真度量（在 m 小的位置降误差）最有价值。")
    elif flip_small_m < 0.3:
        print("→ 判读2：flip 分布较广，由整体误差驱动。压整体误差（P0/膨胀场）比边界特化更划算。")
    else:
        print("→ 判读3：混合来源。先看热力图哪个 (m,e) 桶密度最高，再定。")
    corr = np.corrcoef(m, flip)[0, 1]
    print(f"m 与 flip 的 Pearson 相关：{corr:.3f}（负 = 边际越小越易翻转，预期）")


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
    ap.add_argument("--dataset", default="wikitext2")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dev = args.device
    fp = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=dev, trust_remote_code=True)
    fp.eval()

    import copy
    q_model = copy.deepcopy(fp)
    q_model.eval()
    quantize_model(q_model, args.bits, args.groupsize)

    data = prepare_data(tok, args.seqlen, args.n_samples, args.dataset)
    loader = [data] if data.ndim == 2 else data
    print(f"样本 {len(loader)} 条 × {args.seqlen} tokens，位宽 {args.bits}，groupsize {args.groupsize}")

    m, e, flip = collect_stats(fp, q_model, loader, dev, args.topk)
    analyze(m, e, flip)


if __name__ == "__main__":
    main()
