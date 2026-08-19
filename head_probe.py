# -*- coding: utf-8 -*-
"""
head_probe.py —— LM head 子空间/行级精度探针
=============================================

动机：head 是残差流最后一层，误差无下游吸收器，故生产管线一直回退 bf16。
但 logits = W_u·h 只依赖 h 落在低维流形 S 上的分量，S^⊥ 方向在数据流形上
逐比特不参与计算 —— 理论上可以精确丢掉。

三个实验（全部只需 fp 模型一次前向 + 若干矩阵运算）：

E1 内在秩：Σ_h = E[hhᵀ] 的谱，有效秩 r 与能量捕获曲线、S^⊥ 残余能量。
E2 子空间投影量化：W_u = W_S·V_rᵀ + W_⊥，丢 W_⊥，量化 W_S，测 flip 率/远 flip 率/内存。
E3 行级精度分配：按 margin 暴露(该行是否出现在决策窗口内)分配行位宽，扫描预算曲线。

用法：
  python head_probe.py --model D:/models/Qwen3.5-4B --dataset D:/论文/sinq.txt \
      --n_samples 6 --bits 3 --groupsize 128 --capture 0.99 --device cuda
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from flip_diagnosis import prepare_data, quantize_groupwise

TAU = 0.5  # 决策窗口（logit 单位）：某行落在 max−τ 内即视为有 margin 暴露


@torch.no_grad()
def collect_head_data(fp_model, data, device, chunk=2):
    """收集 h_final、fp logits、fp top-8 索引。head-only 分析只需 fp 一次前向。"""
    h_list, lf_list, tidx_list = [], [], []
    n = data.shape[0]
    pbar = tqdm(total=n, desc="collect head data")
    for i in range(0, n, chunk):
        seq = data[i:i + chunk].to(device)
        out = fp_model(seq, output_hidden_states=True)
        h = out.hidden_states[-1].float()      # [c, L, d]
        lf = out.logits.float()                # [c, L, V]
        _, topi = lf.topk(8, dim=-1)
        h_list.append(h.reshape(-1, h.shape[-1]).cpu())
        lf_list.append(lf.reshape(-1, lf.shape[-1]).cpu())
        tidx_list.append(topi.reshape(-1, 8).cpu())
        pbar.update(seq.shape[0])
    pbar.close()
    return torch.cat(h_list), torch.cat(lf_list), torch.cat(tidx_list)


def spectrum_report(h, capture_levels=(0.90, 0.95, 0.99, 0.995)):
    """E1: 非中心协方差谱、有效秩、能量捕获、S^⊥ 残余能量。返回 (evals, evecs, r_choices)。"""
    d = h.shape[-1]
    cov = (h.t() @ h) / h.shape[0]             # [d,d]
    evals, evecs = torch.linalg.eigh(cov)      # 升序
    evals = evals.flip(0)
    evecs = evecs.flip(1)
    total = evals.sum().clamp(min=1e-12)
    cum = evals.cumsum(0) / total
    r_choices = {}
    for c in capture_levels:
        idx = (cum >= c).nonzero()
        r_choices[c] = int(idx[0].item()) if len(idx) else d
    pr = (evals.sum() ** 2) / evals.pow(2).sum().clamp(min=1e-12)  # 参与比
    print(f"--- E1 内在秩 ---  d={d}  参与比={pr:.1f}")
    for c in capture_levels:
        print(f"   能量捕获 {c*100:.0f}% → r={r_choices[c]} ({r_choices[c]/d*100:.1f}% of d)")
    # S^⊥ 残余能量（r 取 99% 捕获）：纯投影丢 ⊥ 的损失上界
    r = r_choices[0.99]
    Vr = evecs[:, :r]                           # [d, r]
    P = Vr @ Vr.t()
    res = (h - h @ P).pow(2).sum(-1).mean().item() / h.pow(2).mean(-1).mean().clamp(min=1e-12)
    print(f"   99% 捕获时 ‖P_⊥h‖²/‖h‖² = {res:.4f}（纯投影丢 ⊥ 的 logit 相对损失）")
    return evals, evecs, r_choices


def metrics(lf, lvar, tidx):
    fa = lf.argmax(-1)
    va = lvar.argmax(-1)
    flip = (va != fa)
    far = flip & ~((va.unsqueeze(-1) == tidx).any(-1))
    nf = max(int(flip.sum()), 1)
    return (flip.float().mean().item() * 100,
            far.float().mean().item() * 100,
            int(far.sum()) / nf * 100)


def run_variant(lf, h, tidx, W, bits, groupsize, Vr=None, W_keep=None):
    """算一个 head 变体的 flip 指标。W: [V,d]（全量）或 [V,r]（子空间内）。"""
    if W_keep is not None:                      # 行级混合精度：W_keep 为 True 的行保持原值
        Wq = W.clone()
        cold = ~W_keep
        if cold.any():
            Wq[cold] = quantize_groupwise(W[cold], bits, groupsize)
    else:
        Wq = quantize_groupwise(W, bits, groupsize) if bits else W.clone()
    if Vr is not None:
        lvar = (h @ Vr) @ Wq.t()                # ℓ = (V_rᵀh)·W_S  —— 子空间内
    else:
        lvar = h @ Wq.t()
    return metrics(lf, lvar, tidx)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="D:/models/Qwen3.5-4B")
    ap.add_argument("--dataset", default="wikitext2")
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--n_samples", type=int, default=6)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--groupsize", type=int, default=128)
    ap.add_argument("--capture", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    try:
        fp = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map=args.device,
            trust_remote_code=True)
    except TypeError:
        fp = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map=args.device,
            trust_remote_code=True)
    fp.eval()

    lm_head = fp.get_output_embeddings()
    W = lm_head.weight.float().cpu()            # [V, d]
    V, d = W.shape
    print(f"LM head: {V}×{d}  (占模型内存大头的部分，当前方案为 bf16 全保)")

    data = prepare_data(tok, args.seqlen, args.n_samples, args.dataset)
    print(f"语料 {len(data)} 条 × {args.seqlen} tok\n")
    h, lf, tidx = collect_head_data(fp, data, args.device, args.chunk)

    # ---------- E1 内在秩 ----------
    evals, evecs, r_choices = spectrum_report(h)
    r = r_choices[args.capture]
    Vr = evecs[:, :r]                            # [d, r]

    # 纯投影(不量化)丢 ⊥ 的 logit 误差与 flip —— 子空间充分性检查
    lvar_drop = (h @ Vr) @ (W @ Vr).t()          # 丢 ⊥、未量化
    dp = (lvar_drop - lf).abs().max().item()
    dp_mean = (lvar_drop - lf).abs().mean().item()
    fl_drop, fa_drop, fr_drop = metrics(lf, lvar_drop, tidx)
    print(f"纯投影丢 ⊥（未量化）：max|Δlogit|={dp:.3f}  mean={dp_mean:.4f}  "
          f"flip率={fl_drop:.2f}%  远flip率={fa_drop:.2f}%")

    # ---------- E2 子空间投影量化 ----------
    W_S = W @ Vr                                 # [V, r]
    bits = args.bits
    mem = {"bf16": V * d * 16}
    print(f"\n--- E2 子空间投影量化（{bits}bit, gs={args.groupsize}, r={r}）---")
    print(f"{'变体':<34}{'flip率':>8}{'远flip率':>9}{'远flip占比':>10}{'内存(bits)':>12}{'内存%bf16':>9}")
    rows = []
    f_flip, f_far, f_fr = run_variant(lf, h, tidx, W, None, args.groupsize)
    rows.append(("head bf16(参考)", f_flip, f_far, f_fr, mem["bf16"], 100.0))
    mem["rtn"] = V * d * bits
    r_flip, r_far, r_fr = run_variant(lf, h, tidx, W, bits, args.groupsize)
    rows.append((f"全量 RTN {bits}bit", r_flip, r_far, r_fr, mem["rtn"],
                 mem["rtn"] / mem["bf16"] * 100))
    mem["proj"] = V * r * bits + r * d * 16
    p_flip, p_far, p_fr = run_variant(lf, h, tidx, W_S, bits, args.groupsize, Vr=Vr)
    rows.append((f"投影丢⊥ + W_S@{bits}bit + R@bf16", p_flip, p_far, p_fr, mem["proj"],
                 mem["proj"] / mem["bf16"] * 100))
    mem["proj_bf16"] = V * r * 16 + r * d * 16
    pb_flip, pb_far, pb_fr = run_variant(lf, h, tidx, W_S, None, args.groupsize, Vr=Vr)
    rows.append((f"投影丢⊥ + W_S@bf16(上限)", pb_flip, pb_far, pb_fr, mem["proj_bf16"],
                 mem["proj_bf16"] / mem["bf16"] * 100))
    for nm, a, b, c, mem_b, mem_p in rows:
        print(f"{nm:<34}{a:>7.2f}%{b:>8.2f}%{c:>9.1f}%{mem_b:>12.0f}{mem_p:>8.1f}%")

    # ---------- E3 行级精度分配（在子空间内，按 margin 暴露） ----------
    tau = args.tau
    exposure = ((lf >= lf.max(-1, keepdim=True).values - tau).float().mean(0))  # [V]
    print(f"\n--- E3 行级精度分配（τ={tau} 决策窗口内，投影 r={r}，热行 bf16 / 冷行 {bits}bit）---")
    print(f"{'热行比例p':<10}{'热行数':>8}{'flip率':>8}{'远flip率':>9}{'内存(bits)':>12}{'内存%bf16':>9}")
    for p in (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.5, 1.0):
        n_hot = max(int(p * V), 1 if p > 0 else 0)
        if n_hot == 0:
            keep = torch.zeros(V, dtype=torch.bool)
        else:
            keep = torch.zeros(V, dtype=torch.bool)
            keep[torch.argsort(exposure, descending=True)[:n_hot]] = True
        a, b, c = run_variant(lf, h, tidx, W_S, bits, args.groupsize, Vr=Vr, W_keep=keep)
        m = int(keep.sum()) * r * 16 + (V - int(keep.sum())) * r * bits + r * d * 16
        print(f"{p:<10.3f}{int(keep.sum()):>8}{a:>7.2f}%{b:>8.2f}%{m:>12.0f}{m/mem['bf16']*100:>8.1f}%")


if __name__ == "__main__":
    main()
