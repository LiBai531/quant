# -*- coding: utf-8 -*-
"""
head_standalone_test.py —— 热行选择能否纯权重代理（免前向）？
================================================================

E4 的热行选择需要 exposure（logits 前向）。若纯权重代理（行范数等）选出的
热行与 exposure 重合度高、且 flip 恢复效果相当，则整个 head 量化可以做成
零前向的独立脚本（只读 safetensors 的 lm_head.weight）。

评估 4 种选择器（各保护 top-n_hot 行 bf16，其余 3bit）：
  exp    : exposure（真值，需前向）
  norm   : 行范数 ‖w_v‖（纯权重）
  cnorm  : 去均值行范数 ‖w_v − w̄‖（argmax 平移不变，纯权重）
  random : 随机（对照）
指标：IoU（各选择器 vs exp）、flip 率、远flip 率。

用法：python head_standalone_test.py --cache head_cache --p 0.01 --bits 3
"""
import argparse
import sys

import torch

from flip_diagnosis import quantize_groupwise

TAU = 0.5


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="head_cache")
    ap.add_argument("--p", type=float, default=0.01)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--groupsize", type=int, default=128)
    ap.add_argument("--tau", type=float, default=TAU)
    args = ap.parse_args()

    h = torch.load(args.cache + ".h.pt", map_location="cpu", weights_only=True)
    lf = torch.load(args.cache + ".lf.pt", map_location="cpu", weights_only=True)
    tidx = torch.load(args.cache + ".t.pt", map_location="cpu", weights_only=True)
    W = torch.load(args.cache + ".w.pt", map_location="cpu", weights_only=True)
    V, d = W.shape
    n_hot = max(int(args.p * V), 1)
    fa = lf.argmax(-1)

    # ---- 各选择器 ----
    exposure = ((lf >= lf.max(-1, keepdim=True).values - args.tau).float().mean(0))
    exp_top = torch.argsort(exposure, descending=True)[:n_hot]
    norm = W.norm(dim=-1)
    norm_top = torch.argsort(norm, descending=True)[:n_hot]
    Wc = W - W.mean(0, keepdim=True)          # 去均值（argmax 平移不变）
    cnorm_top = torch.argsort(Wc.norm(dim=-1), descending=True)[:n_hot]
    # 语料 token 频率：fp argmax 近似观察到的 next-token 序列 -> 频率前 n_hot
    fa = lf.argmax(-1)
    freq = torch.zeros(V)
    freq.scatter_add_(0, fa, torch.ones_like(fa, dtype=freq.dtype))
    freq_top = torch.argsort(freq, descending=True)[:n_hot]
    rng = torch.Generator().manual_seed(0)
    rand_top = torch.randperm(V, generator=rng)[:n_hot]

    def iou(a, b):
        return len(set(a.tolist()) & set(b.tolist())) / n_hot

    print(f"热行数 {n_hot}  选择器间 IoU vs exp：")
    for nm, top in (("norm", norm_top), ("cnorm", cnorm_top), ("freq", freq_top),
                    ("random", rand_top)):
        print(f"  {nm:<7} IoU={iou(exp_top, top)*100:.1f}%")

    # ---- 全量化 logits（一次大 matmul，缓存避免重复），各选择器用差量修正拼接 ----
    import os
    lq_path = args.cache + ".lvarq.pt"
    Wq = quantize_groupwise(W, args.bits, args.groupsize)
    if os.path.exists(lq_path):
        print(f"\n加载全量化 logits 缓存 {lq_path}")
        lvar_q = torch.load(lq_path, map_location="cpu", weights_only=True)
    else:
        print("\n计算全量化 logits（一次大 matmul）...")
        lvar_q = h @ Wq.t()
        torch.save(lvar_q, lq_path)
        print("完成，已缓存\n")

    def eval_sel(name, top):
        hot = torch.zeros(V, dtype=torch.bool)
        hot[top] = True
        C = W[hot] - Wq[hot]
        lv = lvar_q.clone()
        lv[:, hot] = lvar_q[:, hot] + h @ C.t()   # 等价于用 bf16 热行重算
        va = lv.argmax(-1)
        flip = (va != fa)
        far = flip & ~((va.unsqueeze(-1) == tidx).any(-1))
        nf = max(int(flip.sum()), 1)
        return (flip.float().mean().item() * 100, far.float().mean().item() * 100,
                int(far.sum()) / nf * 100)

    print(f"{'选择器':<8}{'flip率':>9}{'远flip率':>10}{'远flip占比':>11}")
    for nm, top in (("exp", exp_top), ("norm", norm_top), ("cnorm", cnorm_top),
                    ("freq", freq_top), ("random", rand_top)):
        a, b, c = eval_sel(nm, top)
        tag = " ← 真值" if nm == "exp" else (" ← 纯权重" if nm != "random" else " ← 对照")
        print(f"{nm:<8}{a:>8.2f}%{b:>9.2f}%{c:>10.1f}%{tag}")

    a, b, c = eval_sel("all_cold", torch.tensor([], dtype=torch.long))
    print(f"{'all_cold':<8}{a:>8.2f}%{b:>9.2f}%{c:>10.1f}%  ← 基线(全3bit)")


if __name__ == "__main__":
    main()
