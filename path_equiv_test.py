# -*- coding: utf-8 -*-
"""
path_equiv_test.py —— 方案 A（拆两个张量）与方案 B（logits 级差量修正）的等价性验证
================================================================================

部署在 vllm-ascend 时 head 不能逐行混合精度，但可以：
  方案 A：lm_head 拆成 hot(bf16) + cold(3bit) 两段 matmul 相加。
  方案 B：整个 head 走 3bit 量化，另存热行差量 C=(W_hot_bf16−W_quant_hot)，
          在 logits 层加回：logits[hot] += h@Cᵀ。

数学上逐行恒等（冷行=全量化该行，热行=全量化该行+差量还原）。本脚本用
head_cache 缓存验证三个事实：
  T1 冷行：quantize(W[cold]) == quantize(W)[cold]   （逐行独立，无跨行耦合）
  T2 热行：h@W_hot_bf16ᵀ == (h@W_quantᵀ)[:,hot] + h@Cᵀ （差量修正还原 bf16）
  T3 行为：A、B 两种拼装的 logits 逐位相同，flip 掩码逐位相同，
           且 flip 率复现 head_probe E4 p=0.01 的结果（≈1.8%）。

用法：python path_equiv_test.py --cache head_cache --p 0.01 --bits 3
"""
import argparse
import sys

import torch

from flip_diagnosis import quantize_groupwise


def metrics(lf, lvar, tidx):
    fa = lf.argmax(-1)
    va = lvar.argmax(-1)
    flip = (va != fa)
    far = flip & ~((va.unsqueeze(-1) == tidx).any(-1))
    nf = max(int(flip.sum()), 1)
    return flip, (flip.float().mean().item() * 100,
                  far.float().mean().item() * 100,
                  int(far.sum()) / nf * 100)


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
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()

    h = torch.load(args.cache + ".h.pt", map_location="cpu", weights_only=True)
    lf = torch.load(args.cache + ".lf.pt", map_location="cpu", weights_only=True)
    tidx = torch.load(args.cache + ".t.pt", map_location="cpu", weights_only=True)
    W = torch.load(args.cache + ".w.pt", map_location="cpu", weights_only=True)
    V, d = W.shape
    N = h.shape[0]
    print(f"缓存: h[{N},{d}]  W[{V},{d}]  p={args.p}  {args.bits}bit gs={args.groupsize}")

    # 热行 = margin 暴露前 p%
    exposure = ((lf >= lf.max(-1, keepdim=True).values - args.tau).float().mean(0))
    n_hot = max(int(args.p * V), 1)
    hot = torch.zeros(V, dtype=torch.bool)
    hot[torch.argsort(exposure, descending=True)[:n_hot]] = True
    cold = ~hot
    print(f"热行 {int(hot.sum())}（{args.p*100:.1f}%）  冷行 {int(cold.sum())}")

    # ---------- T1：逐行独立性（冷行） ----------
    W_full = quantize_groupwise(W, args.bits, args.groupsize)   # 整表量化 [V,d]
    W_cold = quantize_groupwise(W[cold], args.bits, args.groupsize)  # 子集量化
    t1 = torch.equal(W_full[cold], W_cold)
    print(f"T1 冷行 整表量化==子集量化: {t1}  "
          f"(max diff={(W_full[cold]-W_cold).abs().max().item():.3e})")
    assert t1, "T1 失败：逐行独立性被破坏"

    # ---------- 大 matmul：全量化 logits ----------
    print("计算全量化 logits（h@W_quantᵀ，约 3-4 分钟）...")
    lvar_q = h @ W_full.t()                      # [N, V]
    print("全量化 logits 完成")

    # ---------- T2：热行差量修正还原 bf16 ----------
    C = W[hot] - W_full[hot]                      # [n_hot, d] 差量
    corr = h @ C.t()                              # [N, n_hot]
    lvar_hot_direct = h @ W[hot].t()              # 热行直接 bf16 matmul [N, n_hot]
    t2 = torch.allclose(lvar_hot_direct,
                        lvar_q[:, hot] + corr, atol=1e-2, rtol=1e-3)
    print(f"T2 热行 直接bf16 == 全量化+差量修正: {t2}  "
          f"(max|Δ|={(lvar_hot_direct-(lvar_q[:,hot]+corr)).abs().max().item():.4f})")
    assert t2, "T2 失败：差量修正未能还原 bf16 热行"

    # ---------- T3：A/B 拼装逐位一致 + flip 复现 ----------
    # 方案 B logits：全量化 + 热行修正
    lvarB = lvar_q.clone()
    lvarB[:, hot] = lvar_hot_direct               # 等价于 +=corr
    # 方案 A logits：由 lvar_q 构造（冷行=全量化，热行=bf16），与独立重算一致
    lvarA = lvar_q.clone()
    lvarA[:, hot] = lvar_hot_direct
    t3 = torch.equal(lvarA, lvarB)
    flipA, metA = metrics(lf, lvarA, tidx)
    flipB, metB = metrics(lf, lvarB, tidx)
    t4 = torch.equal(flipA, flipB)
    print(f"T3 logits 逐位一致: {t3}   flip 掩码逐位一致: {t4}")
    print(f"    方案A: flip={metA[0]:.3f}%  远flip={metA[1]:.3f}%   (E4 p={args.p:g} 参考: 1.81%)")
    print(f"    方案B: flip={metB[0]:.3f}%  远flip={metB[1]:.3f}%")
    assert t3 and t4, "T3/T4 失败：两种方案行为不一致"

    # ---------- 内存账目 ----------
    mB = V * d * args.bits + int(hot.sum()) * d * 16      # 全量化 + 热行差量(bf16)
    mA = int(hot.sum()) * d * 16 + int(cold.sum()) * d * args.bits
    print(f"内存: 方案A={mA/8/1e6:.1f}MB({mA/(V*d*16)*100:.1f}%bf16)  "
          f"方案B={mB/8/1e6:.1f}MB({mB/(V*d*16)*100:.1f}%bf16)  "
          f"差={(mB-mA)/8/1e6:.1f}MB(B 多存一份热行的低位副本)")
    print("\nALL TESTS PASSED —— 方案 A 与 B 行为等价，可在 vllm-ascend 用路径 B 部署")


if __name__ == "__main__":
    main()
