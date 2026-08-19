# -*- coding: utf-8 -*-
"""e4_one.py —— E4（全量 head 行级分配）单行计算器，低内存、可杀续。

每个进程只算一个 p 值（一次完整 matmul，~4 分钟，内存 ~5GB，不加载模型），
结果追加到 cache.e4.txt。被杀后换下一个 p 继续即可。
"""
import argparse
import sys

import torch

from flip_diagnosis import quantize_groupwise

TAU = 0.5


def metrics(lf, lvar, tidx):
    fa = lf.argmax(-1)
    va = lvar.argmax(-1)
    flip = (va != fa)
    far = flip & ~((va.unsqueeze(-1) == tidx).any(-1))
    nf = max(int(flip.sum()), 1)
    return (flip.float().mean().item() * 100,
            far.float().mean().item() * 100,
            int(far.sum()) / nf * 100)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="head_cache")
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--groupsize", type=int, default=128)
    ap.add_argument("--tau", type=float, default=TAU)
    args = ap.parse_args()

    h = torch.load(args.cache + ".h.pt", map_location="cpu", weights_only=True)
    lf = torch.load(args.cache + ".lf.pt", map_location="cpu", weights_only=True)
    tidx = torch.load(args.cache + ".t.pt", map_location="cpu", weights_only=True)
    W = torch.load(args.cache + ".w.pt", map_location="cpu", weights_only=True)
    V, d = W.shape

    exposure = ((lf >= lf.max(-1, keepdim=True).values - args.tau).float().mean(0))
    n_hot = max(int(args.p * V), 1 if args.p > 0 else 0)
    if n_hot == 0:
        keep = torch.zeros(V, dtype=torch.bool)
    else:
        keep = torch.zeros(V, dtype=torch.bool)
        keep[torch.argsort(exposure, descending=True)[:n_hot]] = True

    Wq = W.clone()
    cold = ~keep
    if cold.any():
        Wq[cold] = quantize_groupwise(W[cold], args.bits, args.groupsize)

    lvar = h @ Wq.t()
    a, b, c = metrics(lf, lvar, tidx)
    m = int(keep.sum()) * d * 16 + (V - int(keep.sum())) * d * args.bits
    line = (f"p={args.p:g} nhot={int(keep.sum())} flip={a:.4f} far={b:.4f} "
            f"mem={m} pct={m / (V * d * 16) * 100:.3f}\n")
    with open(args.cache + ".e4.txt", "a", encoding="utf-8") as f:
        f.write(line)
    print(line.strip())


if __name__ == "__main__":
    main()
