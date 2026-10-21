"""E_probe_shapes.py — 只读探针：`attention_masked` / `attention_band` 的**入出秩**矩阵（agent E，缺陷一）。

零写入、零产品改动：只 import `sys1` 与 torch，把 2/3/4 维入口在两条路上的出口形状、
是否崩溃、批与逐样本是否同值，一次全示众。修复前后各跑一次即得对照表。

用法（release 目录下）：
    .venv/bin/python ascend/kernels/reconcile/E_probe_shapes.py            # 秩矩阵 + 批/逐样本一致性
    .venv/bin/python ascend/kernels/reconcile/E_probe_shapes.py compare    # 数值指纹（改前后必须逐位同）
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]      # release/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sys1.layers import attention as LA

SEQ, WIN, CHUNK, HEADS, DIM, SEED = 32, 8, 16, 2, 4, 20261005


def synth(rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """同一份 seed 的数据按 2/3/4 维开本取出（4 维 = 3 维加一截 batch=1 前导维）。"""
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    q = torch.randn(HEADS, SEQ, DIM, generator=gen)
    k = torch.randn(HEADS, SEQ, DIM, generator=gen)
    v = torch.randn(HEADS, SEQ, DIM, generator=gen)
    if rank == 2:
        return q[0].contiguous(), k[0].contiguous(), v[0].contiguous()
    if rank == 4:
        return q[None].contiguous(), k[None].contiguous(), v[None].contiguous()
    return q, k, v


def probe(fn, kwargs: dict, rank: int) -> str:
    q, k, v = synth(rank)
    try:
        out = fn(q, k, v, **kwargs).out
    except Exception as e:                                        # noqa: BLE001  示众用，原样打
        return f"CRASH {type(e).__name__}: {e}"
    tag = "OK" if out.dim() == rank else "RANK-VIOLATION"
    return f"{tag:<16} in={tuple(q.shape)} -> out={tuple(out.shape)}"


def cmd_matrix() -> None:
    print(f"[env] torch={torch.__version__} seq={SEQ} heads={HEADS} dim={DIM} window={WIN} chunk={CHUNK}")
    for rank in (2, 3, 4):
        print(f"  masked window={WIN:>4}  rank={rank}: {probe(LA.attention_masked, {'window': WIN}, rank)}")
    for rank in (2, 3, 4):
        print(f"  masked window=None  rank={rank}: {probe(LA.attention_masked, {'window': None}, rank)}")
    for route in ("kernel", "local"):
        for rank in (2, 3, 4):
            kw = {"window": WIN, "chunk": CHUNK, "route": route}
            print(f"  band {route:<6}       rank={rank}: {probe(LA.attention_band, kw, rank)}")


def cmd_batch_vs_loop() -> None:
    """4 维批入 ≡ 逐样本 3 维循环（同秩 + 同数值）——修复前这一条根本走不到（崩）。"""
    cases = (
        (LA.attention_masked, {"window": WIN}),
        (LA.attention_band, {"window": WIN, "chunk": CHUNK, "route": "local"}),
        (LA.attention_band, {"window": WIN, "chunk": CHUNK, "route": "kernel"}),
    )
    for fn, kwargs in cases:
        q4, k4, v4 = synth(4)
        label = f"{fn.__name__}({kwargs})"
        try:
            batched = fn(q4, k4, v4, **kwargs).out
        except Exception as e:                                    # noqa: BLE001
            print(f"  {label}: CRASH {type(e).__name__}: {e}")
            continue
        per = torch.stack([fn(q4[b], k4[b], v4[b], **kwargs).out for b in range(q4.size(0))])
        same_shape = tuple(batched.shape) == tuple(per.shape)
        max_abs = float((batched - per).abs().max()) if same_shape else float("nan")
        print(f"  {label}: batched={tuple(batched.shape)} loop={tuple(per.shape)} "
              f"shape_eq={same_shape} bitwise_eq={bool(same_shape and torch.equal(batched, per))} "
              f"max_abs={max_abs!r}")


def cmd_compare() -> None:
    """数值指纹：形状修复**不许**改数学——3 维掩码路出口字节哈希 + compare_routes 差值须与改前一致。"""
    q, k, v = synth(3)
    out3 = LA.attention_masked(q, k, v, window=WIN).out
    flat = out3.detach().reshape(-1).contiguous()          # 展平再哈希：形状修复会改秩，数学不许改
    fp = hashlib.sha1(flat.numpy().tobytes()).hexdigest()[:16]
    rep = LA.compare_routes(200, 32, chunk=48)
    print(f"  masked(3d).out shape={tuple(out3.shape)} numel={flat.numel()} sha1_16={fp}")
    print(f"  compare_routes(200,32,chunk=48): kernel_vs_masked_max_abs={rep['kernel_vs_masked_max_abs']!r} "
          f"local_vs_masked_max_abs={rep['local_vs_masked_max_abs']!r} "
          f"kernel_allclose={rep['kernel_allclose']} local_allclose={rep['local_allclose']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "matrix"
    if cmd == "matrix":
        cmd_matrix()
        cmd_batch_vs_loop()
    elif cmd == "compare":
        cmd_compare()
    else:
        raise SystemExit(f"未知子命令 {cmd!r}（matrix/compare）")
