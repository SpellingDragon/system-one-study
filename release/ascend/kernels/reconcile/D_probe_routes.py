"""D_probe_routes.py — 只读探针：查 `test_sliding_routes_agree_with_mask_reference` 的位等断言到底在比什么、
位等性由什么决定（agent D，red→green 无归因调查）。

零写入、零产品改动：只 import `sys1` 与 torch，按被测口径重跑并拆解。
用法（release 目录下）：
    .venv/bin/python ascend/kernels/reconcile/D_probe_routes.py route      # 确认带状 kernel 路是否真走方言内核
    .venv/bin/python ascend/kernels/reconcile/D_probe_routes.py split      # 逐算子拆解位差来源（matmul/softmax/@V）
    .venv/bin/python ascend/kernels/reconcile/D_probe_routes.py threads    # torch 线程数扫描（同一机器）
    .venv/bin/python ascend/kernels/reconcile/D_probe_routes.py shapebug   # attention_masked 出口形状缺陷示众
    .venv/bin/python ascend/kernels/reconcile/D_probe_routes.py reps N     # 同进程重复 N 次 + 分配扰动
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]      # release/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sys1.kernels import attn_sw_mps, backends
from sys1.layers import attention as LA

SEQ, WIN, CHUNK, HEADS, DIM, SEED = 200, 32, 48, 2, 16, 20261005


def synth():
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    q = torch.randn(HEADS, SEQ, DIM, generator=gen)
    k = torch.randn(HEADS, SEQ, DIM, generator=gen)
    v = torch.randn(HEADS, SEQ, DIM, generator=gen)
    return q, k, v


def ulp_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    """两份 fp32 之间最大的 ULP 距离（把"最后一位"量化成整数，便于说"差几个 bits"）。"""
    ia, ib = a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)
    return int((ia - ib).abs().max().item())


def cmd_route() -> None:
    """带状 route="kernel" 到底走没走 tilelang？逐条闸口示众。"""
    q, k, v = synth()
    band_len = CHUNK + WIN - 1
    qb = q[:, 0:band_len].contiguous()
    kb, vb = k[:, 0:band_len].contiguous(), v[:, 0:band_len].contiguous()
    out = torch.empty(qb.shape, dtype=torch.float32)
    print(f"[env] torch={torch.__version__} threads={torch.get_num_threads()} "
          f"cpu_cap={torch.backends.cpu.get_cpu_capability()} os_cpu={os.cpu_count()} "
          f"device={q.device}")
    act = backends.active_backend(q.device)
    print(f"[gate1] backends.active_backend(cpu)==TILELANG ? {act == backends.TILELANG} (actual={act!r})")
    print(f"[gate2] plan(fp32 输入, band_len={band_len}) -> {attn_sw_mps.plan(qb, kb, vb, out, WIN, DIM ** -0.5)} "
          f"# plan 第一道门槛要求 q3/k3/v3 是 float16（attn_sw_mps.py:165），带状路喂的是 fp32")
    print(f"[gate2'] 即便换成 fp16：plan -> {attn_sw_mps.plan(qb.half(), kb.half(), vb.half(), out, WIN, DIM ** -0.5)} "
          f"# seq={band_len} 不被 BLOCK_Q=16 整除 → 仍 None")
    rep = LA.compare_routes(SEQ, WIN, heads=HEADS, dim=DIM, chunk=CHUNK, seed=SEED)
    print(f"[result] kernel_vs_masked_max_abs={rep['kernel_vs_masked_max_abs']!r} "
          f"kernel_allclose={rep['kernel_allclose']} local_vs_masked_max_abs={rep['local_vs_masked_max_abs']!r}")
    print("[conclusion] 本用例的带状『kernel 路』从未触达 tilelang/MSL 内核，实际执行体是 attn_sw_kernel._eager()")


def cmd_split() -> None:
    """逐算子定位位差：同一批 (i,j) 分数、同一份概率、同一份 V，看位差在哪一步产生。"""
    q, k, v = synth()
    ref = LA.attention_masked(q, k, v, window=WIN)
    got = LA.attention_band(q, k, v, WIN, chunk=CHUNK, route="kernel")
    print(f"[out] max_abs={(got.out - ref.out).abs().max().item()!r} "
          f"ndiff={int((got.out != ref.out).sum().item())}/{got.out.numel()} ulp={ulp_diff(got.out, ref.out)}")

    score_full = q @ k.transpose(-1, -2) * (DIM ** -0.5)          # 掩码路的 matmul（M=200,N=200）
    for start in (0, 48, 96):
        stop = min(SEQ, start + CHUNK)
        bs = max(0, start - (WIN - 1))
        band_len = stop - bs
        qb, kb, vb = q[:, bs:stop], k[:, bs:stop], v[:, bs:stop]
        score_band = qb @ kb.transpose(-1, -2) * (DIM ** -0.5)    # 带状路的 matmul（M=band_len,N=band_len）
        rows = slice(start, stop)
        s_band_tail = score_band[:, -(stop - start):, :]           # 带尾 chunk 行，列 = 全局 bs..stop
        s_mask_win = score_full[:, rows, bs:stop]                  # 同一批 (i,j) 对，来自整张矩阵
        n_ne = int((s_band_tail != s_mask_win).sum().item())
        print(f"  [chunk start={start} band_len={band_len}] 同一批(i,j)分数：位不等={n_ne}/"
              f"{s_band_tail.numel()} max_abs={(s_band_tail - s_mask_win).abs().max().item()!r} "
              f"ulp={ulp_diff(s_band_tail, s_mask_win)}")

        # softmax：带状只在带内归一，掩码路在 200 列（窗外为 -inf→exp 0）上归一
        idx = torch.arange(band_len)
        d = idx[:, None] - idx[None, :]
        pb = torch.softmax(score_band.masked_fill(~((d >= 0) & (d < WIN)), float("-inf")), dim=-1)[:, -(stop - start):, :]
        band3 = LA.window_band(SEQ, WIN).squeeze(0)          # (1,SEQ,SEQ)→与 score_full 同秩，避免 masked_fill 广播出多余前导维
        pm = torch.softmax(score_full.masked_fill(~band3, float("-inf")), dim=-1)
        p_common = pm[:, rows, bs:stop]
        print(f"    概率 P：位不等={int((pb != p_common).sum().item())}/{pb.numel()} "
              f"max_abs={(pb - p_common).abs().max().item()!r}")
        # 喂同一份概率，只比 P@V 这一步（带状 K=band_len，掩码 K=200）
        o_band = p_common @ vb
        o_mask = pm[:, rows] @ v
        print(f"    同一概率 @V：位不等={int((o_band != o_mask).sum().item())}/{o_mask.numel()} "
              f"max_abs={(o_band - o_mask).abs().max().item()!r}")


def cmd_threads() -> None:
    """线程数扫描：位等是否与 intra-op 线程数绑定（若是→随 runner 核数/cgroup 翻转，属环境敏感）。"""
    print(f"  [env] torch={torch.__version__} cpu_cap={torch.backends.cpu.get_cpu_capability()} os_cpu={os.cpu_count()}")
    for n in (1, 2, 3, 4, 6, 8, 12, 16):
        torch.set_num_threads(n)
        rep = LA.compare_routes(SEQ, WIN, heads=HEADS, dim=DIM, chunk=CHUNK, seed=SEED)
        eq = rep["kernel_vs_masked_max_abs"] == 0.0
        print(f"  threads={torch.get_num_threads():>2} kernel_vs_masked_max_abs="
              f"{rep['kernel_vs_masked_max_abs']!r:>12} bitwise_eq={eq} allclose={rep['kernel_allclose']}")


def cmd_reps(n: int) -> None:
    """同进程重复 n 次 + 分配扰动：区分『同进程可复现』与『跨堆态稳定』。"""
    base = LA.compare_routes(SEQ, WIN, heads=HEADS, dim=DIM, chunk=CHUNK, seed=SEED)["kernel_vs_masked_max_abs"]
    print(f"  baseline max_abs={base!r}")
    for i in range(n):
        junk = torch.randn(3, 517, 733) @ torch.randn(3, 733, 611)   # 扰动分配历史/指针对齐
        del junk
        got = LA.compare_routes(SEQ, WIN, heads=HEADS, dim=DIM, chunk=CHUNK, seed=SEED)["kernel_vs_masked_max_abs"]
        print(f"  rep{i:<2} perturbed max_abs={got!r} eq={got == 0.0}")


def cmd_shapes() -> None:
    """形状扫描：位等到底是『结构性保证』还是『这一组形状碰巧』？在同机换形状看是否破。"""
    combos = [
        (200, 32, 48),   # 被测口径
        (200, 32, 16), (200, 32, 64), (200, 16, 48), (200, 64, 48), (200, 128, 48),
        (96, 32, 48), (128, 32, 48), (160, 32, 48), (240, 32, 48), (256, 32, 48),
        (300, 32, 48), (200, 32, 24), (200, 32, 32), (200, 32, 50), (200, 33, 48),
        (128, 16, 32), (144, 24, 40), (210, 40, 56), (256, 16, 64),
    ]
    broken = 0
    print(f"  [env] torch={torch.__version__} threads={torch.get_num_threads()} "
          f"cpu_cap={torch.backends.cpu.get_cpu_capability()}  # 同机、同 seed、只换形状")
    for seq, win, chunk in combos:
        rep = LA.compare_routes(seq, win, heads=HEADS, dim=DIM, chunk=chunk, seed=SEED)
        ke = rep["kernel_vs_masked_max_abs"]
        le = rep["local_vs_masked_max_abs"]
        eq = (ke == 0.0)
        broken += 0 if eq else 1
        flag = "" if eq else "   <-- 位等破"
        print(f"  seq={seq:>3} win={win:>3} chunk={chunk:>3} | kernel_vs_masked={ke!r:>22} "
              f"eq={eq!s:<5} | local_vs_masked={le!r:>22}{flag}")
    print(f"  [summary] {broken}/{len(combos)} 组形状上『带状 kernel 路 vs 掩码路』位等**不成立**"
          f"（同一台机器、同一个 BLAS、同 seed，只改 seq/window/chunk）")


def cmd_shapebug() -> None:
    """附带缺陷示众：attention_masked 出口形状违背自身契约（3 维入→4 维出；4 维入→崩）。"""
    q, k, v = synth()
    m = LA.attention_masked(q, k, v, window=WIN)
    b = LA.attention_band(q, k, v, WIN, chunk=CHUNK, route="kernel")
    print(f"  attention_masked.out   = {tuple(m.out.shape)}  # AttnOut 声明 (heads,seq,dim)")
    print(f"  attention_band.out     = {tuple(b.out.shape)}")
    print(f"  相减广播后             = {tuple((b.out - m.out).shape)}  # 用例实为 4 维 vs 3 维广播相减")
    q4 = q[None].contiguous()
    try:
        m4 = LA.attention_masked(q4, q4.clone(), q4.clone(), window=WIN)
        print(f"  4 维入参             = {tuple(m4.out.shape)}（期望 (1,2,200,16)）")
    except RuntimeError as e:
        print(f"  4 维入参             = RuntimeError: {e}  # _flatten_heads 明写支持 (B,heads,seq,dim)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "route"
    if cmd == "route":
        cmd_route()
    elif cmd == "split":
        cmd_split()
    elif cmd == "threads":
        cmd_threads()
    elif cmd == "shapebug":
        cmd_shapebug()
    elif cmd == "shapes":
        cmd_shapes()
    elif cmd == "reps":
        cmd_reps(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    else:
        raise SystemExit(f"未知子命令 {cmd!r}（route/split/threads/shapes/shapebug/reps）")
