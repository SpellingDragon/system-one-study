#!/usr/bin/env python3
"""reconcile/U2_probe_k48_dma_semantics.py — P1-1q 的**装填语义直读**探针（下窗一条命令级，不扩案）。

【判什么】`linear_prod` 在 k=48 真机错值/NaN，而 host 两轨（cpu 方言直调正文 + cpu-alias 产品
路由）**均不复现**（判据见 U2_matrix_k48.md）。归档生成码把嫌疑收到唯一一处：`bk < k` 时 W 块
编成**多突发跨步 GM→UB DMA**（k=48 实发 `…, 32, 64, 0,0,0, mode, 192, 64`），而所有 PASS 档
（k=16/32/64）`bk == k` ⇒ tilelang 的 `plan_single_row` 把它折叠成**单发连续**、步长参数根本不
参与（`src/ascend/op/copy.cc` 的 plan_single_row；对照生成码 5d732a8f vs 77a8a3c4 见
U2_K48_EVIDENCE.md §2）。而 910B 原生件 `copy_gm_to_ubuf` 的 stride 究竟是
**pitch（相邻块首地址间距）还是 gap（块间空隙）**，本仓只登记过单位（32B）却从未数值证真
（`attempts/C/compat_gap_C.md` G-C4/G-C5："定性有据，数值仍待上卡证真"）。

【手法——语义无关，不靠猜】把 W 编码成"每个元素自带门牌号"：`W[i][j] = i*1000 + j`；
A 取 one-hot（第 p 行只在第 p 列放 1，p=0..bk-1）⇒ 出口 `C[p][n]` **恰好等于**"被装进 UB 槽
(行=n, 列=p) 的那个 GM 元素的门牌号"。一次发射即**量出 DMA 实际的地址映射图**，与候选语义
的预测图对拍：谁的失配最少，硬件就是谁。越界/stale 槽拿不到合法门牌号 ⇒ 单列为
`UNATTRIBUTED`（这正是"没报 507015 却出 NaN"该看的量）。

【候选语义】(srcPitchBytes, dstPitchBytes)，codegen 本意是 (k*4, bk*4)：
    P  (192, 64) 双 pitch —— codegen 与 compat §10b-GAPB 的**假设**（正确解）
    C  ( 64, 64) srcStride 被忽略 ⇒ 连续读
    G  (256,128) 双 gap    ⇒ 原生件把 stride 当"块间空隙"（= pitch + blockLen）
    Gs (256, 64) 只源侧 gap        Gd (192,128) 只目的侧 gap
    Z  (192,  0) dstStride 被忽略 ⇒ 所有块叠在同一处
gap 类会读越界（k=48 读上界 20416B > 张量 18432B，见 U2_probe_k48_host.py T4）⇒ 预期表现为
UNATTRIBUTED/NaN；dstStride 类会写出 w_ub 之外踩到 c_ub ⇒ 预期表现为门牌号错位。

【两轨跑法（判决分档）】
    host 自测（校验解码器 + 正文数学，应全 OK）：
        cd release && .venv/bin/python ascend/kernels/reconcile/U2_probe_k48_dma_semantics.py --dev cpu
    下窗卡上直判（一条命令，单进程）：
        python3 ascend/kernels/reconcile/U2_probe_k48_dma_semantics.py --dev npu
【形状档】n=96, k ∈ {48(跨步, 病档), 96(跨步, dA 必经档), 32(单发连续, ctrl)}；act=none（不让
截断干扰门牌号）；m=16（= BLOCK_M，one-hot 探针数正好铺满 bk 列）。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))            # .../release
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

KS = (48, 96, 32)
LADDER = (64, 32, 16)


def largest_block(v: int) -> int:
    for c in LADDER:
        if v % c == 0:
            return c
    raise SystemExit(f"n/k={v} 不在 BLOCK_LADDER 表达窗内")


def encode_W(n: int, k: int) -> np.ndarray:
    """W[i][j] = i*1000 + j —— 每个元素自带"我在 GM 里第几行第几列"的门牌号。"""
    i = np.arange(n, dtype=np.float64)[:, None]
    j = np.arange(k, dtype=np.float64)[None, :]
    return i * 1000.0 + j


def onehot_A(m: int, k: int, bk: int) -> np.ndarray:
    """第 p 行只在第 p 列放 1（p < bk）⇒ C[p][n] 直接读出 UB 槽 (n, p) 的来源门牌号。"""
    a = np.zeros((m, k), dtype=np.float64)
    for p in range(min(m, bk)):
        a[p, p] = 1.0
    return a


def decode(C: np.ndarray, n: int, k: int, nb: int, bn: int, bk: int) -> np.ndarray:
    """门牌号 → 测得的源元素 flat 下标；非法/越界/stale → NaN。返回 (min(m,bk), n) 的源行号矩阵。"""
    probes = min(C.shape[0], bk)
    src_row = np.full((probes, n), np.nan, dtype=np.float64)
    src_col = np.full((probes, n), np.nan, dtype=np.float64)
    for p in range(probes):
        for ix in range(n):
            v = float(C[p, ix])
            if not np.isfinite(v):
                continue
            e = int(round(v))
            r, c = e // 1000, e % 1000
            if 0 <= r < n and 0 <= c < k:
                src_row[p, ix], src_col[p, ix] = r, c
    return src_row, src_col


def predict(nb: int, n: int, k: int, bn: int, bk: int, src_pitch_b: int,
            dst_pitch_b: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """按给定步长语义算"UB 槽 (j_1, c) ← GM 元素 flat"的预测图（纯地址算术）。

    返回 (源行图, 源列图, 读越界元素数, 写出 w_ub 之外的块数)——后两者分别对应
    "读到 stale/GM 外（NaN/错值的来源）"与"踩到 c_ub（累加区被 W 数据覆盖）"。
    """
    sp = src_pitch_b // 4 if src_pitch_b % 4 == 0 else -1
    dp = dst_pitch_b // 4 if dst_pitch_b % 4 == 0 else -1
    nb_out = dst_oob = 0
    src_row = np.full((bk, n), np.nan, dtype=np.float64)
    src_col = np.full((bk, n), np.nan, dtype=np.float64)
    span = n * k
    for mx in range(nb):
        base = mx * bn * k                       # ko=0 时的块起点（元素）
        for r in range(bn):                       # DMA 块序
            for c in range(bk):
                sflat = base + r * sp + c
                dslot = r * dp + c                # UB 内元素槽
                j_1, kk = dslot // bk, dslot % bk
                if j_1 >= bn:                     # 写在 w_ub 之外 ⇒ 踩 c_ub/a_ub
                    dst_oob += 1
                    continue
                if sflat < 0 or sflat >= span:
                    nb_out += 1
                    continue
                src_row[kk, mx * bn + j_1] = sflat // k
                src_col[kk, mx * bn + j_1] = sflat % k
    return src_row, src_col, nb_out, dst_oob


def run_case(m, n, k, dev, act="none"):
    """发射一次，返回测得图 + 各候选失配。"""
    bn, bk = largest_block(n), largest_block(k)
    nb = n // bn
    Wn = encode_W(n, k)
    An = onehot_A(m, k, bk)
    W = torch.from_numpy(Wn).float()
    A = torch.from_numpy(An).float()
    bias = torch.zeros(n, dtype=torch.float32)
    if dev == "npu":
        # ★ 故意不走 gemm_kernel.forward()：plan() 的选档在 P1-1p 在途提案里已改成"整行连续优先"
        #   （`_pick_bk_ascend`；本窗 21:13 后实测 bn=32 时 k∈{16..96} 全给 bk==k ⇒ 恒 SINGLE），
        #   那样本探针**再也编不出多突发跨步 DMA**，判别力直接归零（把"未证真"变成"测不到"）。
        #   这里显式钉住 ladder bk，经 `gemm_asc.run` 走与产品**同一** impl/compile/搬运参数路径，
        #   只是绕开选档策略——不改正产品件一个字。
        from ascend.kernels import gemm_asc
        A, W, bias = A.npu(), W.npu(), bias.npu()
        C = torch.zeros(m, n, dtype=torch.float32).npu()
        spec = {
            "target": "ascend",
            "key": (f"gemm[ascend]|U2PIN-n{n}k{k}b{gemm_asc.BLOCK_M}x{bn}x{bk}a{act}dfloat32"),
            "kwargs": {"n": n, "k": k, "bm": gemm_asc.BLOCK_M, "bn": bn, "bk": bk,
                       "act_mode": gemm_asc.ACT_MODE[act], "idt": "float32", "odt": "float32"},
            "impl": gemm_asc.gemm_asc_impl,
            "shape": (n, k),
        }
        if not gemm_asc.run(A, W, bias, C, spec):
            raise SystemExit("U2-PROBE-ABORT ascend 轨编译失败（强钉 bk=%d）：%s" % (bk, spec["key"]))
        torch.npu.synchronize()
        got_t = C
        print("U2-BK-POLICY dev=npu 强钉 bk=%d（ladder 值，不经 plan；"
              "plan 若选整行连续则产品面不可达此形态，本探针仍可判）" % bk)
    else:
        import tilelang
        from ascend.kernels import ascend_env, gemm_asc    # cpu 方言直调同一正文生成器
        C = torch.zeros(m, n, dtype=torch.float32)
        prog = gemm_asc.gemm_scalar_body(ascend_env.dialect("cpu"), A, W, bias, C,
                                         n=n, k=k, bm=gemm_asc.BLOCK_M, bn=bn, bk=bk,
                                         act_mode=gemm_asc.ACT_MODE[act])
        kern = tilelang.compile(prog, out_idx=[], **ascend_env.compile_kwargs("cpu"))
        kern(A, W, bias, C)
    got = (C if dev != "npu" else got_t).detach().float().cpu().numpy().astype(np.float64)
    return bn, bk, nb, got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cpu", choices=("cpu", "npu"))
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--ks", default=",".join(str(x) for x in KS))
    ap.add_argument("--acts", default="none,relu",
                    help="W 编码恒非负 ⇒ relu 对本探针的门牌号是恒等变换；两 act 各编一枚内核，"
                         "差异只可能来自 codegen 变体（真机可据此判 NaN 是否来自 act 侧）")
    a = ap.parse_args()
    os.environ.setdefault("DMLAYA_ASCEND_TARGET", "ascend")
    if a.dev == "npu" and hasattr(torch, "npu"):
        torch.npu.init()
    from ascend.kernels import gemm_asc  # noqa: F401  (触发方言一致性，与产品同环境)

    # 候选语义按 (k, bk) 参数化（元素单位）。"单位=32B 块"这一条已被 X 取证关掉
    # （attempts/X_FRACTAL_EVIDENCE.md §0 裁决③：DEFAULT_C0_SIZE=32 + 地址模型
    #  kernel_check_data_copy_overflow.h:505-520），故此处不再列"字节当块"候选，
    #  只判真正未定论的那一轴：**pitch（首地址间距）vs gap（块间空隙）vs 被忽略**。
    def cands_for(k, bk):
        return [("P(pitch,pitch)", k, bk), ("C(srcStride被忽略)", bk, bk),
                ("G(双gap)", k + bk, 2 * bk), ("Gs(源gap)", k + bk, bk),
                ("Gd(目的gap)", k, 2 * bk), ("Z(dstStride被忽略)", k, 0)]
    print("U2-DMA-READOUT dev=%s m=%d n=%d ks=%s | W[i][j]=i*1000+j, A=onehot ⇒ C[p][n]=门牌号"
          % (a.dev, a.m, a.n, a.ks.split(",")), flush=True)
    bad = 0
    for k in [int(x) for x in a.ks.split(",")]:
      for act in a.acts.split(","):
        bn, bk, nb, got = run_case(a.m, a.n, k, a.dev, act)
        mrow, mcol = decode(got, a.n, k, nb, bn, bk)
        probes = mrow.shape[0]
        expect = np.tile(np.arange(a.n, dtype=np.float64), (probes, 1))
        unattr = int((~np.isfinite(mrow)).sum())
        mism = int(((mrow != expect) & np.isfinite(mrow)).sum())
        exp_col = np.arange(probes, dtype=np.float64)[:, None]
        colm = int(((mcol != exp_col) & np.isfinite(mrow)).sum())
        print("U2-MEASURE k=%-3d/%-4s bn=%-2d bk=%-2d nb=%d nk=%d dma=%-8s probes=%d×n%d | "
              "OK=%d 行错位=%d 列错位=%d UNATTRIBUTED(越界/stale)=%d" % (
                  k, act, bn, bk, nb, k // bk,
                  "strided" if bk < k else "single", probes, a.n,
                  int((np.isfinite(mrow) & (mrow == expect)).sum()), mism, colm, unattr),
              flush=True)
        scores = []
        for tag, sp_e, dp_e in cands_for(k, bk):
            pr, pc, oob, doob = predict(nb, a.n, k, bn, bk, sp_e * 4, dp_e * 4)
            take = pr[:probes] if pr.shape[0] >= probes else pr
            mm = int(((take != mrow) & np.isfinite(mrow) & np.isfinite(take)).sum())
            nn = int((np.isfinite(take) & ~np.isfinite(mrow)).sum())
            scores.append((tag, mm + nn, oob, doob))
        scores.sort(key=lambda s: s[1])
        print("U2-CANDIDATE k=%-3d/%-4s 失配(升序)：%s" % (
            k, act, "  ".join("%s mismatch=%d 读越界=%d 写越界块=%d" % s for s in scores)),
            flush=True)
        same = [t for t, mm, _o, _d in scores if mm == 0 and not t.startswith("P")]
        print("U2-DISCRIMINATE k=%-3d/%-4s 与 P 同解的其它候选=%s ⇒ 判别力=%s" % (
            k, act, same or "无", "弱（该档不能区分语义，须靠跨步档）" if same else "强"),
            flush=True)
        best = scores[0]
        if a.dev == "cpu":
            # host 自测：解码器必须认账 P（c 后端按 pitch 语义正确装填），且零 UNATTRIBUTED
            ok = (best[0].startswith("P") and best[1] == 0 and unattr == 0)
            bad += 0 if ok else 1
            print("U2-READOUT-VERDICT k=%-3d/%-4s %s（host 轨期望：P 且全 OK）" % (
                k, act, "SELFTEST-OK" if ok else "SELFTEST-FAIL"), flush=True)
        else:
            print("U2-READOUT-VERDICT k=%-3d/%-4s 硬件实际装填语义 = %s（mismatch=%d）；"
                  "若非 P ⇒ 跨步 DMA 语义错位坐实（compat 不在白名单，须上报裁决）" % (
                      k, act, best[0], best[1]))
            bad += 0 if best[0].startswith("P") and best[1] == 0 else 1
    print("U2-DMA-READOUT-DONE unexpected=%d（dev=%s）" % (bad, a.dev), flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
