#!/usr/bin/env python3
"""reconcile/S_perf_track_cube.py — Cube 形态**存档**（performance-track 参照，非产品、无人 import）。

【这是什么】D-cube1 两轨制裁决后，P1-1f 把 `gemm_asc.py` / `gemm_bwd_dw_asc.py` 的 ascend 正文
改道成可移植标量体（correctness 轨）。改道前 K 波那两份 T.Cube 形态（P1-4 合流成果）整段拷进本
文件留档，供 **performance-track**（§12 与官方 dav_c220 Matmul 构造链逐参对照的 intrinsic 考古）
继续推进时作 DSL 骨架参照。本文件**不被任何产品代码 import**，也不参与 `compile_all_asc.py`；
只在想复现 Cube 形态编译判决时手动调用下面两个工厂函数（需容器 target=ascend）。

【为什么留档而不是删掉】Cube 形态的编译面已经绿过（不是白写的）：
  * `gemm_asc_impl`（Q1=UB-direct：操作数/累加全落 UB，主乘加 `T.Cube`+`T.gemm(transpose_B)`，
    收尾标量 `T.serial`）——代理 K 在容器判到 `K-GEMM ASC-COMPILE-PASS`（4 例 f16/f32×relu/none
    全驱动，keys=`ascend|gemm[ascend]|n64k64b16x64x64{arelu,anone}`，无 SimtVF，.o≈2182B）。
  * `dw_asc_impl`（l0tr 主案：转置下移到 L1→L0 装填，mad 恒吃 NT，L0C→GM 直出）——P1-4b 由
    §12 补 `asc_fill_l1`→`set_l1_2d` 后转 PASS（代理 L，编排者容器复验 9/9）。
  即：**两件都能编出 .o**，卡上却双双 aicore exception **507015**（DMA 非法访问）。

【真机判决（wave2 2026-10-10 run oncard-wave2，本文件的"待解问题清单"）】
  * P1-1d 的 `asc_copy_l12l0a/b` **9 参位序**修复（sid 回穿 + transpose 位补齐，trunk 971 行）
    已证"必要不充分"——修完 gemm_l1 / dW 仍 507015。
  * 唯余未证真的参数（P1-1f 因此改道，D-cube1）：
      - V1/V2：L1→L0 装填的 **stride 单位**（V2 元素 vs 块）——官方 `LoadData2DL12L0ACal` 的
        `srcStride/dstGap` 到底按哪个单位给，trunk 与官方 mm_impl.h 说法不一致；
      - V3：GM→L1 的 **分形（NZ）布局** `padFuncMode`——L1 侧是否需要 nd2nz 分形 pad 才能喂 mad；
      - V4/V5/V6（P1-4b 登记）：`asc_fill_l1` 的 value 位段（非 0 填充才暴露）、dst_gap 拆趟、
        repeat 步距；
      - fp32 操作数是否需 `set_hf32_mode`（dW 云端待验项，K 波 docstring 已挂）。
  * 另：`FixpipeL0C2UBImpl = assert(false)`（port910b 卷宗 b10，910B 硬件 L0C 搬不回 UB）——
    这是 Cube 形态收尾只能做标量/直出 GM、不能在 UB 二次重排的根因，两份正文都据此摆。

【与 correctness 轨的接口关系】两份工厂函数签名与 K 波当时的 `gemm_asc_impl` / `dw_asc_impl`
**逐字一致**（`(A, W, bias, C, n, k, bm, bn, bk, act_mode)` / `(dY, A, GW, n, k, tc, bn, bk)`，
返回追踪好的 PrimFunc），所以考古时可以直接把它们塞回接口做编译判决：
`tilelang.compile(gemm_cube_impl(A, W, b, C, **kwargs), target="ascend", out_idx=[])`。
"""
import os
import sys

#: 与 K 波接口 home 同一组常量（存档用，不改口径）
BLOCK_M = 16
BLOCK_TC = 16
BLOCK_LADDER = (64, 32, 16)
NUM_BLOCKS = 8
VEC_THREADS = 64
NUM_STAGES = 2


def gemm_cube_impl(A, W, bias, C, n: int, k: int, bm: int, bn: int, bk: int, act_mode: int):
    """【存档】K 波 gemm_asc.gemm_asc_impl 逐字原样（Q1=UB-direct Cube 形态）。

    910B 方言纪律（当时的取证）：操作数与累加都落 UB(shared)——L0C 搬不回 UB，故收尾标量段
    只能在 UB 上做；首轮靠 `clear_accum=ko==0` 清零（fp32 的 `T.clear` 在 UB 上生成坏 float2
    store）；网格是常量 NUM_BLOCKS，动态行轴 m 落在 `T.serial` 行块循环上界；尾块交给 T.copy
    自动边界谓词（**卡上是否真夹住地址 = V 系待证项之一，correctness 轨已改成夹址标量填装**）。

    :return: 追踪好的 PrimFunc（`tilelang.compile(target="ascend")` 已证可编出 .o）。
    """
    import tilelang.ascend.language as T

    m = T.dynamic("m")
    nb = n // bn

    @T.prim_func
    def gemm_impl(A: T.Tensor((m, k), "float32"), W: T.Tensor((n, k), "float32"),
                  bias: T.Tensor((n,), "float32"), C: T.Tensor((m, n), "float32")):
        """被追踪的那一层：网格固定为若干个立方体核块，行数 m 是活的。

        白话：同一本账换到立方体单元上摆——两个操作数和累加都放公共小格子，一次搬多宽、
        竖着切几列都定死，有多少行都不影响切法；末尾不够一整块的部分交给搬运自带的那道闸
        拦住（越界写回被谓词挡下），不需手写守卫。
        """
        with T.Kernel(NUM_BLOCKS) as bx:
            a = T.alloc_shared((bm, bk), "float32")
            w = T.alloc_shared((bn, bk), "float32")
            cl = T.alloc_shared((bm, bn), "float32")
            with T.Cube():
                for chunk in T.serial(T.ceildiv(m, bm * NUM_BLOCKS)):
                    my = chunk * NUM_BLOCKS + bx
                    for mx in T.serial(nb):
                        # K 分块连着乘加，首轮 clear_accum 清零，后续续在同一个 UB 格子里
                        for ko in T.serial(T.ceildiv(k, bk)):
                            T.copy(A[my * bm, ko * bk], a)
                            T.copy(W[mx * bn, ko * bk], w)
                            T.gemm(a, w, cl, transpose_B=True, clear_accum=ko == 0)
                        # 收尾标量化：加偏置、按需压负为零（910B 无 SimtVF/T.Parallel）
                        for i in T.serial(bm):
                            for j in T.serial(bn):
                                v = cl[i, j] + bias[mx * bn + j]
                                if act_mode == 1:
                                    if v < 0:
                                        v = 0.0
                                cl[i, j] = v
                        T.copy(cl, C[my * bm, mx * bn])

    return gemm_impl


def dw_l0tr_cube_impl(dY, A, GW, n: int, k: int, tc: int, bn: int, bk: int):
    """【存档】K 波 gemm_bwd_dw_asc.dw_asc_impl 逐字原样（l0tr 主案：转置下移到 L1→L0 装填）。

    口径要点（当时的取证，考古时别丢）：
      * mad 只认"左不翻、右翻"（NT）——`gemm_mad.py` 对 shared.l1 路直接 assert
        `not trans_A and trans_B`，而 dW 天然 trans_A（TN），故把转置下移到 L1→L0 装填
        （`T.copy(dy_l1, a_l0, transpose=True)`），累加器与出口同朝向。
      * L0C→GM 直出（A-4 已落 `asc_copy_l0c2gm`），不在 UB 二次重排（硬件 fixpipe 不支持）。
      * 被否方案（别重走）：950 载体 SimtVF 转置写回=COMPILE-FAIL；madta（GM→L0 无 DMA 面，
        codegen 退化标量解引用 `__ca__`，命中 `only __ubuf__/__gm__/local can be dereferenced`，
        卷宗 G-D2）；ntt（`asc_copy_gm2l1_dn2nz` 只是 nd2nz 同名别名 stub，编译绿≠数值绿）。
      * 动态 m 末 token 块不满 tc 时 codegen 会发 `asc_fill_l1`（§12，P1-4b 已由
        `set_l1_2d` 补齐，位段 V4/V5/V6 待卡证）。

    :return: 追踪好的 PrimFunc（target=ascend 已证可编；卡上 507015 待 V 系裁决）。
    """
    import tilelang.ascend.language as T

    m = T.dynamic("m")
    nb, nk = n // bn, k // bk
    tiles = nb * nk
    blocks = min(tiles, NUM_BLOCKS)
    iters = (tiles + blocks - 1) // blocks

    @T.prim_func
    def dw_impl(dY: T.Tensor((m, n), "float32"), A: T.Tensor((m, k), "float32"),
                GW: T.Tensor((n, k), "float32")):
        """被追踪的那一层：出口表形状是常量，行方向串行累加、列方向按块摊开。"""
        with T.Kernel(blocks) as bid:
            dy_l1 = T.alloc_l1((tc, bn), "float32")   # L1 自然 (token, n)
            x_l1 = T.alloc_l1((tc, bk), "float32")    # L1 自然 (token, k)
            a_l0 = T.alloc_l0a((bn, tc), "float32")   # NT 要求 (M=BN, K=TC)
            b_l0 = T.alloc_l0b((bk, tc), "float32")   # NT 要求 (N=BK, K=TC)
            acc = T.alloc_l0c((bn, bk), "float32")
            with T.Cube():
                for it in T.serial(iters):
                    tid = bid + it * blocks
                    n_tile = tid // nk
                    k_tile = tid % nk
                    # 归约轴 = 动态 token 数 m；末块不满 tc 触发 §12 asc_fill_l1
                    for kt in T.serial(T.ceildiv(m, tc)):
                        T.copy(dY[kt * tc, n_tile * bn], dy_l1)
                        T.copy(A[kt * tc, k_tile * bk], x_l1)
                        T.copy(dy_l1, a_l0, transpose=True)   # 转置下移到 L1→L0 装填
                        T.copy(x_l1, b_l0, transpose=True)
                        T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=kt == 0)
                    T.copy(acc, GW[n_tile * bn, k_tile * bk])  # L0C→GM 直出，出口自然朝向

    return dw_impl


if __name__ == "__main__":
    # 自检：在容器里可独立复现"存档形态仍然可编"（不判数值——数值正是它卡住的地方）。
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    import torch
    import tilelang

    os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
    m, n, k = 32, 64, 64
    A = torch.randn(m, k, dtype=torch.float32)
    W = torch.randn(n, k, dtype=torch.float32)
    b = torch.randn(n, dtype=torch.float32)
    C = torch.empty(m, n, dtype=torch.float32)
    dY = torch.randn(m, n, dtype=torch.float32)
    GW = torch.empty(n, k, dtype=torch.float32)
    builds = (
        ("gemm-cube", lambda: gemm_cube_impl(A, W, b, C, n=n, k=k, bm=BLOCK_M, bn=64, bk=64,
                                            act_mode=1)),
        ("dw-l0tr", lambda: dw_l0tr_cube_impl(dY, A, GW, n=n, k=k, tc=BLOCK_TC, bn=64, bk=64)),
    )
    rc = 0
    for tag, build in builds:
        try:
            # 追溯与编译都在 try 内：本机没注册 ascend tileop 时（host venv）追溯阶段就抛，
            # 那是环境缺件、不是存档件缺陷，必须与"编不出"分开报，不冒充 PASS。
            prog = build()
            tilelang.compile(prog, target="ascend", out_idx=[])
            print(f"S-PERF-TRACK ARCHIVE-COMPILE-PASS[{tag}]（Cube 形态仍可编，数值待 V 系裁决）")
        except Exception as exc:  # noqa: BLE001  存档件编不出也要如实报
            head = str(exc).splitlines()[0][:200] if str(exc) else repr(exc)[:200]
            if "is not registered" in head:
                print(f"S-PERF-TRACK ARCHIVE-SKIP(环境)[{tag}] {head}")
            else:
                rc = 1
                print(f"S-PERF-TRACK ARCHIVE-COMPILE-FAIL[{tag}] {head}")
    print("S-PERF-TRACK ARCHIVE-OK（形态可编性以容器为准；本件不被产品代码 import）")
    sys.exit(rc)
