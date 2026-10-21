"""gemm 方言件：C = act(A @ Wᵀ + bias)，行数 m 是动态符号，一次编译服务任意批大小。

【做什么】同一套乘加口径写两份方言正文：昇腾那份与 CPU 那份现在是**同一个标量正文生成器**
（`gemm_scalar_body`）在两种方言下各追一遍——语言模块由 `ascend_env.dialect(target)` 注入，正文
只用两方言公共子集（`T.serial` 循环 + `T.copy`(GM→UB) + 标量 load/store + 整数索引算术），
不含任何立方体（Cube/mad/T.gemm）或 SIMT 向量构造。两份都由 plan() 判形状、run() 负责取编译
产物并就地写出口。位宽口径（P1-1m dtype 矩阵）：CPU 那份仍恒 fp32 出口；昇腾那份按实际位宽
   参数化（`idt`/`odt`）——cast 只发生在**标量 load/store** 面，DMA 装填缓冲恒与源张量同宽，因为
   昇腾搬运不允许顺带转类型（ascend/analysis/vf_checker：“DMA copies cannot perform type casting”）。
【怎么做】① 件形是"张量参数在前、整数常量在后"的工厂函数；run() 现场调用工厂拿到追踪好的
   PrimFunc，再交给 tilelang.compile 出产物并按形状键缓存。刻意**不用 tilelang.jit**：这种写法
   属 lazy 风格，jit(...) 只回 Kernel 对象不执行（实测出口保持全零），而它的 call-form 缓存拿
   张量做 `==` 比较上次调用（jit/__init__.py 的 _CallFormCache._matches_last），换一批新张量就
   RuntimeError。② 本文件不写 `from __future__ import annotations`——tilelang 取 prim_func 的
   张量注解时只认闭包非局部名（eager/builder.py 的 get_func_nonlocals），一旦 PEP 563 把注解变成
   字符串，只在注解里出现的形状常量就没有闭包单元，编译期报 NameError: name 'n' is not defined。
   ③ 动态维 m 的去处：两方言同一条纪律——**网格上界必须是编译期常量**（CPU c 后端把 m 写进网格
   会一次都不发射，实测 C 全零、误差恰好等于 |A@Wᵀ| 最大值；昇腾 `T.Kernel` 是 1-D 核块网格，
   同样只给常量），m 一律落在 `T.serial(T.ceildiv(m, bm))` 的行块循环上界上。④ 收尾（加偏置、
   按需压负为零）在 UB/local 累加块上逐元素做，出口走**逐元素标量落全局 + 运行时行号守卫**，
   不整块写回（见 ⑤）。⑤ 尾块口径从"交给 T.copy 的自动边界谓词"改成**夹址标量填装**：行块不满
   bm 时，A 的每一行先按 `T.min(row, m-1)` 把地址夹进合法区间、再用 `T.if_then_else(row < m, 读值,
   0.0)` 决定取不取（`if_then_else` 两侧都会求值，所以地址必须先夹，否则就是越界读）；写回同样按
   行号守卫。这一改是 correctness 轨的核心动作——越界 DMA 读在卡上的表现正是 aicore exception
   507015（DMA 非法访问），而本地 CPU 轨能真跑验证这条路径。
【为什么】五处口径是被约束逼出来的，不是风格：(a) 权重按 (N,K) 存、标量三重循环天然吃这个方向
   （`C[i,j] += Σ_k A[i,k]·W[j,k]` 就是 transpose_B 的算式本体），与 sys1/`gemm_ref` 同式；(b)
   出口不在方言里降位宽——昇腾 DMA 搬运不允许顺带转类型（tilelang/ascend/analysis/vf_checker.py
   明确 "DMA copies cannot perform type casting"），CPU 侧实测把 fp32 块直接 copy 成 fp16 全局会撞
   生成代码里的 `no matching conversion for C-style cast from 'float4' to 'half4'`，故两边统一 fp32
   出口、入口层降位；(c) 被否方案：CPU 侧用 T.Parallel 收尾——实测本方言的并行块索引 local 缓存
   会被语义检查拒绝（"Local buffer ... is thread-private"），serial 才成立；(d) 被否方案：把 m 写进
   网格——Metal 认动态网格，c 后端不认，照搬会得到"跑得通、数是零"的静默错，比直接报错更危险；
   (e) **P1-1f 改道（D-cube1 两轨制）**：早先 910B 那份是 UB-direct 的 T.Cube 形态（编译绿、.o 出
   得来，`K-GEMM ASC-COMPILE-PASS`），但真机 gemm_l1 连两窗崩在同一处（507015）——P1-1d 的
   `asc_copy_l12l0a/b` 九参位序修复被证"必要不充分"，V1/V2 的 stride 单位与 V3 的 GM→L1 分形（NZ）
   布局仍未证真，继续押注等于把 P2 训练步数值里程碑吊在最深的研究问题上。裁决：ascend 实现改可
   移植标量体（wave2 已证该形态真机 1e-7 级可靠——gdn delta / addln / rope / attn_sw / gdn-conv
   全是标量面），**同一份正文**由 dialect(target) 注入两遍 ⇒ cpu 轨真跑当数值 oracle（本地闭环，
   这是 cube 数值第一次可在本地验）、ascend 轨只判编译；Cube 形态整体存档到
   `reconcile/S_perf_track_cube.py` 归 performance-track 考古。另两条 910B 纪律照旧：累加不放 L0C
   （硬件 fixpipe L0C→UB = assert(false)，卷宗 b10），也不用 fp32 的 `T.clear`（K 波实测在 UB 上
   生成坏 float2 store `no viable overloaded '='`）——清零改标量置 0。
"""
import tilelang

from typing import Any

import torch

from ascend.kernels import ascend_env

#: 内核侧真正落地的截断口径；其余口径由入口层走普通写法（与 P1 同名同取值）
SUPPORTED_ACTS = ("none", "relu")
ACT_MODE = {"none": 0, "relu": 1}

#: 昇腾侧内核位宽白名单（P1-1m）：fp32/fp16 已在正文按位宽参数化并本地闭环（reconcile/
#: C2_dtype_matrix.md）。**bf16 不在名单里是有意的**：bisheng 15.0.5 在 dav-2201 vector 面对 bf16
#: 标量转换直接 `fatal error: error in backend: not support bf16 type cast`（取证 attempts/B/
#: b_addln_910b.py:23-24 与 attempts/B/dump_bf16.py），而生产口径本就是 fp16。早先这里收 bf16/fp16
#: 而正文声明 fp32 ⇒ 编译按**声明**位宽出核、非 fp32 张量到发射期才被当 fp32 解读（静默错数据，取证
#: reconcile/I_RESULT.md §⑥-3；host 侧则崩在 cython `_check_buffer_dtype`）。位宽不符现在 plan 期拒。
_ASC_DTYPES = (torch.float32, torch.float16)
#: torch dtype → DSL 位宽名（白名单外的 dtype 进不了这里，plan 已挡）
_TL_DT = {torch.float32: "float32", torch.float16: "float16"}

#: M 轴块高：行块太大会浪费并行度（批大小可能等于 1）
BLOCK_M = 16
#: N/K 轴候选块宽，从大到小取能整除的那个；都不能整除则该形状表达不了
BLOCK_LADDER = (64, 32, 16)
#: 昇腾侧一次发射占用的核块数（correctness 轨＝列块数常量网格，此量只作上限口径保留）
NUM_BLOCKS = 8
VEC_THREADS = 64
NUM_STAGES = 2


def gemm_cpu_impl(A, W, bias, C, n: int, k: int, bm: int, bn: int, bk: int, act_mode: int):
    """CPU(c) 方言正文：全程 fp32，出口 fp32；网格只铺编译期常量列块，行块用动态上界串行推。

    白话：竖着切成固定的几列，每一列自己从上往下把行一段段搬进小格子算；有多少行都不影响切法，
    算完先在公共小格子里加加成、把负数压成零，再整块写回大表格。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile）。
    """
    import tilelang.cpu.language as T

    m = T.dynamic("m")
    nb = n // bn  # 列块数：n 已被 plan 判过整除，是编译期常量

    @T.prim_func
    def gemm_impl(A: T.Tensor((m, k), "float32"), W: T.Tensor((n, k), "float32"),
                  bias: T.Tensor((n,), "float32"), C: T.Tensor((m, n), "float32")):
        """被追踪的那一层：形状在这里落定，只有行数 M 是活的。

        白话：CPU 件这一份，模具在这一层定型——列切成几块、每块多宽、要不要压负数都写死，
        只有"这批有多少行"留成活的，运行时多长都按同一块模具走，行与行之间不串。
        
        """
        with T.Kernel(nb) as mx:
            As = T.alloc_local((bm, bk), "float32")
            Ws = T.alloc_local((bn, bk), "float32")
            Cs = T.alloc_local((bm, bn), "float32")
            for my in T.serial(T.ceildiv(m, bm)):
                T.clear(Cs)
                for ko in T.serial(T.ceildiv(k, bk)):
                    T.copy(A[my * bm, ko * bk], As)
                    T.copy(W[mx * bn, ko * bk], Ws)
                    T.gemm(As, Ws, Cs, transpose_B=True)
                for i in T.serial(bm):
                    for j in T.serial(bn):
                        Cs[i, j] = Cs[i, j] + bias[mx * bn + j]
                        if act_mode == 1:
                            if Cs[i, j] < 0:
                                Cs[i, j] = 0
                T.copy(Cs, C[my * bm, mx * bn])

    return gemm_impl


def _alloc_tile(T, shape, dtype):
    """tile 暂存的方言落点：cpu 走 local（day-1 口径），ascend 走 UB shared（H 波口径）。

    这是正文生成器里**唯一**按方言分叉的一处，分的是"存储层归属"而不是算式：两方言各自的公共面
    都不保证对方那层能落地（c 后端没有 UB 语义、910B 的 local 面未证），故在此收口，算式仍单份。
    """
    if getattr(T, "__tilelang_dialect__", "") == ascend_env.TARGET_ASCEND:
        return T.alloc_shared(shape, dtype)
    return T.alloc_local(shape, dtype)


def gemm_scalar_body(T, A, W, bias, C, n: int, k: int, bm: int, bn: int, bk: int,
                     act_mode: int, idt: str = "float32", odt: str = "float32"):
    """可移植标量正文（D-cube1 correctness 轨）：同一份算式在 cpu/ascend 两方言下各追一遍。

    白话：列切成固定的几块，每块自己从上往下把行一段段取进小格子；取数时末尾不满一行块的部分
    先夹住地址再判"这行要不要"（不要就记零），乘加全用标量一笔笔算，算完逐格落回大表格、越界
    的那几行根本不碰——所以批大小怎么变都不会读到表外的数。

    四条纪律（都是取证逼出来的）：
      1. 零 cube intrinsic / 零 SimtVF / 零 T.Parallel：只用 `T.serial` + 标量，910B(dav-2201)
         面 SIMT 载体根本不存在，cube 面 V1/V2/V3 未证真（见模块 docstring ⑤/(e)）。
      2. 清零走标量置 0，不用 `T.clear`：fp32 的 clear 在 UB 上生成坏 float2 store（K 波取证）。
      3. 越界防线自己做（夹址 + 守卫），不赌 T.copy 的自动边界谓词：卡上越界读的表现就是 507015。
      4. W 块（N/K 两轴都是编译期常量、且 plan 已判整除）永不越界 ⇒ 仍走 `T.copy` 整块 GM→UB，
         保住"搬运走 DMA、计算走标量"的分层；A 块行轴是活的 ⇒ 走夹址标量填装。

    :return: 追踪好的 PrimFunc（cpu 轨交 tilelang.compile 真跑，ascend 轨只判编译）。
    """
    m = T.dynamic("m")
    nb = n // bn   # 列块数＝网格上界：纯常量，两方言同一条纪律
    nk = k // bk   # K 分块数：同样常量（k 已被 plan 判过整除）
    # 位宽参数化（P1-1m）：入参/出口声明跟着实际位宽走，算式恒在 fp32 面上做。缺省 float32 时
    # 下面两个 helper 是**恒等函数**（一个 cast 都不插）⇒ fp32 轨的生成码与参数化之前逐字相同。
    def _ld(e):            # 标量读：idt → fp32 累加面
        return e if idt == "float32" else T.cast(e, "float32")

    def _st(e):            # 标量写：fp32 结算面 → odt 交货
        return e if odt == "float32" else T.cast(e, odt)

    @T.prim_func
    def gemm_impl(A: T.Tensor((m, k), idt), W: T.Tensor((n, k), idt),
                  bias: T.Tensor((n,), "float32"), C: T.Tensor((m, n), odt)):
        """被追踪的那一层：列块/块宽/要不要压负数都是常量，只有行数 m 是活的。

        白话：模具在这一层定型——竖着切几列、每列多宽、K 分几段都写死；有多少行不影响切法，
        行与行之间不串，末尾不满一块的行既不读表外也不写表外。
        """
        with T.Kernel(nb) as mx:
            a_ub = _alloc_tile(T, (bm, bk), "float32")   # 夹址标量填装面：恒 fp32（读时才升宽）
            w_ub = _alloc_tile(T, (bn, bk), idt)           # DMA 装填：必须与源张量同宽（搬运不许转类型）
            c_ub = _alloc_tile(T, (bm, bn), "float32")
            acc = T.alloc_var("float32", T.float32(0.0))
            v = T.alloc_var("float32", T.float32(0.0))
            for my in T.serial(T.ceildiv(m, bm)):
                row0 = my * bm
                # 清零（标量置 0，不用 T.clear：910B UB 上会生成坏 float2 store）
                for i in T.serial(bm):
                    for j in T.serial(bn):
                        c_ub[i, j] = T.float32(0.0)
                for ko in T.serial(nk):
                    # W 块：N/K 全常量且整除 ⇒ 永不越界，整块 GM→UB 交搬运
                    T.copy(W[mx * bn, ko * bk], w_ub)
                    # A 块：行轴是活的 ⇒ 夹址标量填装，无效行记 0，杜绝越界读
                    for i in T.serial(bm):
                        row = row0 + i
                        rc = T.min(row, m - 1)
                        for kk in T.serial(bk):
                            a_ub[i, kk] = T.if_then_else(row < m, _ld(A[rc, ko * bk + kk]),
                                                         T.float32(0.0))
                    # 标量三重循环：c[i,j] += Σ_kk A[i,kk]·W[j,kk]（W 按 (N,K) 存 ⇒ 天然右翻方向）
                    for i in T.serial(bm):
                        for j in T.serial(bn):
                            acc = T.float32(0.0)
                            for kk in T.serial(bk):
                                acc = acc + a_ub[i, kk] * _ld(w_ub[j, kk])
                            c_ub[i, j] = c_ub[i, j] + acc
                # 收尾：加偏置、按需压负为零，逐格落全局并按运行时行号守卫（尾行不碰）
                for i in T.serial(bm):
                    row = row0 + i
                    if row < m:
                        for j in T.serial(bn):
                            v = c_ub[i, j] + bias[mx * bn + j]
                            if act_mode == 1:
                                if v < 0:
                                    v = T.float32(0.0)
                            C[row, mx * bn + j] = _st(v)

    return gemm_impl


def gemm_asc_impl(A, W, bias, C, n: int, k: int, bm: int, bn: int, bk: int, act_mode: int,
                  idt: str = "float32", odt: str = "float32"):
    """昇腾 910B 正文＝可移植标量体（P1-1f 改道，D-cube1 correctness 轨）。

    与 cpu 侧共用 `gemm_scalar_body` 这一份算式，只是语言模块换成 `tilelang.ascend.language`：
    零 `T.Cube`/零 `T.gemm`/零 SimtVF/零 L0C-L1 参与，搬 W 用 `T.copy`(GM→UB)，其余全标量。
    这样 cpu 轨能真跑当 oracle、ascend 轨只判编译，数值在本地就闭环（cube 形态那套 V1/V2 stride
    单位与 GM→L1 分形布局的未证真参数，一律不再出现在本件的依赖里）。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")，判决见
        reconcile/S_cpu_oracle.py --ascend 与 reconcile/verify_gemm.py）。
    """
    return gemm_scalar_body(ascend_env.dialect(ascend_env.TARGET_ASCEND), A, W, bias, C,
                            n=n, k=k, bm=bm, bn=bn, bk=bk, act_mode=act_mode, idt=idt, odt=odt)


def _largest_block(total: int) -> int | None:
    """在候选块宽里取能整除 total 的最大值；取不到说明这个维度表达不了（返回 None）。"""
    for blk in BLOCK_LADDER:
        if total % blk == 0:
            return blk
    return None


# ── P1-1p / P1-1q：W 块装填档的选择纪律 ────────────────────────────────────────
# 取证出处：release/runs/1011-p2-13-oncard-accept-2002/artifacts/sys1_kernels.tgz（归档真机生成码）
# 与 reconcile/M2_FP16_NAN_EVIDENCE.md。要点是生成码原文可复核的两点对置，不是推测：
#   · `bk < k` 时 `T.copy(W[mx*bn, ko*bk], w_ub)` 编成**多突发跨步 GM→UB DMA**，单突发宽 = bk*字节宽。
#     卡上实测出错的两枚同形：`ea1059e6…` device_kernel.asc:31（fp16 dA n32k96bk32 ⇒ 32 突发 ×64B、
#     跨距 192B）＝P1-1p 的 NaN 起点；`5d732a8f…` :31（fp32 n96k48bk16 ⇒ 同一组 32/64/192/64）＝P1-1q
#     的真机错值。经 compat `TL910B_BYTES_TO_BLK`（port910b_compat.h:599，÷32）换算即
#     (blockLen, srcStride, dstStride) = (2, 6, 2) 个 32B 块。
#   · `bk == k` 时 tilelang `src/ascend/op/copy.cc` 的 `plan_single_row` 折叠成**整块单发连续**
#     （blockCount=1），同位宽在卡上实测清白：fp16 前向 `8f30a21f…`/`e07da648…` 各 `1×2048B`
#     （H 案同窗前向 15 次调用零 NaN 零 inf），fp32 见 `3249db2f…` `1×8192B`、`32d6c39c…` `1×16384B`。
_MIN_STRIDED_BURST_B = 128
#: 动态 UB 预算 = 归档里**已在卡上编出产物**的最大申报值（`32d6c39c…` host_kernel.c 申报 24576B；
#: 与 reconcile/S_RESULT.md 算过的 gemm 最坏档 24KB 同量级）。是工程预算，不是硬件规范。
_UB_BUDGET_B = 24576


def _ub_bytes(bm: int, bn: int, bk: int, elem: int) -> int:
    """`gemm_scalar_body` 每核块的 UB 需求：a_ub(bm×bk, 恒 fp32) + w_ub(bn×bk, 随源位宽)
    + c_ub(bm×bn, fp32)。与生成码里 `buf_dyn_shmem` 的分块一一对应（对账见取证探针）。"""
    return bm * bk * 4 + bn * bk * elem + bm * bn * 4


def _pick_bk_ascend(bm: int, bn: int, k: int, elem: int) -> int | None:
    """ascend 轨的 K 块宽选择：**先试整行连续（bk==k ⇒ 单发突发档）**。

    白话：竖条整块一次取进小格子这种做法在卡上是证过清白的；切成几段、跨步去取，段宽只剩 64 字节
    的两处实测都错了。所以能整块取就整块取；整块取装不下小格子才退回分段，且分段只允许每段够宽
    （>=128 字节）的那几种；两条都不满足就交 None，由入口层换普通写法（不硬凑）。
    """
    if _ub_bytes(bm, bn, k, elem) <= _UB_BUDGET_B:
        return k
    ladder = _largest_block(k)
    if ladder is not None and ladder * elem >= _MIN_STRIDED_BURST_B:
        return ladder
    return None


def plan(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None, C: torch.Tensor,
         act: str, target: str | None = None) -> dict[str, Any] | None:
    """判断这组形状/口径能不能交给方言内核；能则返回编译参数与缓存键，否则返回 None（不硬凑）。

    硬条件三条：截断口径在支持表里；N/K 都能被某个候选块宽整除；位宽组合在该 target 的白名单里
    （CPU 侧只收 fp32；昇腾侧收 **fp32/fp16** 同宽操作数与出口 + fp32 偏置，bf16 显式不收，理由见
    `_ASC_DTYPES`）。任一不满足就回
    None，由入口层换普通写法。

    白话：先看这活儿的尺寸和记法合不合现成模具；不合就不勉强开模，直接换手工做法。
    """
    name = ascend_env.normalize_target(target)
    if act not in SUPPORTED_ACTS:
        return None
    n, k = int(W.size(0)), int(W.size(1))
    bn, bk = _largest_block(n), _largest_block(k)
    if bn is None or bk is None:
        return None
    if C.dtype != A.dtype or bias is not None and bias.dtype != torch.float32:
        return None
    if name == ascend_env.TARGET_CPU and A.dtype != torch.float32:
        return None
    if name == ascend_env.TARGET_ASCEND and A.dtype not in _ASC_DTYPES:
        return None
    if A.dtype != W.dtype or not (A.is_contiguous() and W.is_contiguous() and C.is_contiguous()):
        return None
    dt = _TL_DT.get(A.dtype, "float32")   # plan 已判 C.dtype == A.dtype 且位宽在白名单内 ⇒ 单段位宽
    if name == ascend_env.TARGET_ASCEND:
        # 装填档纪律（P1-1p/P1-1q，取证见 `_pick_bk_ascend` 上方）：只动 bk 的取法，
        # 接口/正文算式/交货位宽不动；cpu 轨（产物里没有该 DMA 面）维持 ladder 原样。
        bk = _pick_bk_ascend(BLOCK_M, bn, k, 2 if dt == "float16" else 4)
        if bk is None:
            return None
    kwargs = {"n": n, "k": k, "bm": BLOCK_M, "bn": bn, "bk": bk, "act_mode": ACT_MODE[act]}
    impl = gemm_asc_impl if name == ascend_env.TARGET_ASCEND else gemm_cpu_impl
    if name == ascend_env.TARGET_ASCEND:
        kwargs.update(idt=dt, odt=dt)
    # 键尾的位宽段是防“陈旧模具复用”：同 (n,k,b,act) 换位宽必须另开一份产物，否则 fp32 的产物
    # 会被 fp16 形状命中缓存、发射期把 fp16 内存当 fp32 读（静默错数据）。
    key = f"gemm[{name}]|n{n}k{k}b{BLOCK_M}x{bn}x{bk}a{act}d{dt}"
    return {"target": name, "key": key, "kwargs": kwargs, "impl": impl, "shape": (n, k)}


def run(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None, C: torch.Tensor,
        spec: dict[str, Any]) -> bool:
    """真跑一次内核并把结果写进 C；编译产物取不到就返回 False（由入口层落回退）。

    缓存键不含行数：同一份产物服务 8、33、8192 等不同批大小（m 是 PrimFunc 里的动态符号），
    换批重编的秒级开销会吞掉收益。首编用当次张量做形状追踪，之后只喂出口缓冲。

    白话：模具没开成就说没开成，返回一个假字让上层换手工做法，绝不"假装算过"把空表交出去。

    :returns: True 表示 C 已被内核写入。
    """
    zero = bias if bias is not None else torch.zeros(spec["shape"][0], dtype=torch.float32,
                                                     device=A.device)
    kernel = ascend_env.get_compiled(
        spec["key"],
        lambda: tilelang.compile(spec["impl"](A, W, zero, C, **spec["kwargs"]),
                                 out_idx=[], **ascend_env.compile_kwargs(spec["target"])),
        spec["target"])
    if kernel is None:
        return False
    kernel(A, W, zero, C)
    return True
