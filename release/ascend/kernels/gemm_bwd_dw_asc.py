"""gemm 权重梯度方言件：dW = dYᵀ @ A，沿 token 轴归约（token 数 m 是动态符号）。

【做什么】算"每个连接权该往哪儿挪"的那张表：把一批头信号转置后乘回输入，沿"这批有几条样本"的轴
累加成与权重同形的 (N, K) 结果。前向件的模具把动态轴放在行数上、归约轴是编译期常量，表达不了
本件事，所以另开一模。昇腾那份与 CPU 那份的标量口径同形（同一套"沿活轴串行累加"的账），只是
语言模块由 `ascend_env.dialect(target)` 注入。
【怎么做】① 出口按 (N,K) 分块，网格上界是纯常量（块数），token 轴用 T.serial(ceildiv(m, tc))
   动态上界串行推进——换批大小不重编（实测同一产物连跑 m=8/33 误差都是 0.0，compile_count=1）；
   ② 整数常量必须是被追踪函数自己的参数，且本文件不写 from __future__ import annotations：
   tilelang 取张量注解只认闭包非局部名，PEP 563 会让只在注解里出现的 n/k 失去闭包单元
   （实测 NameError: name 'n' is not defined）；③ 取产物走 tilelang.compile(现场追踪的 PrimFunc)
   而非 tilelang.jit——后者在本件这种写法下只回 Kernel 不执行，且它的 call-form 缓存拿张量做
   等值比较，换一批新张量就抛错；④ 累加器 (bn,bk) 与出口 GW 恒 fp32（归约不降位），交货降位由入口层负责；入参位宽由 `idt`
   参数化（fp32/fp16，P1-1m），cast 只发生在标量 load 处（填装走标量、不走 DMA，故缓冲可留 fp32）；⑤ 尾块
   （token 块不满 tc 行）不赌搬运的自动边界谓词：先按 `T.min(row, m-1)` 把地址夹进合法区间，再用
   `T.if_then_else(row < m, 读值, 0.0)` 决定取不取——`if_then_else` 两侧都会求值，地址不夹就是
   越界读，而越界读在卡上的表现正是 aicore exception 507015。
【为什么】昇腾那份从 P1-1f 起改走**可移植标量体**（D-cube1 两轨制的 correctness 轨）：`dw_scalar_body`
   是 cpu/ascend 共用的**同一份正文**，只用两方言公共子集（T.serial + 标量 load/store + 整数索引
   算术 + 常量网格），零 T.Cube/零 T.gemm/零 SimtVF/零 L0C-L1 参与 ⇒ cpu 轨真编真跑当数值 oracle
   （本地闭环）、ascend 轨只判编译。此前那套 **l0tr 主案**（把转置下移到 L1→L0 装填、mad 恒吃 NT、
   L0C→GM 直出）编译面已绿（§12 补 `asc_fill_l1`→`set_l1_2d` 后 P1-4b PASS），但真机 dW 与 gemm_l1
   同型崩在 507015：P1-1d 的九参位序修复（sid 回穿 + transpose 位补齐）只算"必要不充分"，V1/V2 的
   L1→L0 stride 单位、V3 的 GM→L1 分形（NZ）布局、V4/V5/V6 的 fill 位段都还没在卡上证真——把这些
   吊在 P2 训练步数值里程碑上不成（D-cube1 原话）。l0tr 形态整体存档在 `reconcile/S_perf_track_cube.py`
   交 performance-track 考古。
   算式口径不变：GW(k 轴按 (N,K) 出口) = Σ_t dY[t, n]·A[t, k]，天然 trans_A——标量三重循环直接按
   这个方向累加，不再需要"mad 只认左不翻右翻"的下移动作（该约束是 cube 面的，标量面无此说）。
   被否方案一（承旧）：照 950 载体用 SimtVF 在 UB 做转置写回——910B 方言无此构造，COMPILE-FAIL；
   被否方案二：继续押 l0tr intrinsic 单轨——三窗两崩已证其不可预期，且本地无法验数值（cpu 轨编不了
   cube intrinsic），改道后 oracle 才成立；被否方案三：token 数做编译期常量换批重编——训练批形多变，
   秒级重编吞掉收益。
云端待验清单（本波只判编译＋本地 cpu 数值）：标量体真机 rel（判据已备：run_numerics gemm_l1/dw +
train_step）；性能代价见 reconcile/S_RESULT.md（标量 vs cube 的理论差在 10² 量级，属 performance-track）。
"""
import tilelang

from typing import Any

import torch

from ascend.kernels import ascend_env

#: 昇腾侧内核位宽白名单（P1-1m）：fp32/fp16；**bf16 显式不收**（bisheng dav-2201 vector 面缺
#: bf16 标量 cast，取证 attempts/B/b_addln_910b.py:23-24 + dump_bf16.py；口径同 gemm_asc._ASC_DTYPES）
_ASC_DTYPES = (torch.float32, torch.float16)
#: torch dtype → DSL 位宽名
_TL_DT = {torch.float32: "float32", torch.float16: "float16"}

#: token 轴块长：与前向 BLOCK_M 同量级，太小喂不饱并行、太大尾块浪费
BLOCK_TC = 16
#: N/K 轴块宽候选（沿用前向阶梯）；都不能整除则表达不了，交回入口层回退
BLOCK_LADDER = (64, 32, 16)
#: 昇腾侧一次发射占用的核块数上限（correctness 轨＝出口块数常量网格，此量作摊开上限保留）
NUM_BLOCKS = 8
VEC_THREADS = 64
NUM_STAGES = 2


def dw_cpu_impl(dY, A, GW, n: int, k: int, tc: int, bn: int, bk: int):
    """CPU(c) 方言正文：dW(N,K) 沿动态 token 轴累加；网格是常量，token 循环是动态上界。

    白话：把梯度表切成固定大小的小方格，每格自己把"这一批所有样本对该格的贡献"一笔笔记到同一个
    账本上；样本有多少条不影响账本格子的大小，记完一次交账。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile）。
    """
    import tilelang.cpu.language as T

    m = T.dynamic("m")
    nb, nk = n // bn, k // bk

    @T.prim_func
    def dw_impl(dY: T.Tensor((m, n), "float32"), A: T.Tensor((m, k), "float32"),
                GW: T.Tensor((n, k), "float32")):
        """被追踪的那一层：权重表 (N,K) 是常量，样本行数 M 是活的累加轴。

        白话：CPU 件这一份，把每条样本的头信号竖起来，和当时的输入两两对上记总账；行数多几条
        少几条只是"记的次数"不同，账本形状从头到尾没变，所以换一批不用重新开模具。
        
        """
        with T.Kernel(nb, nk) as (iy, ix):
            Ds = T.alloc_local((tc, bn), "float32")
            As = T.alloc_local((tc, bk), "float32")
            Cs = T.alloc_local((bn, bk), "float32")
            T.clear(Cs)
            for to in T.serial(T.ceildiv(m, tc)):
                T.copy(dY[to * tc, iy * bn], Ds)
                T.copy(A[to * tc, ix * bk], As)
                T.gemm(Ds, As, Cs, transpose_A=True)
            T.copy(Cs, GW[iy * bn, ix * bk])

    return dw_impl


def _alloc_tile(T, shape, dtype):
    """tile 暂存的方言落点：cpu 走 local（day-1 口径），ascend 走 UB shared（H 波口径）。

    与 gemm_asc._alloc_tile 同口径（分的是"存储层归属"不是算式，两方言各自公共面都不保证对方的
    层能落地），为免件间互相依赖在此各留一份。
    """
    if getattr(T, "__tilelang_dialect__", "") == ascend_env.TARGET_ASCEND:
        return T.alloc_shared(shape, dtype)
    return T.alloc_local(shape, dtype)


def dw_scalar_body(T, dY, A, GW, n: int, k: int, tc: int, bn: int, bk: int,
                   idt: str = "float32"):
    """可移植标量正文（D-cube1 correctness 轨）：同一份算式在 cpu/ascend 两方言下各追一遍。

    白话：账本 (N,K) 先切成固定的小方格，每格自己沿"这批有多少条样本"一笔笔记账；取样本时末尾
    不满一块的行先夹住地址再判要不要（不要就记零），记完逐格落回总账——样本条数怎么变都不越界。

    四条纪律（与前向件同一套）：
      1. 零 cube intrinsic / 零 SimtVF / 零 T.Parallel：只用 T.serial + 标量 + 常量网格；
      2. 清零走标量置 0，不用 `T.clear`（fp32 clear 在 910B UB 上生成坏 float2 store）；
      3. 尾块夹址 + 守卫，不赌搬运的自动边界谓词（越界读＝卡上 507015 的路子）；
      4. 出口 (N,K) 两轴都是编译期常量且 plan 已判整除 ⇒ 每个出口元素恰好写一次，无需守卫。

    :return: 追踪好的 PrimFunc（cpu 轨交 tilelang.compile 真跑，ascend 轨只判编译）。
    """
    m = T.dynamic("m")
    nb, nk = n // bn, k // bk
    tiles = nb * nk          # 出口块数：纯常量 ⇒ 可直接当网格上界
    blocks = min(tiles, NUM_BLOCKS)
    iters = (tiles + blocks - 1) // blocks
    # 位宽参数化（P1-1m）：入参按实际位宽声明、累加面与出口 GW 恒 fp32（归约不降位）。缺省
    # float32 时 `_ld` 是恒等函数 ⇒ fp32 轨生成码与参数化之前逐字相同（零回归）。
    def _ld(e):            # 标量读：idt → fp32 累加面
        return e if idt == "float32" else T.cast(e, "float32")

    @T.prim_func
    def dw_impl(dY: T.Tensor((m, n), idt), A: T.Tensor((m, k), idt),
                GW: T.Tensor((n, k), "float32")):
        """被追踪的那一层：出口表形状是常量，token 轴串行累加、出口块按块摊开。

        白话：一张大账本按 (N,K) 切成几页，每页自己把这一批的数一笔笔记上去；页数与每页多宽都
        定死，来多少行就记多少轮，不会把别页的数字串了行。
        """
        with T.Kernel(blocks) as bid:
            dy_ub = _alloc_tile(T, (tc, bn), "float32")
            a_ub = _alloc_tile(T, (tc, bk), "float32")
            acc_ub = _alloc_tile(T, (bn, bk), "float32")
            acc = T.alloc_var("float32", T.float32(0.0))
            for it in T.serial(iters):
                tid = bid + it * blocks
                iy = tid // nk     # 出口块的行页号（N 轴）
                ix = tid % nk      # 出口块的列页号（K 轴）
                # 清零（标量置 0）：每个出口块自己记自己那本账
                for i in T.serial(bn):
                    for j in T.serial(bk):
                        acc_ub[i, j] = T.float32(0.0)
                for to in T.serial(T.ceildiv(m, tc)):
                    row0 = to * tc
                    # 夹址标量填装：dY 与 A 的 token 轴都是活的，不满一块的行记 0（加了个寂寞也不影响）
                    for tt in T.serial(tc):
                        row = row0 + tt
                        rc = T.min(row, m - 1)
                        for i in T.serial(bn):
                            dy_ub[tt, i] = T.if_then_else(row < m, _ld(dY[rc, iy * bn + i]),
                                                          T.float32(0.0))
                        for j in T.serial(bk):
                            a_ub[tt, j] = T.if_then_else(row < m, _ld(A[rc, ix * bk + j]),
                                                         T.float32(0.0))
                    # 标量三重循环：GW[i,j] += Σ_tt dY[tt,i]·A[tt,j]（天然 trans_A，出口自然朝向）
                    for i in T.serial(bn):
                        for j in T.serial(bk):
                            acc = T.float32(0.0)
                            for tt in T.serial(tc):
                                acc = acc + dy_ub[tt, i] * a_ub[tt, j]
                            acc_ub[i, j] = acc_ub[i, j] + acc
                # 出口：逐格标量落全局；(N,K) 全常量且整除 ⇒ 不越界、不重不漏
                for i in T.serial(bn):
                    for j in T.serial(bk):
                        GW[iy * bn + i, ix * bk + j] = acc_ub[i, j]

    return dw_impl


def dw_asc_impl(dY, A, GW, n: int, k: int, tc: int, bn: int, bk: int, idt: str = "float32"):
    """昇腾 910B 正文＝可移植标量体（P1-1f 改道，D-cube1 correctness 轨）。

    与前向件同一招：共用 `dw_scalar_body`，语言模块换成 `tilelang.ascend.language`。零
    `T.Cube`/零 `T.gemm`/零 `alloc_l1|l0a|l0b|l0c`/零 SimtVF——l0tr 那套"转置下移到 L1→L0 装填 +
    L0C→GM 直出"连同它未证真的 V 系参数一起存档到 reconcile/S_perf_track_cube.py；本件对 §12 的
    依赖面缩到零（不再有 codegen 自发 `asc_fill_l1` 的尾形，尾块在正文里就夹住了）。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")，判决见
        reconcile/S_cpu_oracle.py --ascend 与 reconcile/verify_dw.py）。
    """
    return dw_scalar_body(ascend_env.dialect(ascend_env.TARGET_ASCEND), dY, A, GW,
                          n=n, k=k, tc=tc, bn=bn, bk=bk, idt=idt)


def _largest_block(total: int) -> int | None:
    """候选块宽里取能整除 total 的最大值；取不到即表达不了（返回 None 交回退）。"""
    for blk in BLOCK_LADDER:
        if total % blk == 0:
            return blk
    return None


def plan(dY: torch.Tensor, A: torch.Tensor, GW: torch.Tensor, target: str | None = None) -> dict[str, Any] | None:
    """这组形状能否交给方言模具：两输入同行数、出口 fp32、N/K 可整除、位宽在该 target 白名单内。

    白话：先看尺寸和记法合不合现成模具；不合就老实说干不了，由入口换普通做法，绝不硬凑。
    """
    name = ascend_env.normalize_target(target)
    if dY.dim() != 2 or A.dim() != 2 or dY.size(0) != A.size(0):
        return None
    if GW.dtype != torch.float32:
        return None
    n, k = int(GW.size(0)), int(GW.size(1))
    if dY.size(1) != n or A.size(1) != k:
        return None
    bn, bk = _largest_block(n), _largest_block(k)
    if bn is None or bk is None:
        return None
    if name == ascend_env.TARGET_CPU:
        if dY.dtype != torch.float32 or A.dtype != torch.float32:
            return None
    elif dY.dtype != A.dtype or dY.dtype not in _ASC_DTYPES:
        return None
    if not (dY.is_contiguous() and A.is_contiguous() and GW.is_contiguous()):
        return None
    dt = _TL_DT.get(dY.dtype, "float32")   # plan 已判 A 与 dY 同宽、GW 恒 fp32、位宽在白名单内
    kwargs = {"n": n, "k": k, "tc": BLOCK_TC, "bn": bn, "bk": bk}
    impl = dw_asc_impl if name == ascend_env.TARGET_ASCEND else dw_cpu_impl
    if name == ascend_env.TARGET_ASCEND:
        kwargs.update(idt=dt)
    # 键尾位宽段：同 (n,k,块) 换位宽必得新模具（防陈旧产物把 fp16 内存当 fp32 读，见 gemm_asc 同注）
    key = f"gemm_dw[{name}]|n{n}k{k}tc{BLOCK_TC}b{bn}x{bk}d{dt}"
    return {"target": name, "key": key, "kwargs": kwargs, "impl": impl}


def run(dY: torch.Tensor, A: torch.Tensor, GW: torch.Tensor, spec: dict[str, Any]) -> bool:
    """真跑内核并把结果写进 GW；返回 False 表示模具没开成，须由入口层回退（不许假装算过）。

    缓存键不含行数：同一份产物服务不同批大小——权重梯度每步都要算，换批重编会把收益吃光。

    白话：模具只在第一次开，之后不管一批里有多少条样本都用同一个；开不了就如实报回去。
    """
    kernel = ascend_env.get_compiled(
        spec["key"],
        lambda: tilelang.compile(
            spec["impl"](dY, A, GW, **spec["kwargs"]),
            out_idx=[], **ascend_env.compile_kwargs(spec["target"])),
        spec["target"])
    if kernel is None:
        return False
    kernel(dY, A, GW)
    return True
