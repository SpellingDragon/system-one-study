"""letter_readout 方言件：按行号表从一张大表里"抽行"（gather），反向是把梯度按同一张行号表加回原表。

【做什么】前向 `forward(rows, ids, out_dtype, target)` 交回 `out[i] = rows[ids[i]]`——设计文档里
这条叫"读出 letter_rows"，即字母表那几十行从权重表里点出来交给下游，不做任何算术。反向
`backward(row_count, ids, dY, target)` 交回 `dRows[ids[i]] += dY[i]`：同一行被点到多次时必须是
**加**，不是覆盖。
【怎么做】① 前向按"一次一行"走：把那一行整条搬进本地/UB，再整条搬出到出口——搬运自带边界谓词，
   所以抽行数（picked）与表行数（total）都可以是动态符号；② 反向对全局表做累加：CPU 那份只有
   一个块、串行扫抽行号，天然无竞争；昇腾那份一行可能落在不同核上，必须用原子加，绝不能用普通
   写覆盖（先例：本包 add_ln 的列梯度同一口径）；③ 行号表必须是 int32 且连续——搬运的下标要
   直接进地址算式，dtype 不符时入口层先转（不做静默截断）。
【为什么】这一件看着像"顺手就能用 torch.index_select"，仍写成方言件的原因是它在训练链上：反向的
   scatter-add 若走 torch 就得把整张表拉回来再算，表宽时这是纯带宽浪费；内核里按行搬、按行加，
   一行的代价就是一行。被否方案：把 gather 并进下游的 gemm（只读需要的权重行）——省一次搬运，
   但会把"读出"和"乘加"两件事绑死，letter_rows 的下游并不只有 gemm，且会让 gemm 件的形状判据
   多一条动态行号分支，得不偿失。昇腾侧原子加是否够快属未定项，云端 C1 若不稳就换"每核一份
   偏量表 + 第二段归约"（本包 add_ln_asc 的注释里记了同一处置）。
"""
import tilelang

from typing import Any

import torch

from ascend.kernels import ascend_env

#: 行号表允许的位宽：内核只收 int32（入口层负责降级），int64 走 torch 回退时也能直接用
ID_DTYPES = (torch.int32, torch.int64)
#: 昇腾侧一次发射占用的块数与流水深度
NUM_BLOCKS = 8
VEC_THREADS = 64
NUM_STAGES = 2


def gather_cpu_impl(Rows, Idx, Out, dim: int):
    """CPU(c) 前向正文：一行一次整条搬进本地缓冲、再整条搬出。

    白话：按点名册从大表里一行一行抽出来，抽出来原样摆进新表，一个数都不改。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile）。
    """
    import tilelang.cpu.language as T

    total = T.dynamic("total")
    picked = T.dynamic("picked")

    @T.prim_func
    def gather_impl(Rows: T.Tensor((total, dim), "float32"), Idx: T.Tensor((picked,), "int32"),
                    Out: T.Tensor((picked, dim), "float32")):
        """被追踪的那一层：被点名的行数与总行数都是动态维，行宽是常量。

        白话：CPU 件这一份，按点名册从大表里一行行抽出来摆成新表，整行一次搬完；册子里重复点
        同一行就摆两遍，抽出来的数字一个都不改，也不做任何换算。
        
        """
        with T.Kernel(1) as bx:
            line = T.alloc_local((dim,), "float32")
            for i in T.serial(picked):
                # 下标从行号表里取，搬运本身带边界谓词，所以抽多少行都能动态
                T.copy(Rows[Idx[i], 0], line)
                T.copy(line, Out[i, 0])

    return gather_impl


def gather_asc_impl(Rows, Idx, Out, dim: int):
    """昇腾前向正文：每块认领一段点名册，逐行经 UB 搬进搬出。

    白话：几台机器分着念点名册，念到谁就把那一整行原样搬到台面上再放进出口，绝不中途改写。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")，本地不实编）。
    """
    import tilelang.ascend.language as T

    total = T.dynamic("total")
    picked = T.dynamic("picked")

    @T.prim_func
    def gather_impl(Rows: T.Tensor((total, dim), "float32"), Idx: T.Tensor((picked,), "int32"),
                    Out: T.Tensor((picked, dim), "float32")):
        """被追踪的那一层：抽多少行是活的，搬运按整行发起、起点直接取自册子。

        白话：昇腾件这一份，册子上的行号直接当搬运起点用，一次搬一整行到新表对应的位置；
        不够一轮的量就不发起，搬过来的数字原封不动，这一步连算术都不做。
        
        """
        with T.Kernel(NUM_BLOCKS) as bx:
            line_ub = T.alloc_shared((dim,), "float32")
            T.annotate_buffer_versions({line_ub: NUM_STAGES})
            for unit in T.Persistent([picked], NUM_BLOCKS, bx,
                                     group_size=1, num_stages=NUM_STAGES):
                i = unit * NUM_BLOCKS + bx
                if i < picked:
                    # 搬运不许顺带转类型，所以全程 fp32 同宽；行号直接进地址算式
                    T.copy(Rows[Idx[i], 0], line_ub)
                    T.copy(line_ub, Out[i, 0])

    return gather_impl


def scatter_add_cpu_impl(GOut, Idx, GRows, dim: int):
    """CPU(c) 反向正文：单块串行把每行梯度加回原表（同一行被点两次就是两次加法）。

    白话：按同一本点名册，把手里这批行一条条添回大表对应的行里；同一行被添两次就是真的加两次，
    不会后一次盖掉前一次。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile）。
    """
    import tilelang.cpu.language as T

    total = T.dynamic("total")
    picked = T.dynamic("picked")

    @T.prim_func
    def scatter_impl(GOut: T.Tensor((picked, dim), "float32"), Idx: T.Tensor((picked,), "int32"),
                     GRows: T.Tensor((total, dim), "float32")):
        """被追踪的那一层：退回的行数是动态维，目标表行数与行宽按调用方形状落定。

        白话：CPU 件这一份，把抽出来的这些行按同一张点名册原路退回大表；同一行退回两次就是
        真的加两次，绝不互相盖掉。目标表在退回之前必须已经是清零的，这一步只加不写。
        
        """
        with T.Kernel(1) as bx:
            # 出口表必须先清零：本件只做"加一笔"，不做"盖一笔"
            for i in T.serial(picked):
                for j in T.serial(dim):
                    GRows[Idx[i], j] = GRows[Idx[i], j] + GOut[i, j]

    return scatter_impl


def scatter_add_asc_impl(GOut, Idx, GRows, dim: int):
    """昇腾反向正文：多核并发回写同一张表，逐元素原子加（口径同 add_ln 的列梯度）。

    白话：几个人同时往同一张大表上记数，每人每格都先看清楚再往上加，谁也不许把别人的数抹掉。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")，本地不实编）。
    """
    import tilelang.ascend.language as T

    total = T.dynamic("total")
    picked = T.dynamic("picked")

    @T.prim_func
    def scatter_impl(GOut: T.Tensor((picked, dim), "float32"), Idx: T.Tensor((picked,), "int32"),
                     GRows: T.Tensor((total, dim), "float32")):
        """被追踪的那一层：退回动作按行摊开，累加落在同一张公共表上。

        白话：昇腾件这一份，每个工人手里拿几行往大表对应行上添，同一行被多个人添也各自都算数；
        这一步只做加法，不比较、不覆盖、不做任何位宽换算。
        
        """
        with T.Kernel(NUM_BLOCKS) as bx:
            g_ub = T.alloc_shared((dim,), "float32")
            T.annotate_buffer_versions({g_ub: NUM_STAGES})
            for unit in T.Persistent([picked], NUM_BLOCKS, bx,
                                     group_size=1, num_stages=NUM_STAGES):
                i = unit * NUM_BLOCKS + bx
                if i < picked:
                    T.copy(GOut[i, 0], g_ub)
                    with T.SimtVF(threads=VEC_THREADS):
                        for j in T.Parallel(dim):
                            T.atomic_add(GRows[Idx[i], j], g_ub[j])

    return scatter_impl


#: 活种 x target 到方言正文的选件表，四条组合都摆出来，避免表达式里藏分支
_IMPLS = {
    ("gather", ascend_env.TARGET_CPU): gather_cpu_impl,
    ("gather", ascend_env.TARGET_ASCEND): gather_asc_impl,
    ("scatter_add", ascend_env.TARGET_CPU): scatter_add_cpu_impl,
    ("scatter_add", ascend_env.TARGET_ASCEND): scatter_add_asc_impl,
}


def _plan(src: torch.Tensor, ids: torch.Tensor, dst: torch.Tensor, dim: int,
          target: str | None, kind: str) -> dict[str, Any] | None:
    """两条路共用的判据：行号表是 int32 且连续、参与方 fp32、宽度一致；不过关回 None。

    白话：先确认点名册是标准编号（不是省格子的短编号）、表是细尺子记的、行宽两边对得上，
    不合模具就换手工。
    """
    name = ascend_env.normalize_target(target)
    if ids.dtype != torch.int32 or not ids.is_contiguous():
        return None
    if src.dtype != torch.float32 or dst.dtype != torch.float32:
        return None
    if not (src.is_contiguous() and dst.is_contiguous()):
        return None
    if int(src.size(-1)) != dim or int(dst.size(-1)) != dim:
        return None
    # 选件按 (活种, target) 显式查表：三元链在这里极易写反（ascend+scatter 会错选到 gather 件）
    impl = _IMPLS[(kind, name)]
    key = f"{kind}[{name}]|d{dim}"
    return {"target": name, "key": key, "kwargs": {"dim": dim}, "impl": impl}


def _run(args: tuple, spec: dict[str, Any]) -> bool:
    """取（或首编）内核并真跑一次；False 表示模具没开成，由入口层回退（不许假装算过）。

    缓存键不含表行数与抽行数：两个都是动态符号，换批不换产物。

    白话：模具只开第一次，之后不管表多大、抽多少行都用同一个。
    """
    kernel = ascend_env.get_compiled(
        spec["key"],
        lambda: tilelang.compile(spec["impl"](*args, **spec["kwargs"]),
                                 out_idx=[], **ascend_env.compile_kwargs(spec["target"])),
        spec["target"])
    if kernel is None:
        return False
    kernel(*args)
    return True


def forward(rows: torch.Tensor, ids: torch.Tensor, out_dtype: torch.dtype = torch.float32,
            target: str | None = None) -> torch.Tensor:
    """读出：`out[i] = rows[ids[i]]`（纯搬运，不做算术），出口位宽降到 out_dtype。

    白话：按点名册从大表里一行行抽出来摆成新表；册子里可以重复点同一行，重复就摆两遍。

    :raises ValueError: rows 至少二维、ids 不是整型一维或两者设备不一致时抛出。
    """
    _check(rows, ids)
    dim = int(rows.size(-1))
    lead = rows.shape[:-2]
    r2 = rows.reshape(-1, dim)
    if ascend_env.active_backend(target, rows.device) == ascend_env.TILELANG:
        r32 = r2.float().contiguous()
        idx = ids.to(torch.int32).contiguous()
        out = torch.empty((int(idx.numel()), dim), dtype=torch.float32, device=rows.device)
        spec = _plan(r32, idx, out, dim, target, "gather")
        if spec is not None and _run((r32, idx, out), spec):
            return out.to(out_dtype).reshape(*lead, idx.numel(), dim)
    return r2.to(out_dtype)[ids].reshape(*lead, ids.numel(), dim)


def backward(row_count: int, ids: torch.Tensor, dy: torch.Tensor,
             target: str | None = None) -> torch.Tensor:
    """反向：把 dy 按同一张行号表**加**回 (row_count, dim) 的表里（重复行号会累加）。

    白话：手里这批抽出来的行要原路退回大表；同一行退两次就是真的加两次，绝不互相盖掉。

    :raises ValueError: ids 不是整型一维，或 dy 的行数与 ids 的个数不等时抛出。
    """
    if ids.dim() != 1 or ids.dtype not in ID_DTYPES:
        raise ValueError(f"ids 需是一维整型行号表（int32/int64），实得 {tuple(ids.shape)} / {ids.dtype}")
    if dy.dim() < 2:
        raise ValueError(f"dy 至少二维，实得 {dy.dim()} 维")
    dim = int(dy.size(-1))
    if int(dy.reshape(-1, dim).size(0)) != int(ids.numel()):
        raise ValueError(f"dy 的行数 {dy.reshape(-1, dim).size(0)} 需等于行号个数 {ids.numel()}")
    lead = dy.shape[:-2]
    d2 = dy.reshape(-1, dim)
    if ascend_env.active_backend(target, dy.device) == ascend_env.TILELANG:
        acc = torch.zeros((int(row_count), dim), dtype=torch.float32, device=dy.device)
        g32 = d2.float().contiguous()
        idx = ids.to(torch.int32).contiguous()
        spec = _plan(g32, idx, acc, dim, target, "scatter_add")
        if spec is not None and _run((g32, idx, acc), spec):
            return acc.reshape(*lead, row_count, dim) if lead else acc
    acc = torch.zeros((int(row_count), dim), dtype=torch.float32, device=dy.device)
    acc.index_add_(0, ids, d2.float())
    return acc.reshape(*lead, row_count, dim) if lead else acc


def _check(rows: torch.Tensor, ids: torch.Tensor) -> None:
    """入口契约：rows 至少二维、ids 是一维整型且落在表行数之内、两者同设备。"""
    if rows.dim() < 2:
        raise ValueError(f"rows 至少二维 (..., 行数, 行宽)，实得 {rows.dim()} 维")
    if ids.dim() != 1 or ids.dtype not in ID_DTYPES:
        raise ValueError(f"ids 需是一维整型行号表，实得 {tuple(ids.shape)} / {ids.dtype}")
    if int(ids.numel()) and (int(ids.min()) < 0 or int(ids.max()) >= int(rows.size(-2))):
        raise ValueError(f"行号越界：需落在 [0, {rows.size(-2)})，实得 [{int(ids.min())}, {int(ids.max())}]")
    if rows.device != ids.device:
        raise ValueError(f"rows 与 ids 需同设备，实得 {rows.device} / {ids.device}")
