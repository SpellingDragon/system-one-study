"""add_ln 方言件：残差相加 + 层内整形（前向），以及它的闭式反向（一次求两遍均值就能收工）。

【做什么】前向 `forward(x, residual, weight, bias, eps, out_dtype, target)` 交回两样东西：
y（整形后的输出）和 h（相加后的原始 fp32 残差流，必须原样交给下一层）。反向
`backward(h, weight, dy, eps)` 交回 dx=dh=dr（因为 h = x + residual，两个输入的梯度是同一个
表达式）以及 weight/bias 的累计梯度。
【怎么做】① 一行交给一个处理单元走三遍：第一遍把整行加起来求均值，第二遍把"与均值的差"平方
   求和得到尺度，第三遍才按尺度把每格改到位——这是层内整形的标准闭式，不需要任何迭代求解；
   ② 反向同样闭式：先由 h 复算 xhat（与前向完全同式），再取两个均值 mean(wdy) 与
   mean(wdy*xhat)，dx = rstd * (wdy - 前者 - xhat*后者)，weight/bias 的梯度按列求和；
   ③ 行数动态、dim 是编译期常量；网格上界只放常量（块数），行号走动态串行/流水循环，与
   gemm/rope 三件同一口径；④ 内核全程 fp32（入口层负责升降位）；⑤ 尾块必须显式守卫：
   CPU 那份只有 T.copy 自带边界谓词，逐元素读写全局内存的行号要 `if row < rows` 拦一道，
   否则 rows 不被块宽整除时会写出界（昇腾那份照先例用 `valid = T.min(bm, rows - row0)` 的
   切片搬运 + 整块 if 守卫）。
【为什么】反向用闭式而不是"把前向再跑一遍拿中间量"：中间量只有均值与尺度两个标量，复算比
   存下来再回读更省带宽，而昇腾侧的回读要走搬运，搬运不许顺带转类型（vf_checker 明确
   "DMA copies cannot perform type casting"），少一份缓冲就少一处要小心的地方。被否方案一：
   用自动微分式反向（把前向的三遍循环反过来重放）——同一行要多跑两遍 dim，纯浪费；被否方案二：
   把 h 降成 fp16 存给下一层——残差流跨层累加，降位会把每层的微小偏差放大成系统性漂移，P1 的
   add_ln_kernel 也强制 h 必须是 fp32，这里同口径。闭式反向在昇腾方言有先例（TileKernels
   quant/norm_backward_asc.py 就是这个形状），不是本地生造。列梯度用 atomic_add 而非"每核一份
   偏量表 + 第二段归约"（先例的写法）：少一份全局缓冲、少一次核间同步；代价是云端 C1 要确认
   原子加在该后端的精度与吞吐，若不稳就换两段式（口径见 ln_bwd_asc_impl 注释）。
"""
import tilelang

from typing import Any

import torch

from ascend.kernels import ascend_env

#: 一次处理几行：与 P1 add_ln_mps.ROW_BLOCK 同值，非整除交给显式守卫
ROW_BLOCK = 16
#: 默认容差，与 torch_ref.DEFAULT_EPS / P1 同值；两边都要改必须同时改，否则对拍会白抖
DEFAULT_EPS = 1e-5
#: 昇腾侧一次发射占用的向量核块数与流水深度
NUM_BLOCKS = 8
VEC_THREADS = 64
NUM_STAGES = 2
#: 昇腾侧每块 (bm, dim) fragment 的元素数预算（宽行缩块用，云端 C1 再按实测 UB 调）
ASCEND_FRAG_BUDGET = 8192


def add_ln_cpu_impl(X, Res, G, Bt, Y, Hout, dim: int, eps: float, bm: int):
    """CPU(c) 前向正文：一行三遍（求均值 / 求尺度 / 改到位），h 以 fp32 原样落到第二出口。

    白话：一行数据交给一个人从头看到尾，看三遍——第一遍把数加起来并顺手把原始和记下来，
    第二遍看每格和平均差了多少、把差平方加起来，第三遍才按算好的尺度把每格改到位。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile）。
    """
    import tilelang.cpu.language as T

    rows = T.dynamic("rows")
    nb = T.ceildiv(rows, bm)

    @T.prim_func
    def add_ln_impl(X: T.Tensor((rows, dim), "float32"), Res: T.Tensor((rows, dim), "float32"),
                    G: T.Tensor((dim,), "float32"), Bt: T.Tensor((dim,), "float32"),
                    Y: T.Tensor((rows, dim), "float32"), Hout: T.Tensor((rows, dim), "float32")):
        """被追踪的那一层：特征轴宽度是常量，行数走动态维。

        白话：CPU 件这一份，先把两摞纸逐位合成一摞并原样留一份，再算出这一行的平均水平与散开
        程度，用这两把尺子把每格挪到位，最后乘倍率表、加小抄表；末尾不够一整块的格子直接
        跳过，不会拿零去搅和统计。
        
        """
        with T.Kernel(1) as bx:
            h = T.alloc_local((bm, dim), "float32")
            acc = T.alloc_var("float32")
            mu = T.alloc_var("float32")
            rs = T.alloc_var("float32")
            for blk in T.serial(nb):
                T.copy(X[blk * bm, 0], h)
                for r in T.serial(bm):
                    row = blk * bm + r
                    # 尾块守卫：搬运自带谓词，但下面这些逐元素读写全局内存的可不会自动拦
                    if row < rows:
                        for j in T.serial(dim):
                            h[r, j] = h[r, j] + Res[row, j]
                            Hout[row, j] = h[r, j]
                        acc = 0.0
                        for j in T.serial(dim):
                            acc = acc + h[r, j]
                        mu = acc / dim
                        acc = 0.0
                        for j in T.serial(dim):
                            acc = acc + (h[r, j] - mu) * (h[r, j] - mu)
                        rs = T.rsqrt(acc / dim + eps)
                        for j in T.serial(dim):
                            Y[row, j] = (h[r, j] - mu) * rs * G[j] + Bt[j]

    return add_ln_impl


def add_ln_asc_impl(X, Res, G, Bt, Y, Hout, dim: int, eps: float, bm: int):
    """昇腾前向正文：一叠行搬进 UB，在寄存器（fragment）上走完三遍，y 与 h 各一次整块搬出。

    白话：一次只处理一小叠纸，先在台面上把两叠合成一叠并留一份原样，再量平均、量散开程度，
    最后按这两把尺子把每格改到位，改完的两份各自放回自己的格子。

    写法对齐本方言的既有先例（TileKernels moe/normalize_weight_asc.py）：UB 只负责搬运，
    算的东西放 fragment，行内归约用 `T.reduce_sum(输入 fragment, 输出 fragment, dim=1)`，
    尾块用 `valid = T.min(bm, rows - row0)` 的切片搬运，整块再套一层 if 守卫。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")，本地不实编）。
    """
    import tilelang.ascend.language as T

    rows = T.dynamic("rows")

    @T.prim_func
    def add_ln_impl(X: T.Tensor((rows, dim), "float32"), Res: T.Tensor((rows, dim), "float32"),
                    G: T.Tensor((dim,), "float32"), Bt: T.Tensor((dim,), "float32"),
                    Y: T.Tensor((rows, dim), "float32"), Hout: T.Tensor((rows, dim), "float32")):
        """被追踪的那一层：一行一格的统计量放进手边小格表，搬运按有效行数切片。

        白话：昇腾件这一份，合成、求平均、求散开、挪位、乘倍率加小抄，这一串都在工人手边的小
        格子里做完；每轮只搬"真的存在"的那几行，末尾不够一整块的部分根本不进机器。
        
        """
        with T.Kernel(NUM_BLOCKS) as bx:
            h_ub = T.alloc_shared((bm, dim), "float32")
            r_ub = T.alloc_shared((bm, dim), "float32")
            y_ub = T.alloc_shared((bm, dim), "float32")
            g_ub = T.alloc_shared((dim,), "float32")
            b_ub = T.alloc_shared((dim,), "float32")
            T.annotate_buffer_versions({h_ub: NUM_STAGES, r_ub: NUM_STAGES, y_ub: NUM_STAGES})
            T.copy(G[0], g_ub)
            T.copy(Bt[0], b_ub)
            for blk in T.Persistent([T.ceildiv(rows, bm)], NUM_BLOCKS, bx,
                                    group_size=1, num_stages=NUM_STAGES):
                row0 = blk * bm
                valid = T.min(bm, rows - row0)
                if row0 < rows:
                    T.copy(X[row0:row0 + valid, :], h_ub[:valid, :])
                    T.copy(Res[row0:row0 + valid, :], r_ub[:valid, :])
                    with T.SimtVF(threads=VEC_THREADS):
                        h_fr = T.alloc_fragment((bm, dim), "float32")
                        d_fr = T.alloc_fragment((bm, dim), "float32")
                        sq_fr = T.alloc_fragment((bm, dim), "float32")
                        y_fr = T.alloc_fragment((bm, dim), "float32")
                        s_fr = T.alloc_fragment((bm,), "float32")
                        v_fr = T.alloc_fragment((bm,), "float32")
                        mu_fr = T.alloc_fragment((bm,), "float32")
                        rs_fr = T.alloc_fragment((bm,), "float32")
                        g_fr = T.alloc_fragment((dim,), "float32")
                        b_fr = T.alloc_fragment((dim,), "float32")
                        T.copy(h_ub, h_fr)
                        T.copy(g_ub, g_fr)
                        T.copy(b_ub, b_fr)
                        # 第一遍：残差相加（h 同时是下一层的残差流，原样搬回 Hout）
                        for i, j in T.Parallel(bm, dim):
                            h_fr[i, j] = h_fr[i, j] + r_ub[i, j]
                        T.copy(h_fr, h_ub)
                        # 第二遍：行内求和 → 均值；差的平方和 → 尺度
                        T.reduce_sum(h_fr, s_fr, dim=1)
                        for i in T.Parallel(bm):
                            mu_fr[i] = s_fr[i] / dim
                        for i, j in T.Parallel(bm, dim):
                            d_fr[i, j] = h_fr[i, j] - mu_fr[i]
                            sq_fr[i, j] = d_fr[i, j] * d_fr[i, j]
                        T.reduce_sum(sq_fr, v_fr, dim=1)
                        for i in T.Parallel(bm):
                            rs_fr[i] = T.rsqrt(v_fr[i] / dim + eps)
                        # 第三遍：按尺度改到位，乘该层倍率再补小抄
                        for i, j in T.Parallel(bm, dim):
                            y_fr[i, j] = d_fr[i, j] * rs_fr[i] * g_fr[j] + b_fr[j]
                        T.copy(y_fr, y_ub)
                    T.copy(h_ub[:valid, :], Hout[row0:row0 + valid, :])
                    T.copy(y_ub[:valid, :], Y[row0:row0 + valid, :])

    return add_ln_impl


def ln_bwd_cpu_impl(H, G, DY, DX, Dg, Db, dim: int, eps: float, bm: int):
    """CPU(c) 反向正文：一行四趟（复算尺度 → 两个均值 → 改每格 → 累计列梯度），闭式不迭代。

    白话：给每行先按前向同款量一遍平均和散开程度，再把"该往哪儿挪"的数先各自乘上该层的倍率、
    量出两个平均，然后每格一次算完；列方向上还要把这一批的贡献加总成两张倍率表和小抄表。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile）。
    """
    import tilelang.cpu.language as T

    rows = T.dynamic("rows")
    nb = T.ceildiv(rows, bm)

    @T.prim_func
    def ln_bwd_impl(H: T.Tensor((rows, dim), "float32"), G: T.Tensor((dim,), "float32"),
                    DY: T.Tensor((rows, dim), "float32"), DX: T.Tensor((rows, dim), "float32"),
                    Dg: T.Tensor((dim,), "float32"), Db: T.Tensor((dim,), "float32")):
        """被追踪的那一层：与正向同一套形状约定，出口另附两列按列汇总的账。

        白话：CPU 件这一份，已知"改完的格子该挪多少"，先按行的平均与散开把账倒推回合成那一摞
        的每格，再顺带把这一批对倍率表和小抄表的总影响各累成一列；汇总列的初值在块顶就清好，
        行数不够整块时同样只碰有效行。
        
        """
        with T.Kernel(1) as bx:
            h = T.alloc_local((bm, dim), "float32")
            dy = T.alloc_local((bm, dim), "float32")
            w = T.alloc_local((bm, dim), "float32")
            xh = T.alloc_local((bm, dim), "float32")
            acc = T.alloc_var("float32")
            mu = T.alloc_var("float32")
            rs = T.alloc_var("float32")
            m1 = T.alloc_var("float32")
            m2 = T.alloc_var("float32")
            for blk in T.serial(nb):
                T.copy(H[blk * bm, 0], h)
                T.copy(DY[blk * bm, 0], dy)
                for r in T.serial(bm):
                    row = blk * bm + r
                    if row < rows:
                        acc = 0.0
                        for j in T.serial(dim):
                            acc = acc + h[r, j]
                        mu = acc / dim
                        acc = 0.0
                        for j in T.serial(dim):
                            xh[r, j] = h[r, j] - mu
                            acc = acc + xh[r, j] * xh[r, j]
                        rs = T.rsqrt(acc / dim + eps)
                        for j in T.serial(dim):
                            xh[r, j] = xh[r, j] * rs
                            w[r, j] = dy[r, j] * G[j]
                        acc = 0.0
                        for j in T.serial(dim):
                            acc = acc + w[r, j]
                        m1 = acc / dim
                        acc = 0.0
                        for j in T.serial(dim):
                            acc = acc + w[r, j] * xh[r, j]
                        m2 = acc / dim
                        for j in T.serial(dim):
                            DX[row, j] = rs * (w[r, j] - m1 - xh[r, j] * m2)
                            # 列梯度用原始 dy（倍率还没乘上去的那份），且是"加一笔"，
                            # 所以入口层必须先把它清零
                            Dg[j] = Dg[j] + dy[r, j] * xh[r, j]
                            Db[j] = Db[j] + dy[r, j]

    return ln_bwd_impl


def ln_bwd_asc_impl(H, G, DY, DX, Dg, Db, dim: int, eps: float, bm: int):
    """昇腾反向正文：与 CPU 版同一套闭式，中间量走 UB/fragment，列梯度用原子加回落全局。

    白话：台面上摊开这一小叠纸，先把每行的平均和散开量出来，再量两个修正用的平均，然后一次
    算好每格该挪多少；列上那两张总账要跨块累加，所以用"看到就加一笔"的原子写法，避免互相盖掉。

    两处待云端 C1 确认的方言选择：① 列方向跨块累计用 `T.atomic_add`（该方言确实导出了这个
    op），先例里另有"每核一份偏量表 + 第二段归约 + 核间同步"的写法（norm_backward_asc.py），
    若原子加不稳就换过去；② 归约一律按先例在 fragment 上做（`T.reduce_sum(frag, frag, dim=1)`），
    不写成对 UB 直接归约，也不带表达式——要归约的量先物化成一份 fragment。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")，本地不实编）。
    """
    import tilelang.ascend.language as T

    rows = T.dynamic("rows")

    @T.prim_func
    def ln_bwd_impl(H: T.Tensor((rows, dim), "float32"), G: T.Tensor((dim,), "float32"),
                    DY: T.Tensor((rows, dim), "float32"), DX: T.Tensor((rows, dim), "float32"),
                    Dg: T.Tensor((dim,), "float32"), Db: T.Tensor((dim,), "float32")):
        """被追踪的那一层：每格回推与按列汇总同块完成，汇总走原子累加。

        白话：昇腾件这一份，工人各自把手里这一块的账倒推算完，再把该进总账的两笔一笔笔添到
        公共列上；添的动作可以并发，同一列被多块来添也各自都算数，不会互相盖掉。
        
        """
        with T.Kernel(NUM_BLOCKS) as bx:
            h_ub = T.alloc_shared((bm, dim), "float32")
            dy_ub = T.alloc_shared((bm, dim), "float32")
            dx_ub = T.alloc_shared((bm, dim), "float32")
            g_ub = T.alloc_shared((dim,), "float32")
            T.annotate_buffer_versions({h_ub: NUM_STAGES, dy_ub: NUM_STAGES, dx_ub: NUM_STAGES})
            T.copy(G[0], g_ub)
            for blk in T.Persistent([T.ceildiv(rows, bm)], NUM_BLOCKS, bx,
                                    group_size=1, num_stages=NUM_STAGES):
                row0 = blk * bm
                valid = T.min(bm, rows - row0)
                if row0 < rows:
                    T.copy(H[row0:row0 + valid, :], h_ub[:valid, :])
                    T.copy(DY[row0:row0 + valid, :], dy_ub[:valid, :])
                    with T.SimtVF(threads=VEC_THREADS):
                        h_fr = T.alloc_fragment((bm, dim), "float32")
                        dy_fr = T.alloc_fragment((bm, dim), "float32")
                        d_fr = T.alloc_fragment((bm, dim), "float32")
                        sq_fr = T.alloc_fragment((bm, dim), "float32")
                        w_fr = T.alloc_fragment((bm, dim), "float32")
                        xh_fr = T.alloc_fragment((bm, dim), "float32")
                        wd_fr = T.alloc_fragment((bm, dim), "float32")
                        dx_fr = T.alloc_fragment((bm, dim), "float32")
                        s_fr = T.alloc_fragment((bm,), "float32")
                        v_fr = T.alloc_fragment((bm,), "float32")
                        mu_fr = T.alloc_fragment((bm,), "float32")
                        rs_fr = T.alloc_fragment((bm,), "float32")
                        m1_fr = T.alloc_fragment((bm,), "float32")
                        m2_fr = T.alloc_fragment((bm,), "float32")
                        g_fr = T.alloc_fragment((dim,), "float32")
                        T.copy(h_ub, h_fr)
                        T.copy(dy_ub, dy_fr)
                        T.copy(g_ub, g_fr)
                        # 复算前向的两个统计量（与前向完全同式，杜绝"存下来再回读"）
                        T.reduce_sum(h_fr, s_fr, dim=1)
                        for i in T.Parallel(bm):
                            mu_fr[i] = s_fr[i] / dim
                        for i, j in T.Parallel(bm, dim):
                            d_fr[i, j] = h_fr[i, j] - mu_fr[i]
                            sq_fr[i, j] = d_fr[i, j] * d_fr[i, j]
                        T.reduce_sum(sq_fr, v_fr, dim=1)
                        for i in T.Parallel(bm):
                            rs_fr[i] = T.rsqrt(v_fr[i] / dim + eps)
                        for i, j in T.Parallel(bm, dim):
                            xh_fr[i, j] = d_fr[i, j] * rs_fr[i]
                            w_fr[i, j] = dy_fr[i, j] * g_fr[j]
                        # 闭式反向的两个均值：mean(wdy) 与 mean(wdy*xhat)
                        T.reduce_sum(w_fr, m1_fr, dim=1)
                        for i, j in T.Parallel(bm, dim):
                            wd_fr[i, j] = w_fr[i, j] * xh_fr[i, j]
                        T.reduce_sum(wd_fr, m2_fr, dim=1)
                        for i in T.Parallel(bm):
                            m1_fr[i] = m1_fr[i] / dim
                            m2_fr[i] = m2_fr[i] / dim
                        for i, j in T.Parallel(bm, dim):
                            dx_fr[i, j] = rs_fr[i] * (w_fr[i, j] - m1_fr[i] - xh_fr[i, j] * m2_fr[i])
                            # 列方向跨块累计，只能"加一笔"而不是"盖一笔"
                            T.atomic_add(Dg[j], dy_fr[i, j] * xh_fr[i, j])
                            T.atomic_add(Db[j], dy_fr[i, j])
                        T.copy(dx_fr, dx_ub)
                    T.copy(dx_ub[:valid, :], DX[row0:row0 + valid, :])

    return ln_bwd_impl


def _pick_bm(dim: int, name: str) -> int:
    """块宽：CPU 走满 ROW_BLOCK（验证口径），昇腾按 fragment 预算收缩，宽行绝不一上来就摊满。

    白话：手工看一行要看几眼就够，机器把每行摊在随手可及的小格子里，行越宽，一次能摊的行数
    就得越少，否则小格子先满了。
    """
    if name != ascend_env.TARGET_ASCEND:
        return ROW_BLOCK
    return max(1, min(ROW_BLOCK, ASCEND_FRAG_BUDGET // max(dim, 1)))


def _plan_fwd(X, Res, G, Bt, Y, Hout, eps, target):
    """前向判据：二维等形、fp32、连续；不过关回 None 让入口走回退（不硬凑）。

    白话：先看这几摞纸是不是一样高一样宽、记法是不是细尺子、东西是不是真摊在一整张桌上，
    不合模具就换手工。
    """
    name = ascend_env.normalize_target(target)
    if X.dim() != 2 or X.shape != Res.shape or X.shape != Y.shape or X.shape != Hout.shape:
        return None
    if X.dtype != torch.float32 or Res.dtype != torch.float32:
        return None
    if G.dtype != torch.float32 or Bt.dtype != torch.float32:
        return None
    rows, dim = int(X.size(0)), int(X.size(1))
    if G.numel() != dim or Bt.numel() != dim:
        return None
    if not all(t.is_contiguous() for t in (X, Res, G, Bt, Y, Hout)):
        return None
    bm = _pick_bm(dim, name)
    kwargs = {"dim": dim, "eps": float(eps), "bm": bm}
    impl = add_ln_asc_impl if name == ascend_env.TARGET_ASCEND else add_ln_cpu_impl
    key = f"add_ln[{name}]|d{dim}e{float(eps)}b{bm}"
    return {"target": name, "key": key, "kwargs": kwargs, "impl": impl, "rows": rows}


def _plan_bwd(H, G, DY, DX, Dg, Db, eps, target):
    """反向判据：与前向同一套形状/位宽/连续性要求，外加两张列梯度表（长度等于 dim、须已清零）。

    白话：除了看纸摞齐不齐，还要确认那两张总账表是空的——本件对它们是"往上加一笔"，
    不是"重新写一遍"。
    """
    name = ascend_env.normalize_target(target)
    if H.dim() != 2 or H.shape != DY.shape or H.shape != DX.shape:
        return None
    if H.dtype != torch.float32 or DY.dtype != torch.float32 or G.dtype != torch.float32:
        return None
    if Dg.dtype != torch.float32 or Db.dtype != torch.float32:
        return None
    rows, dim = int(H.size(0)), int(H.size(1))
    if G.numel() != dim or Dg.numel() != dim or Db.numel() != dim:
        return None
    if not all(t.is_contiguous() for t in (H, G, DY, DX, Dg, Db)):
        return None
    bm = _pick_bm(dim, name)
    kwargs = {"dim": dim, "eps": float(eps), "bm": bm}
    impl = ln_bwd_asc_impl if name == ascend_env.TARGET_ASCEND else ln_bwd_cpu_impl
    key = f"ln_bwd[{name}]|d{dim}e{float(eps)}b{bm}"
    return {"target": name, "key": key, "kwargs": kwargs, "impl": impl, "rows": rows}


def _run(args: tuple, spec: dict[str, Any]) -> bool:
    """取（或首编）内核并写出口；False 表示模具没开成，由入口层回退（不许假装算过）。

    缓存键不含行数：同一份产物服务任意批行数（换批重编的秒级开销会吞掉收益）。

    白话：模具只开第一次，之后不管这一批有多少行都用同一个；开不了就如实报回去。
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


def forward(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
            eps: float = DEFAULT_EPS, out_dtype: torch.dtype = torch.float16,
            target: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """前向：交回 (y, h)；h 是 x+residual 的 fp32 残差流，y 降到 out_dtype。

    白话：把两摞纸合成一摞并留个原样副本，再按"平均+散开"这两把尺子把每格改到位，
    最后按需要的粗细交货。

    :raises ValueError: x 与 residual 形状不一致，或 weight/bias 长度不等于特征轴时抛出。
    """
    _check(x, residual, weight, bias)
    name = ascend_env.normalize_target(target)
    if ascend_env.active_backend(name, x.device) == ascend_env.TILELANG:
        x32, r32 = x.float().contiguous(), residual.float().contiguous()
        w32, b32 = weight.float().contiguous(), bias.float().contiguous()
        y32, h = torch.empty_like(x32), torch.empty_like(x32)
        spec = _plan_fwd(x32, r32, w32, b32, y32, h, eps, name)
        if spec is not None and _run((x32, r32, w32, b32, y32, h), spec):
            return y32.to(out_dtype), h
    h = (x.float() + residual.float()).to(torch.float32)
    y = _ln(h, weight, bias, eps).to(out_dtype)
    return y, h


def backward(h: torch.Tensor, weight: torch.Tensor, dy: torch.Tensor,
             eps: float = DEFAULT_EPS,
             target: str | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """反向闭式：交回 (dh, dweight, dbias)；dh 同时是 x 与 residual 的梯度（同一条加法）。

    白话：已知"改完的格子该挪多少"，倒推"合成那一摞里每格该挪多少"，顺带把这一批对倍率表
    和小抄表的总影响各加总成一张表。

    :raises ValueError: h 不是二维或 weight/dy 形状不匹配时抛出。
    """
    if h.dim() != 2 or h.shape != dy.shape:
        raise ValueError(f"h 与 dy 需同为二维且同形，实得 {tuple(h.shape)} / {tuple(dy.shape)}")
    if weight.numel() != h.size(1):
        raise ValueError(f"weight 长度需等于特征轴 {h.size(1)}，实得 {weight.numel()}")
    name = ascend_env.normalize_target(target)
    if ascend_env.active_backend(name, h.device) == ascend_env.TILELANG:
        h32 = h.float().contiguous()
        w32 = weight.float().contiguous()
        dy32 = dy.float().contiguous()
        dx = torch.zeros_like(h32)
        dg = torch.zeros(h.size(1), dtype=torch.float32, device=h.device)
        db = torch.zeros(h.size(1), dtype=torch.float32, device=h.device)
        spec = _plan_bwd(h32, w32, dy32, dx, dg, db, eps, name)
        if spec is not None and _run((h32, w32, dy32, dx, dg, db), spec):
            return dx, dg, db
    return _ln_bwd_torch(h, weight, dy, eps)


def _check(x, residual, weight, bias) -> None:
    """入口契约：x/residual 同形二维、weight/bias 长度等于特征轴。"""
    if x.dim() != 2 or x.shape != residual.shape:
        raise ValueError(f"x 与 residual 需同为二维且同形，实得 {tuple(x.shape)} / {tuple(residual.shape)}")
    if weight.numel() != x.size(1) or bias.numel() != x.size(1):
        raise ValueError(f"weight/bias 长度需等于特征轴 {x.size(1)}")


def _ln(h: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """torch 版层内整形（回退尺子）：有偏方差（除以 dim），与内核同口径。"""
    mu = h.mean(-1, keepdim=True)
    var = (h - mu).pow(2).mean(-1, keepdim=True)
    return (h - mu) * torch.rsqrt(var + eps) * weight.float() + bias.float()


def _ln_bwd_torch(h, weight, dy, eps) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """torch 版闭式反向（回退尺子）：与内核同一套两个均值的表达式。"""
    hf = h.float()
    mu = hf.mean(-1, keepdim=True)
    var = (hf - mu).pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    xhat = (hf - mu) * rstd
    wdy = dy.float() * weight.float()
    dx = rstd * (wdy - wdy.mean(-1, keepdim=True)
                 - xhat * (wdy * xhat).mean(-1, keepdim=True))
    return dx, (dy.float() * xhat).sum(0), dy.float().sum(0)
