"""attn_sw 方言件：因果单向滑窗注意力（每个位置只看得见自己前面有限一段）。

【做什么】前向 `forward(q, k, v, window, scale, out_dtype, target)`：q/k/v 都是
(..., heads, seq, dim) 的打包三路，返回与 q 同形、位宽 out_dtype 的输出；不可见的键（未来的、
或太久以前的）权重严格为 0，绝不偷偷分走一点质量。`forward_weights` 只摊开"谁看了谁多少"，
给测试与调试用，不进内核分发。
【怎么做】① 一条查询块行做一遍在线（online）softmax：先算分数 `Q Kᵀ`（一次乘加），按可见性
   把不可见格写成 NEG 哨兵，取本块行最大与历史行最大合并成新最大，旧最大与新最大之差取指数当
   "修正因子"，把已经攒下的分母和输出先乘这个因子，再累加本块的 `P V`——这样一次过就能得到
   与全量 softmax 逐位一致的结果，不需要把整张分数表存下来；② 分数取负 1e30 作哨兵后
   `exp(哨兵 - 新最大)` 会自然下溢成 0，但"新最大本身就是哨兵"（整块都被掩掉）时会算出
   exp(0)=1，所以权重重算那一步**必须再判一次可见性**、显式写 0，这是本件最容易踩的坑；
   ③ 网格上界只放常量（头数是编译期已知的），查询块号与键块号走动态串行/流水循环；尾块靠
   "行号 < seq"守卫与搬运边界谓词解决，所以 seq 不必被块宽整除；④ 内核全程 fp32，升降位由
   入口层负责。
【为什么】算法只写一份、两个 target 同一口径：被否方案一"CPU 走两趟（先求最大再求和）"——
   分数要重算一遍乘加，多花的量级正好是最贵的部分，而且两趟与在线两版的语义会各自漂移；
   被否方案二"把分数表整块存下来"——(bm, seq) 在大 seq 下直接爆本地内存。scale 固定成
   `1/sqrt(dim)` 与换底系数合并进一次乘法的口径沿用 P1（`attn_sw_mps`），别的系数走普通写法。
   昇腾那份与 CPU 那份同式，只是把"寄存器块"换成 UB/fragment、把 V 先转置成 (dim, bn) 再乘
   （该后端 L1 乘加路只认 transpose_B=True）；向量化归约留到云端 C2 按实测改写（本地不实编）。
"""
import tilelang

from typing import Any

import torch

from ascend.kernels import ascend_env

#: 查询块高与键块宽：与 P1 attn_sw_mps 同值
BLOCK_Q = 16
BLOCK_KV = 16
#: 不可见格子的哨兵：足够小、与任何真实分数相加都不会翻盘，且 exp(哨兵-真实最大) 下溢成 0
NEG = -1.0e30
#: 分母地板：一行一个键都没看见时兜住除零（正常情况下因果保证至少看得见自己）
FLOOR = 1.0e-30
#: 昇腾侧一次发射占用的块数与流水深度
NUM_BLOCKS = 8
VEC_THREADS = 64
NUM_STAGES = 2


def attn_sw_cpu_impl(Q, K, V, O, heads: int, dim: int, window: int, scale: float,
                     bm: int, bn: int):
    """CPU(c) 正文：每头一块，查询块串行推进，块内一次在线 softmax。

    白话：一摞问题交给一个人，他只能回头看自己前面那一小段答案；每翻新的一页答案就当场更新
    "到目前为止最亮的是多少、总权重是多少、加权内容是多少"，翻完把总分母一除就交货。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile）。
    """
    import tilelang.cpu.language as T

    seq = T.dynamic("seq")
    nqb = T.ceildiv(seq, bm)

    @T.prim_func
    def attn_impl(Q: T.Tensor((heads, seq, dim), "float32"), K: T.Tensor((heads, seq, dim), "float32"),
                  V: T.Tensor((heads, seq, dim), "float32"), O: T.Tensor((heads, seq, dim), "float32")):
        """被追踪的那一层：头数与每头宽度是常量，序列长度走动态维，窗口写死在比较里。

        白话：CPU 件这一份，每个位置只回头看有限的一段，一块一块地把能看见的内容按分数合起来；
        看不见的（后面的、太远前面的）一律不掺进来，连份额都不给，所以整块都被挡着时也不会
        凭空多出东西，最后按每行的合计除一下就是交货内容。
        
        """
        with T.Kernel(heads) as hh:
            Qs = T.alloc_local((bm, dim), "float32")
            Ks = T.alloc_local((bn, dim), "float32")
            Vs = T.alloc_local((bn, dim), "float32")
            Sc = T.alloc_local((bm, bn), "float32")
            Ps = T.alloc_local((bm, bn), "float32")
            Mx = T.alloc_local((bm,), "float32")
            Sm = T.alloc_local((bm,), "float32")
            Os = T.alloc_local((bm, dim), "float32")
            blkmax = T.alloc_var("float32")
            vnew = T.alloc_var("float32")
            vcorr = T.alloc_var("float32")
            vsum = T.alloc_var("float32")
            vrow = T.alloc_var("float32")
            for qb in T.serial(nqb):
                q_lo = qb * bm
                T.copy(Q[hh, q_lo, 0], Qs)
                for i in T.serial(bm):
                    Mx[i] = -1.0e30
                    Sm[i] = 0.0
                T.clear(Os)
                # 可见键块区间：[本行最老可见键, 本块最新可见键]，两端都按块宽对齐
                kb0 = T.floordiv(T.max(0, q_lo + 1 - window), bn)
                kb1 = T.floordiv(T.min(seq - 1, q_lo + bm - 1), bn) + 1
                for kb in T.serial(kb1 - kb0):
                    k_lo = (kb0 + kb) * bn
                    T.copy(K[hh, k_lo, 0], Ks)
                    T.copy(V[hh, k_lo, 0], Vs)
                    T.clear(Sc)
                    T.gemm(Qs, Ks, Sc, transpose_B=True)
                    for i in T.serial(bm):
                        i0 = q_lo + i
                        blkmax = -1.0e30
                        # 第一遍扫本块：掩码 + 乘尺度 + 找本块行最大
                        for j in T.serial(bn):
                            kk = k_lo + j
                            if (i0 >= kk) & (i0 - kk < window) & (kk < seq):
                                Sc[i, j] = Sc[i, j] * scale
                                if Sc[i, j] > blkmax:
                                    blkmax = Sc[i, j]
                            else:
                                Sc[i, j] = -1.0e30
                        vnew = Mx[i]
                        if blkmax > vnew:
                            vnew = blkmax
                        vcorr = T.exp(Mx[i] - vnew)
                        vsum = 0.0
                        # 第二遍扫本块：算权重并累计分母（哨兵行必须显式写 0，见模块 docstring）
                        for j in T.serial(bn):
                            kk = k_lo + j
                            if (i0 >= kk) & (i0 - kk < window) & (kk < seq):
                                Ps[i, j] = T.exp(Sc[i, j] - vnew)
                                vsum = vsum + Ps[i, j]
                            else:
                                Ps[i, j] = 0.0
                        Mx[i] = vnew
                        Sm[i] = Sm[i] * vcorr + vsum
                        for c in T.serial(dim):
                            Os[i, c] = Os[i, c] * vcorr
                    T.gemm(Ps, Vs, Os)
                for i in T.serial(bm):
                    i0 = q_lo + i
                    if i0 < seq:
                        vrow = Sm[i]
                        if vrow < FLOOR:
                            vrow = FLOOR
                        for c in T.serial(dim):
                            O[hh, i0, c] = Os[i, c] / vrow

    return attn_impl


def attn_sw_asc_impl(Q, K, V, O, heads: int, dim: int, window: int, scale: float,
                     bm: int, bn: int):
    """昇腾正文：与 CPU 同式的在线 softmax，乘加一律走该后端认的 transpose_B=True 那条路。

    白话：同一本账换一台机器算。这台机器的乘加只接受"第二个操作数倒过来摆"，所以 V 先在手边
    倒一遍摆放；掩码、取指数、除分母这些逐格活交给向量核块里的一堆小工一起做。

    两处与 CPU 版的写法差异（都不是算法差异，是该后端的约束）：① `P V` 不能直接乘（L1 乘加路
    只认 transpose_A=False/transpose_B=True），于是把 V 转置成 (dim, bn)，算 `Vᵀ Pᵀ` 得到
    (dim, bm) 的输出转置，最后再倒回来写出；② 逐格活放 SimtVF，行内的最大/求和两趟仍按串行写，
    向量化归约留待云端 C2 依实测改写（先例见 TileKernels 的 SIMD 件）。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")，本地不实编）。
    """
    import tilelang.ascend.language as T

    seq = T.dynamic("seq")

    @T.prim_func
    def attn_impl(Q: T.Tensor((heads, seq, dim), "float32"), K: T.Tensor((heads, seq, dim), "float32"),
                  V: T.Tensor((heads, seq, dim), "float32"), O: T.Tensor((heads, seq, dim), "float32")):
        """被追踪的那一层：查询块摊给多个工人，键值沿序列方向流水搬进来。

        白话：昇腾件这一份，每块查询自己从前往后扫能看见的那一段键值，边搬边把这一段的分数与
        已有结果合起来；这台机器的乘加只接受"第二个操作数倒过来摆"，所以值那一摞先在手边
        倒一遍，算完的输出再倒回来写出去。
        
        """
        with T.Kernel(heads * NUM_BLOCKS) as bx:
            hh = T.floordiv(bx, NUM_BLOCKS)
            core = bx % NUM_BLOCKS
            Q_ub = T.alloc_shared((bm, dim), "float32")
            K_ub = T.alloc_shared((bn, dim), "float32")
            V_ub = T.alloc_shared((bn, dim), "float32")
            Vt_ub = T.alloc_shared((dim, bn), "float32")
            Sc_ub = T.alloc_shared((bm, bn), "float32")
            Ps_ub = T.alloc_shared((bm, bn), "float32")
            Ot_ub = T.alloc_shared((dim, bm), "float32")
            stat_ub = T.alloc_shared((bm, 2), "float32")
            blkmax = T.alloc_var("float32")
            vnew = T.alloc_var("float32")
            vcorr = T.alloc_var("float32")
            vsum = T.alloc_var("float32")
            T.annotate_buffer_versions({Q_ub: NUM_STAGES, K_ub: NUM_STAGES, V_ub: NUM_STAGES})
            nqb = T.ceildiv(seq, bm)
            for unit in T.Persistent([nqb], NUM_BLOCKS, core,
                                     group_size=1, num_stages=NUM_STAGES):
                q_lo = unit * bm
                T.copy(Q[hh, q_lo, 0], Q_ub)
                with T.SimtVF(threads=VEC_THREADS):
                    for i in T.Parallel(bm):
                        stat_ub[i, 0] = -1.0e30
                        stat_ub[i, 1] = 0.0
                    for i, c in T.Parallel(bm, dim):
                        Ot_ub[c, i] = 0.0
                kb0 = T.floordiv(T.max(0, q_lo + 1 - window), bn)
                kb1 = T.floordiv(T.min(seq - 1, q_lo + bm - 1), bn) + 1
                for kb in T.serial(kb1 - kb0):
                    k_lo = (kb0 + kb) * bn
                    T.copy(K[hh, k_lo, 0], K_ub)
                    T.copy(V[hh, k_lo, 0], V_ub)
                    with T.SimtVF(threads=VEC_THREADS):
                        for c, j in T.Parallel(dim, bn):
                            Vt_ub[c, j] = V_ub[j, c]
                    T.clear(Sc_ub)
                    T.gemm(Q_ub, K_ub, Sc_ub, transpose_B=True)
                    # 本块行最大 → 新最大 → 修正因子（先串行，云端 C2 再向量化）
                    for i in T.serial(bm):
                        i0 = q_lo + i
                        blkmax = -1.0e30
                        for j in T.serial(bn):
                            kk = k_lo + j
                            if (i0 >= kk) & (i0 - kk < window) & (kk < seq):
                                Sc_ub[i, j] = Sc_ub[i, j] * scale
                                if Sc_ub[i, j] > blkmax:
                                    blkmax = Sc_ub[i, j]
                            else:
                                Sc_ub[i, j] = -1.0e30
                        vnew = stat_ub[i, 0]
                        if blkmax > vnew:
                            vnew = blkmax
                        vcorr = T.exp(stat_ub[i, 0] - vnew)
                        stat_ub[i, 0] = vnew
                        stat_ub[i, 1] = stat_ub[i, 1] * vcorr
                        for c in T.serial(dim):
                            Ot_ub[c, i] = Ot_ub[c, i] * vcorr
                    # 第二遍扫本块：算权重并累计分母。仍走串行——不可见格必须显式写 0
                    # （哨兵减哨兵等于 0，指数就成 1 了），而行内求和用原子加打在 UB 上
                    # 既不划算也无先例，串行累加最稳（向量化留待云端 C2）
                    for i in T.serial(bm):
                        i0 = q_lo + i
                        vsum = 0.0
                        for j in T.serial(bn):
                            kk = k_lo + j
                            if (i0 >= kk) & (i0 - kk < window) & (kk < seq):
                                Ps_ub[i, j] = T.exp(Sc_ub[i, j] - stat_ub[i, 0])
                                vsum = vsum + Ps_ub[i, j]
                            else:
                                Ps_ub[i, j] = 0.0
                        stat_ub[i, 1] = stat_ub[i, 1] + vsum
                    # V 先转置再乘：这条乘加路只认 transpose_B=True，出口是 (dim, bm) 的转置
                    T.gemm(Vt_ub, Ps_ub, Ot_ub, transpose_B=True)
                with T.SimtVF(threads=VEC_THREADS):
                    for i, c in T.Parallel(bm, dim):
                        i0 = q_lo + i
                        if i0 < seq:
                            # 并行区里不引共享标量临时量，除零直接就地用 max(分母, 地板) 兜住
                            O[hh, i0, c] = Ot_ub[c, i] / T.max(stat_ub[i, 1], FLOOR)

    return attn_impl


def plan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, window: int,
         scale: float, target: str | None = None) -> dict[str, Any] | None:
    """判据：三路同形且与出口同形、头数/维宽是编译期常量、fp32、连续、窗口与尺度合法。

    四条硬门槛：① 三路都是 (heads, seq, dim) 的 fp32 且长度一致、出口同形；② dim 被 16 整除
    （乘加件的宽度要求）；③ window 为正；④ scale 是一个有限的正实数（内核把它并进一次乘法）。
    任一不满足就回 None，由入口层换普通写法（seq 不要求被块宽整除：尾块由行号守卫兜）。

    白话：先看这三路的尺寸、记法合不合现成模具，不合就不勉强开模，直接手工做。
    """
    name = ascend_env.normalize_target(target)
    if q.dim() != 3 or q.shape != k.shape or q.shape != v.shape or q.shape != o.shape:
        return None
    if q.dtype != torch.float32 or o.dtype != torch.float32:
        return None
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and o.is_contiguous()):
        return None
    if int(window) < 1:
        return None
    heads, seq, dim = (int(x) for x in q.shape)
    if dim % 16:
        return None
    if not (scale > 0) or scale != scale:
        return None
    kwargs = {"heads": heads, "dim": dim, "window": int(window), "scale": float(scale),
              "bm": BLOCK_Q, "bn": BLOCK_KV}
    impl = attn_sw_asc_impl if name == ascend_env.TARGET_ASCEND else attn_sw_cpu_impl
    key = f"attn_sw[{name}]|h{heads}d{dim}w{int(window)}b{BLOCK_Q}x{BLOCK_KV}"
    return {"target": name, "key": key, "kwargs": kwargs, "impl": impl, "seq": seq}


def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
        spec: dict[str, Any]) -> bool:
    """真跑一次内核并把结果写进 o；编译产物取不到就返回 False（由入口层落回退）。

    缓存键不含 seq：它是 PrimFunc 里的动态符号，同一份产物服务任意长度（换长重编的秒级开销
    会吞掉收益，与 gemm/rope/add_ln 同一口径）。

    白话：模具没开成就如实说没开成，返回假字让上层换手工做法，绝不"假装算过"交出空表。
    """
    kernel = ascend_env.get_compiled(
        spec["key"],
        lambda: tilelang.compile(spec["impl"](q, k, v, o, **spec["kwargs"]),
                                 out_idx=[], **ascend_env.compile_kwargs(spec["target"])),
        spec["target"])
    if kernel is None:
        return False
    kernel(q, k, v, o)
    return True


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int,
            scale: float | None = None, out_dtype: torch.dtype = torch.float16,
            target: str | None = None) -> torch.Tensor:
    """因果单向滑窗注意力：返回与 q 同形、位宽 out_dtype 的输出（未来位置权重严格为 0）。

    白话：每个位置只能回头看自己前面的有限一段，把看到的内容按分数加权平均；后面的、太远的前面
    的都不参与，也绝不偷偷分走一点权重。

    :raises ValueError: 三路形状不一致、维度过少、窗口非正或设备不匹配时抛出。
    """
    _check_contract(q, k, v, window)
    dim = q.size(-1)
    scale = dim ** -0.5 if scale is None else float(scale)
    lead = q.shape[:-2]
    q3, k3, v3 = (t.reshape(-1, *t.shape[-2:]).contiguous() for t in (q, k, v))
    if ascend_env.active_backend(target, q.device) == ascend_env.TILELANG:
        a3, b3, c3 = q3.float(), k3.float(), v3.float()
        out3 = torch.empty(a3.shape, dtype=torch.float32, device=q.device)
        spec = plan(a3, b3, c3, out3, window, scale, target)
        if spec is not None and run(a3, b3, c3, out3, spec):
            return out3.to(out_dtype).reshape(lead + (out3.size(1), dim))
    acc = _eager(q3.float(), k3.float(), v3.float(), window, scale)
    return acc.to(out_dtype).reshape(lead + (q3.size(1), dim))


def forward_weights(q: torch.Tensor, k: torch.Tensor, window: int,
                    scale: float | None = None) -> torch.Tensor:
    """只算分数并做可见性掩码后的权重（fp32），给测试与调试用，不参与内核分发。

    白话：把"每个位置分别看了谁、各看了多少"摊开给人看，用来核对窗口方向有没有抄反。
    """
    _check_contract(q, k, k, window)
    dim = q.size(-1)
    scale = dim ** -0.5 if scale is None else float(scale)
    q3, k3 = q.reshape(-1, *q.shape[-2:]), k.reshape(-1, *k.shape[-2:])
    seq = q3.size(1)
    score = q3.to(torch.float32) @ k3.to(torch.float32).transpose(-1, -2) * scale
    return torch.softmax(_mask(score, seq, window), dim=-1)


def _check_contract(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> None:
    """校验三路同形、至少二维、窗口为正整数，并卡住设备一致。"""
    if q.dim() < 2:
        raise ValueError(f"q 至少要是 (seq, dim) 两维，实得 {q.dim()} 维")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"三路需同形，实得 {tuple(q.shape)} / {tuple(k.shape)} / {tuple(v.shape)}")
    if int(window) < 1:
        raise ValueError(f"window 需为正整数，实得 {window}")
    if q.device != k.device or q.device != v.device:
        raise ValueError("三路需在同一设备上")


def _mask(score: torch.Tensor, seq: int, window: int) -> torch.Tensor:
    """把不可见的 (查询位, 键位) 写成 -inf：因果 + 滑窗，两个方向都要拦。"""
    idx = torch.arange(seq, device=score.device)
    delta = idx[:, None] - idx[None, :]
    allowed = (delta >= 0) & (delta < int(window))
    return score.masked_fill(~allowed[None, :, :], float("-inf"))


def _eager(q3: torch.Tensor, k3: torch.Tensor, v3: torch.Tensor, window: int,
           scale: float) -> torch.Tensor:
    """回退路径：fp32 全量 softmax（尺子兼兜底），与内核同一可见性口径。"""
    seq = q3.size(1)
    score = q3 @ k3.transpose(-1, -2) * scale
    weight = torch.softmax(_mask(score, seq, window), dim=-1)
    return weight @ v3
