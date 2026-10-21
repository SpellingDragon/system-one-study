"""attn_sw 方言件：因果单向滑窗注意力（每个位置只看得见自己前面有限一段）。

【做什么】前向 `forward(q, k, v, window, scale, out_dtype, target)`：q/k/v 都是
(..., heads, seq, dim) 的打包三路，返回与 q 同形、位宽 out_dtype 的输出；不可见的键（未来的、
或太久以前的）权重严格为 0，绝不偷偷分走一点质量。`forward_weights` 只摊开"谁看了谁多少"，
给测试与调试用，不进内核分发。
【怎么做】① 一条查询块行做一遍在线（online）softmax（CPU 件的路子）：先算分数 `Q Kᵀ`（一次乘加），按可见性
   把不可见格写成 NEG 哨兵，取本块行最大与历史行最大合并成新最大，旧最大与新最大之差取指数当
   "修正因子"，把已经攒下的分母和输出先乘这个因子，再累加本块的 `P V`——这样一次过就能得到
   与全量 softmax 逐位一致的结果，不需要把整张分数表存下来；② 分数取负 1e30 作哨兵后
   `exp(哨兵 - 新最大)` 会自然下溢成 0，但"新最大本身就是哨兵"（整块都被掩掉）时会算出
   exp(0)=1，所以**分块形态**的权重重算那一步必须再判一次可见性、显式写 0，这是本件最容易踩的
   坑；昇腾那份是逐行形态，一行内窗内至少有一格真实分数（自己看得见自己），新最大不可能出自
   哨兵，所以那一步补判在昇腾件里天然不需要（不是省掉了，是不存在）；③ 网格上界只放常量（头数
   是编译期已知的），序列长度走动态符号：CPU 那份查询块号与键块号走串行流水，昇腾那份逐行按
   NUM_BLOCKS 跨步领行；尾行靠"行号 < seq"的行程数上界解决，所以 seq 不必被块宽整除；④ 内核
   全程 fp32，升降位由入口层负责。
【为什么】算法只写一份、两个 target 同一口径：被否方案一"CPU 走两趟（先求最大再求和）"——
   分数要重算一遍乘加，多花的量级正好是最贵的部分，而且两趟与在线两版的语义会各自漂移；
   被否方案二"把分数表整块存下来"——(bm, seq) 在大 seq 下直接爆本地内存。scale 固定成
   `1/sqrt(dim)` 与换底系数合并进一次乘法的口径沿用 P1（`attn_sw_mps`），别的系数走普通写法。
   被否方案三（P1-4 合流时新增）：昇腾侧继续沿用 950 代际那套"UB/fragment + 线程级 SIMT 并行
   掩码 + `T.gemm(transpose_B=True)` 两次乘加"的载体——该代际没有 SIMT 硬件模型，这一族原语在
   patched 910B tilelang 下实测编不出（D-int1），而且本件的可见性判据是逐格的、乘加规模又小，
   落到标量面反而是这台机器的正路：分数与权重各存一个窗宽的 UB 数组（**两个都被访问**，
   避开 G-C8 的"命名空句柄"形态），行内两趟 softmax 走 `expf`（compat §11 软件件）＋原生
   `max`/除法，出口逐格标量写回 GM。真机数值口径不变（attempts/E 的 stage 形态已实测
   rel=1.41e-07，见 P1-1c）。
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
#: 昇腾侧一次发射占用的块数（每头再切 NUM_BLOCKS 份逐行领）。910B 标量面件没有"每块多少线程、
#: 流水几级"这两维，原来的每块线程数与流水深度两个常量随 SIMT 载体一起撤掉，别留成假配置项。
NUM_BLOCKS = 8


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
    """昇腾正文（910B / dav-2201 可编形态）：逐查询行的整窗两趟 softmax，分数与权重各占一格 UB。

    白话：把一整摞问题按人头分给 NUM_BLOCKS 个工人，每人隔行领一条：先把这一行能看见的那一段
    答案逐格打分存到手边（存不下的窗外格直接写哨兵），再在手边找最亮的一格、逐格取指数当权重
    并累出分母，最后按权重把值加权、除以分母交货。

    四条 910B 方言纪律（P1-4 合流，凭据 attempts/E/e_attn_sw_910b.py + RESULT.md §0 + P1-1c
    真机 rel=1.41e-07 的 stage 形态）：
      * 载体只有 `T.Kernel` + `T.Vector()` + 纯 `T.serial`；不碰 950 那族线程级 SIMT 并行载体，
        也不用 `T.gemm`/`T.Persistent`/流水搬运——该代际无 SIMT 硬件模型，那些件实测编不出
        （D-int1）。分数就是窗长个向量的内积，标量面逐格乘加才是这台机器的正路，V 也不必
        先转置再乘（那条 transpose_B=True 的规矩属于 L1 乘加路，本件不走它）。
      * 数学件只吃三样：四则、`T.max`、`T.exp`→`expf`（compat §11 软件件，定标
        max_rel_err=1.103e-06）；不碰 log/sqrt/rsqrt/三角（G-E1：标量面连 sinf/cosf 都没有）。
      * 可见性口径与 CPU 件、与 `_eager`/`_mask` 逐字一致：查询行 i 看得见键
        kk ∈ [max(0, i-window+1), i]，两个方向（未来、太久以前）都拦，窗外权重**严格**为 0——
        靠"哨兵减真实最大 → expf 下溢钳位成 0"结构性保证，不是近似。逐行形态下行最大必出自
        窗内（至少看得见自己），所以生产分块件里"重算权重再判一次可见性"的补判在这里不存在。
      * UB 用两个都被访问的窗宽数组（sc_ub/ps_ub），刻意避开 G-C8 的"被访问 shared 恰好 1 个
        ⇒ 命名空句柄"形态；其余暂存走 `T.alloc_var`。

    两处与其它件不同的写法，都是本接口的既有约束、不是算法差异：① 序列长度是动态符号，故每
    （头，核）按 `NUM_BLOCKS` 跨步领行，行程数 `ceildiv(seq - core, NUM_BLOCKS)` 把尾行天然挡在
    界外；② 窗宽是编译期常量而 seq 是活的，`window > seq` 时循环上界会越过最后一行，所以取数
    下标统一夹到 `min(lo + j, seq - 1)`：被夹到的那一侧本来就被哨兵判为不可见、值不参与结果，
    夹位只把非法地址变成合法地址，数值逐位不变。`bm`/`bn` 在本方言下不再参与寻址（逐行形态），
    留形参只为 `plan` 的 kwargs 与缓存键不动。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")，P1-4 起本地实编判 .o）。
    """
    import tilelang.ascend.language as T

    seq = T.dynamic("seq")
    win = int(window)          # 窗宽是编译期常量（由 plan 带进来的 python int）

    @T.prim_func
    def attn_impl(Q: T.Tensor((heads, seq, dim), "float32"), K: T.Tensor((heads, seq, dim), "float32"),
                  V: T.Tensor((heads, seq, dim), "float32"), O: T.Tensor((heads, seq, dim), "float32")):
        """被追踪的那一层：头数与每头宽度是常量，序列长度走动态维，窗口写死在比较里。

        白话：昇腾件这一份，每个工人隔行领一条查询，逐格给自己看得见的那一段打分、找最亮、
        取指数、按权重把值合起来再除一下；看不见的连份额都不给。
        """
        with T.Kernel(heads * NUM_BLOCKS) as bx:
            hh = T.floordiv(bx, NUM_BLOCKS)
            core = bx % NUM_BLOCKS
            sc_ub = T.alloc_shared((win,), "float32")   # 本行整窗分数（含窗外哨兵）
            ps_ub = T.alloc_shared((win,), "float32")   # 本行整窗权重（窗外自然下溢成 0）
            m = T.alloc_var("float32", T.float32(0.0))
            l = T.alloc_var("float32", T.float32(0.0))
            acc = T.alloc_var("float32", T.float32(0.0))
            with T.Vector():
                for rr in T.serial(T.ceildiv(seq - core, NUM_BLOCKS)):
                    i = core + rr * NUM_BLOCKS           # 本行查询位置（一定 < seq）
                    lo = T.max(i - win + 1, 0)           # 可见窗左端（因果 + 滑窗，两头都拦）
                    n = i - lo + 1                       # 可见键数 ∈ [1, win]
                    # 趟1：整窗打分进 UB，窗外（j >= n）写哨兵；下标夹到 seq-1 只影响哨兵格
                    for j in T.serial(win):
                        kk = T.min(lo + j, seq - 1)
                        acc = T.float32(0.0)
                        for c in T.serial(dim):
                            acc = acc + Q[hh, i, c] * K[hh, kk, c]
                        sc_ub[j] = T.if_then_else(j < n, scale * acc, T.float32(NEG))
                    # 趟2：行最大（哨兵不会赢：窗内至少有一格真实分数）
                    m = T.float32(NEG)
                    for j in T.serial(win):
                        m = T.max(m, sc_ub[j])
                    # 趟3：权重与分母（哨兵侧 expf(NEG - 真实最大) 下溢成 0，自见格自带 exp(0)=1）
                    l = T.float32(0.0)
                    for j in T.serial(win):
                        ps_ub[j] = T.exp(sc_ub[j] - m)
                        l = l + ps_ub[j]
                    # 趟4：按权重合值并归一；分母仍按接口的地板兜一下（与 CPU 件同口径）
                    for c in T.serial(dim):
                        acc = T.float32(0.0)
                        for j in T.serial(win):
                            acc = acc + ps_ub[j] * V[hh, T.min(lo + j, seq - 1), c]
                        O[hh, i, c] = acc / T.max(l, T.float32(FLOOR))

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
