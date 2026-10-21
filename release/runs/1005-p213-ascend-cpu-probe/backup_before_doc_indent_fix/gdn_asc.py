"""GDN（Gated DeltaRule）方言件：线性注意力的逐 token 递推前向；反向本波如实标 partial。

【做什么】前向 `forward(q, k, v, g, beta, out_dtype, target)` 按头维护一份状态矩阵 S (dk, dv)，
逐 token 走五步：① 衰减 `S *= exp(g[t])`（g 是每键通道的对数衰减，这就是 "gated"）；② 用当前键
召回一遍 `pred = kᵀ S`；③ 算 delta 修正项 `u = beta * (v[t] - pred)`（"该写进去的" 减 "已经有的"）；
④ 秩一写回 `S += k ⊗ u`；⑤ 读出 `o[t] = q[t]ᵀ S`。交回 (heads, seq, dv) 的输出。
【怎么做】① 这份状态是**跨 token 串行**的：递推关系不允许在时间轴上并行，所以两份正文都把 token
   循环写成串行；并行度只出现在通道轴（dk/dv）上——CPU 那份用本地二维缓冲按格扫，昇腾那份把 S
   常驻 UB、通道轴交给 SimtVF 的 T.Parallel；② 召回/读出都是"沿 dk 归约"，本件先按串行归约写
   清楚语义（向量化归约留待云端 C2 依实测改写，先例是 TileKernels 的 SIMD 件）；③ dk/dv/头数是
   编译期常量，seq 是动态符号，网格上界只放常量（与 gemm/rope/add_ln/attn_sw 同一口径）；④ q/k
   的 L2 归一与短卷积都在本件之外（上游负责），本件吃的是已经算好的 g/beta；⑤ 全程 fp32。
【为什么】反向（对 q/k/g/beta 与初始状态的梯度）**本波不做内核化**，如实标 partial 并交回 torch
   自动微分：Gated DeltaRule 的反向在开源侧只有分块（chunked）+ WY 表示那一套写法，本地能看到的
   先例全是 CUDA 方言（tilelang/examples/gdn 十件），TileKernels 里也没有 delta rule 件，昇腾方言
   没有可对齐的写法。被否方案：硬凑一份 naive 反向方言件——递推反向要再存一份跨 token 的伴随状态，
   形状与流水都要重设计，本地无法编译验证，硬写出来的东西到云端 C2 只会变成"看着像但对不上"的
   假绿源（红线 R14）。所以本件的口径是：前向进内核，反向走 torch，并用 `BWD_STATUS` 与
   `ascend_env.record_blocker` 把这个边界写成机器可见的状态。
"""
import tilelang

from typing import Any

import torch

from ascend.kernels import ascend_env

#: 反向本波未内核化（原因见模块 docstring 的【为什么】），测试与上层按这个标记判断口径
BWD_STATUS = "partial"
#: 昇腾侧一次发射占用的块数（状态矩阵按头常驻 UB，所以只有通道轴可切块）
VEC_THREADS = 64
NUM_STAGES = 2


def gdn_cpu_impl(Q, K, V, G, Beta, O, heads: int, dk: int, dv: int):
    """CPU(c) 正文：每头一块，token 串行递推，通道轴扫本地缓冲。

    白话：每人守着一块小黑板（状态）。每来一条新记录：先把黑板整片按比例淡掉一点，再照当前键
    把黑板上已有的内容读出来当"预期"，用真值减预期得到"这次真正要补的"，乘上这一步的下笔力度
    写回黑板，最后照查询把黑板读一遍就是这一步的输出。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile）。
    """
    import tilelang.cpu.language as T

    seq = T.dynamic("seq")

    @T.prim_func
    def gdn_impl(Q: T.Tensor((heads, seq, dk), "float32"), K: T.Tensor((heads, seq, dk), "float32"),
                 V: T.Tensor((heads, seq, dv), "float32"), G: T.Tensor((heads, seq, dk), "float32"),
                 Beta: T.Tensor((heads, seq), "float32"), O: T.Tensor((heads, seq, dv), "float32")):
        """被追踪的那一层：头数、键宽、值宽是常量，时间轴走动态维并全程串行。

        白话：CPU 件这一份，每头守一块小黑板，每来一条新记录就走五步——整片按比例淡掉、照当前
        键把已有内容读出来当预期、真值减预期乘上下笔力度补进去、最后照查询读一遍就是这一步
        的输出。时间轴一步压不得，所以这条链只能串行走。
        
        """
        with T.Kernel(heads) as hh:
            S = T.alloc_local((dk, dv), "float32")
            kt = T.alloc_local((dk,), "float32")
            qt = T.alloc_local((dk,), "float32")
            vt = T.alloc_local((dv,), "float32")
            gt = T.alloc_local((dk,), "float32")
            dec = T.alloc_local((dk,), "float32")
            pred = T.alloc_var("float32")
            upd = T.alloc_var("float32")
            acc = T.alloc_var("float32")
            T.clear(S)  # 初始状态恒为零：本件不带跨段状态入口（跨段续推属上层 KV 的事）
            for t in T.serial(seq):
                T.copy(K[hh, t, 0], kt)
                T.copy(Q[hh, t, 0], qt)
                T.copy(V[hh, t, 0], vt)
                T.copy(G[hh, t, 0], gt)
                # ① 衰减：先把每键通道的衰减系数算出来，再整片淡掉黑板
                for kk in T.serial(dk):
                    dec[kk] = T.exp(gt[kk])
                for kk in T.serial(dk):
                    for vv in T.serial(dv):
                        S[kk, vv] = S[kk, vv] * dec[kk]
                bt = Beta[hh, t]
                for vv in T.serial(dv):
                    # ② 召回：照当前键把黑板读一遍
                    pred = 0.0
                    for kk in T.serial(dk):
                        pred = pred + S[kk, vv] * kt[kk]
                    # ③ delta 修正：真值减预期，乘下笔力度
                    upd = (vt[vv] - pred) * bt
                    # ④ 秩一写回
                    for kk in T.serial(dk):
                        S[kk, vv] = S[kk, vv] + kt[kk] * upd
                    # ⑤ 读出：照查询把写回之后的黑板读一遍
                    acc = 0.0
                    for kk in T.serial(dk):
                        acc = acc + qt[kk] * S[kk, vv]
                    O[hh, t, vv] = acc

    return gdn_impl


def gdn_asc_impl(Q, K, V, G, Beta, O, heads: int, dk: int, dv: int):
    """昇腾正文：状态矩阵常驻 UB（每块一头），token 串行、通道轴交 SimtVF。

    白话：同一本账换一台机器算。黑板摆在手边的共享台面上，淡掉黑板、写回、读出这些逐格活由
    台面上的一堆小工同时做；只有"沿键通道求和"这一步还得一个人从头扫到尾（向量化归约留待调优）。

    两处该后端约束（都不是算法差异）：① 状态 S 是 (dk, dv) 的 fp32，64x64 约 16 KB，必须确认
    在 UB 预算内（更宽的状态要按 dv 切块，本件先不切，云端 C1 报预算）；② 衰减系数先物化到
    一份 UB 向量，避免在并行区里重复算指数。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")，本地不实编）。
    """
    import tilelang.ascend.language as T

    seq = T.dynamic("seq")

    @T.prim_func
    def gdn_impl(Q: T.Tensor((heads, seq, dk), "float32"), K: T.Tensor((heads, seq, dk), "float32"),
                 V: T.Tensor((heads, seq, dv), "float32"), G: T.Tensor((heads, seq, dk), "float32"),
                 Beta: T.Tensor((heads, seq), "float32"), O: T.Tensor((heads, seq, dv), "float32")):
        """被追踪的那一层：状态常驻工人手边的小格子，只有通道轴可以铺开并行。

        白话：昇腾件这一份，黑板按头各留一份、整个递推期间不落回大表格；每条记录仍是那五步，
        只是五步里沿通道的那一维可以同时摊给一堆小工算。先后的次序不能改，所以外层一定串行。
        
        """
        with T.Kernel(heads) as hh:
            S = T.alloc_shared((dk, dv), "float32")
            kt_ub = T.alloc_shared((dk,), "float32")
            qt_ub = T.alloc_shared((dk,), "float32")
            vt_ub = T.alloc_shared((dv,), "float32")
            gt_ub = T.alloc_shared((dk,), "float32")
            dec_ub = T.alloc_shared((dk,), "float32")
            pred_ub = T.alloc_shared((dv,), "float32")
            upd_ub = T.alloc_shared((dv,), "float32")
            acc = T.alloc_var("float32")
            T.annotate_buffer_versions({S: 1})
            T.clear(S)
            for t in T.serial(seq):
                T.copy(K[hh, t, 0], kt_ub)
                T.copy(Q[hh, t, 0], qt_ub)
                T.copy(V[hh, t, 0], vt_ub)
                T.copy(G[hh, t, 0], gt_ub)
                with T.SimtVF(threads=VEC_THREADS):
                    for kk in T.Parallel(dk):
                        dec_ub[kk] = T.exp(gt_ub[kk])
                    for kk, vv in T.Parallel(dk, dv):
                        S[kk, vv] = S[kk, vv] * dec_ub[kk]
                # ② 召回：沿键通道的归约先按串行写（向量化留待云端 C2）
                bt = Beta[hh, t]
                for vv in T.serial(dv):
                    acc = 0.0
                    for kk in T.serial(dk):
                        acc = acc + S[kk, vv] * kt_ub[kk]
                    pred_ub[vv] = acc
                with T.SimtVF(threads=VEC_THREADS):
                    # ③ delta 修正：真值减预期，乘这一步的下笔力度
                    for vv in T.Parallel(dv):
                        upd_ub[vv] = (vt_ub[vv] - pred_ub[vv]) * bt
                    for kk, vv in T.Parallel(dk, dv):
                        # ④ 秩一写回
                        S[kk, vv] = S[kk, vv] + kt_ub[kk] * upd_ub[vv]
                # ⑤ 读出：写回之后再沿键通道扫一遍
                for vv in T.serial(dv):
                    acc = 0.0
                    for kk in T.serial(dk):
                        acc = acc + qt_ub[kk] * S[kk, vv]
                    O[hh, t, vv] = acc

    return gdn_impl


def plan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
         beta: torch.Tensor, o: torch.Tensor, target: str | None = None) -> dict[str, Any] | None:
    """判据：四路输入同头同长、键/值宽度是编译期常量、全程 fp32 且连续；不过关回 None。

    硬门槛：q/k/g 的最后一维（键宽 dk）与 v/o 的最后一维（值宽 dv）可以不同，但都必须是常量；
    三路头数与长度必须一致；beta 是 (heads, seq) 的 fp32。任一不合就回 None，由入口走普通写法。

    白话：先核对每个人的三摞纸是不是同样高、宽度对得上、记法是不是细尺子，不合模具就换手工。
    """
    name = ascend_env.normalize_target(target)
    if q.dim() != 3 or q.shape != k.shape or q.shape != g.shape:
        return None
    if v.dim() != 3 or int(v.size(1)) != int(q.size(1)) or int(v.size(0)) != int(q.size(0)):
        return None
    if o.shape != v.shape:
        return None
    if beta.shape != q.shape[:2]:
        return None
    for t in (q, k, v, g, beta, o):
        if t.dtype != torch.float32 or not t.is_contiguous():
            return None
    heads, seq, dk = (int(x) for x in q.shape)
    dv = int(v.size(2))
    if seq == 0 or dk == 0 or dv == 0:
        return None
    kwargs = {"heads": heads, "dk": dk, "dv": dv}
    impl = gdn_asc_impl if name == ascend_env.TARGET_ASCEND else gdn_cpu_impl
    key = f"gdn[{name}]|h{heads}k{dk}v{dv}"
    return {"target": name, "key": key, "kwargs": kwargs, "impl": impl, "seq": seq}


def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
        beta: torch.Tensor, o: torch.Tensor, spec: dict[str, Any]) -> bool:
    """取（或首编）内核并真跑一次递推；False 表示模具没开成，由入口层回退（不许假装算过）。

    缓存键不含 seq：递推长度是动态符号，同一份产物服务任意长度（与其余件同一口径）。

    白话：模具只在第一次开，之后这条链子有多长都照同一个走，不必因为换了批长度
    就重新开一次模。
    """
    kernel = ascend_env.get_compiled(
        spec["key"],
        lambda: tilelang.compile(spec["impl"](q, k, v, g, beta, o, **spec["kwargs"]),
                                 out_idx=[], **ascend_env.compile_kwargs(spec["target"])),
        spec["target"])
    if kernel is None:
        return False
    kernel(q, k, v, g, beta, o)
    return True


def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
            beta: torch.Tensor, out_dtype: torch.dtype = torch.float16,
            target: str | None = None) -> torch.Tensor:
    """前向递推：交回 (..., heads, seq, dv) 的输出；内核不可用时用同口径的普通写法。

    白话：按时间一条条把"淡掉—召回—补写—读出"走完，交回每一步读出来的结果。

    :raises ValueError: 头数/长度不一致或键宽值宽对不上时抛出。
    """
    _check(q, k, v, g, beta)
    lead = q.shape[:-2]
    seq, dk_, dv = int(q.size(-2)), int(q.size(-1)), int(v.size(-1))
    # 有效头轴 = 前面所有轴的乘积（B,H 与单批 H 都按同一条路走），形状还原时再拆回去
    q3, k3, v3, g3 = (t.reshape(-1, seq, s).contiguous() for t, s in ((q, dk_), (k, dk_), (v, dv), (g, dk_)))
    b3 = beta.reshape(-1, seq).contiguous()
    name = ascend_env.normalize_target(target)
    if ascend_env.active_backend(name, q.device) == ascend_env.TILELANG:
        a3, b3f, c3, d3 = q3.float(), k3.float(), v3.float(), g3.float()
        bb, out = b3.float(), torch.empty((q3.size(0), seq, dv), dtype=torch.float32, device=q.device)
        spec = plan(a3, b3f, c3, d3, bb, out, name)
        if spec is not None and run(a3, b3f, c3, d3, bb, out, spec):
            return out.to(out_dtype).reshape(*lead, seq, dv)
    acc = _eager(q3.float(), k3.float(), v3.float(), g3.float(), b3.float())
    return acc.to(out_dtype).reshape(*lead, seq, dv)


def backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
             beta: torch.Tensor, dout: torch.Tensor,
             target: str | None = None) -> tuple[torch.Tensor, ...]:
    """反向（**partial**）：本波不内核化，交回 torch 自动微分对同一条前向递推的梯度。

    白话：这一步还是手工算——把前向那本账交给通用引擎倒着走一遍；口径与前向完全一致，
    但速度不是内核速度，边界已经写进 `BWD_STATUS` 与 blocker 表，云端拿到卡再补。

    :returns: (dq, dk, dv, dg, dbeta) 五个与输入同形的 fp32 梯度。
    """
    ascend_env.record_blocker(
        f"gdn_backward[{ascend_env.normalize_target(target)}]",
        "本波未内核化：昇腾方言无 Gated DeltaRule 反向先例（见 gdn_asc 模块 docstring）")
    leaves = [t.detach().clone().float().requires_grad_(True) for t in (q, k, v, g, beta)]
    out = _eager(*leaves)
    # 注意口径：torch.autograd.grad 只把梯度**返回**给调用方，不会写回 leaf.grad，
    # 所以这里必须用返回值，不能照 backward() 的习惯去读 t.grad（读到的永远是 None）。
    grads = torch.autograd.grad(out, leaves, dout.float(), retain_graph=False)
    return tuple(t.contiguous() for t in grads)


def _check(q, k, v, g, beta) -> None:
    """入口契约：q/k/g 同形、v 与它们同头同长、beta 是 (..., heads, seq)、四者同设备。"""
    if q.dim() < 3:
        raise ValueError(f"q 至少三维 (..., heads, seq, 键宽)，实得 {q.dim()} 维")
    if q.shape != k.shape or q.shape != g.shape:
        raise ValueError(f"q/k/g 需同形，实得 {tuple(q.shape)} / {tuple(k.shape)} / {tuple(g.shape)}")
    if v.dim() < 3 or v.shape[:-1] != q.shape[:-1]:
        raise ValueError(f"v 需与 q 同头同长（只有值宽可不同），实得 {tuple(v.shape)}")
    if beta.shape != q.shape[:-1]:
        raise ValueError(f"beta 需为 {tuple(q.shape[:-1])}，实得 {tuple(beta.shape)}")
    if q.device != v.device or q.device != beta.device:
        raise ValueError("四路输入需在同一设备上")


def _eager(q3: torch.Tensor, k3: torch.Tensor, v3: torch.Tensor, g3: torch.Tensor,
           beta3: torch.Tensor) -> torch.Tensor:
    """回退路径与尺子：把五步递推照写一遍（fp32），与内核逐位同口径。

    白话：没有模具时按同一本账手工走一遍——淡掉、召回、补写、读出，一步都不省略。
    """
    heads, seq, dk = q3.shape
    dv = v3.size(-1)
    out = torch.zeros_like(v3)
    state = torch.zeros(heads, dk, dv, dtype=torch.float32, device=q3.device)
    for t in range(seq):
        state = state * g3[:, t].exp().unsqueeze(-1)      # ① 衰减（每键通道）
        pred = torch.einsum("hcd,hc->hd", state, k3[:, t])  # ② 召回：沿键通道把黑板读一遍
        upd = (v3[:, t] - pred) * beta3[:, t].unsqueeze(-1)   # ③ delta 修正
        state = state + k3[:, t].unsqueeze(-1) * upd.unsqueeze(1)  # ④ 秩一写回
        out[:, t] = torch.einsum("hcd,hc->hd", state, q3[:, t])  # ⑤ 读出：照查询再读一遍
    return out
