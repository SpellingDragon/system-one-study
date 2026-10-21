"""rope 方言件：对打包好的 QKV 就地旋转前两个槽位（查询/键），反向用"角取负"复用同一条路。

【做什么】入口 `forward(qkv, cos, sin, rotate_slots=DEFAULT_SLOTS)`：qkv 是 (tokens,3,heads,dim)
的打包表，只转槽位 0 与 1，槽位 2（值）碰都不碰；算完原地放回，返回同一个对象（与 P1 的
`rope_kernel.forward` 同名同签名）。反向 `backward(gqkv, cos, sin)`：把同一套旋转再走一遍，
但把横向抄（sin）取负——这就是逆旋转，不需要第二个内核。
【怎么做】① 旋转口径是"前后半"（split-half）：前半减后半乘 sin、后半加前半乘 sin；
   ② token 数是动态符号，但网格上界必须是编译期常量（CPU(c) 后端实测：把动态维写进网格就一次
   都不发射），所以两份正文都用"常量网格 + 动态串行/流水循环"；CPU 那份每块串行扫一段 token，
   昇腾那份固定 NUM_BLOCKS 个向量核块、token 段落在 T.Pipelined 上；③ 昇腾侧先把一行的
   q/k 两段搬进 UB，在 SimtVF 里用 fp32 算、按原 dtype 逐元素写回 UB，再整块搬出——刻意不用
   T.copy 跨位宽搬（该后端 DMA 不许顺带转类型），也刻意不引入新的中间全局缓冲（就地语义）。
【为什么】反向复用前向件是数学决定的：2x2 旋转矩阵的逆就是把角度取负，另开一模只会多一份要
同步的代码。被否方案一：为反向单独写"转置旋转"内核——两份额外的方言代码换不到任何收益；
被否方案二：把 qkv 复制一份再转（非就地）——P1 的调用方（autograd 的 _RopeFn）依赖就地语义
拿回同一个对象，改了会让上游的保存/引用全部错位；连续性判据（plan 里必须 contiguous）就是为
这条语义守门的：一旦对方递来的是 contiguous() 现场复制的临时块，转完原对象纹丝不动，"就地"
会静默失效。
"""
import tilelang

from typing import Any

import torch

from ascend.kernels import ascend_env

#: 打包布局里被旋转的槽位：0=查询、1=键；2=值 永远不动（与 P1 rope_mps.ROTATE_SLOTS 同值）
ROTATE_SLOTS = (0, 1)
#: 昇腾侧一次发射占用的向量核块数与流水深度
NUM_BLOCKS = 8
VEC_THREADS = 64
NUM_STAGES = 2


def rope_cpu_impl(QKV, Cos, Sin, heads: int, dim: int, slots: int, sign: int):
    """CPU(c) 方言正文：token 数动态、逐 token 逐头串行转；就地改写同一块内存。

    白话：一格材料交给一个人看，他把前两摞按小抄上的角度转一转——前半减去后半乘纵向抄，
    后半加上前半乘横向抄，算完原地放回；第三摞连碰都不碰。sign 为 -1 时就是倒着转。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile）。
    """
    import tilelang.cpu.language as T

    tokens = T.dynamic("tokens")
    half = dim // 2

    @T.prim_func
    def rope_impl(QKV: T.Tensor((tokens, 3, heads, dim), "float32"),
                  Cos: T.Tensor((tokens, half), "float32"),
                  Sin: T.Tensor((tokens, half), "float32")):
        """被追踪的那一层：形状在这里落定，token 轴是符号，其余三轴是编译期常量。

        白话：模具在这一层定型——多少个头、每头多宽、转几摞、正转还是反转都写死，
        只有"这批有多少条"留成活的，运行时多长都按同一块模具走。
        """
        with T.Kernel(1) as bx:
            for t in T.serial(tokens):
                for s in T.serial(slots):
                    for hh in T.serial(heads):
                        for j in T.serial(half):
                            x1 = QKV[t, s, hh, j]
                            x2 = QKV[t, s, hh, j + half]
                            c = Cos[t, j]
                            sn = sign * Sin[t, j]
                            QKV[t, s, hh, j] = x1 * c - x2 * sn
                            QKV[t, s, hh, j + half] = x2 * c + x1 * sn

    return rope_impl


def rope_asc_impl(QKV, Cos, Sin, heads: int, dim: int, slots: int, sign: int):
    """昇腾方言正文：UB 放一个 token 的两段（q、k），SimtVF 上做 fp32 旋转再原地搬出。

    白话：一次只把一小片材料搬到手边，按抄表转一转，转完立刻放回原来的位置；写回用的是
    逐元素赋值而不是整块搬运，因为这台机器不允许搬运的时候顺手改记法。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")，本地不实编）。
    """
    import tilelang.ascend.language as T

    tokens = T.dynamic("tokens")
    half = dim // 2
    width = slots * heads * dim  # 一次搬进 UB 的元素数（q、k 两段）

    @T.prim_func
    def rope_impl(QKV: T.Tensor((tokens, 3, heads, dim), "float32"),
                  Cos: T.Tensor((tokens, half), "float32"),
                  Sin: T.Tensor((tokens, half), "float32")):
        """被追踪的那一层：头数、每头宽度、转几摞、正转还是反转都是常量，只有条数是活的。

        白话：昇腾件这一份，多个工人各领一段记录，同一条转法照着两张小抄做——前半减去后半乘
        纵向抄，后半加上前半乘横向抄，算完原地放回；第三摞连碰都不碰。
        
        """
        with T.Kernel(NUM_BLOCKS) as bx:
            ub = T.alloc_shared((width,), "float32")
            c_ub = T.alloc_shared((half,), "float32")
            s_ub = T.alloc_shared((half,), "float32")
            T.annotate_buffer_versions({ub: NUM_STAGES})
            for t in T.Pipelined(T.ceildiv(tokens, NUM_BLOCKS), num_stages=NUM_STAGES):
                row = t * NUM_BLOCKS + bx
                T.copy(QKV[row, 0, 0, 0], ub)
                T.copy(Cos[row, 0], c_ub)
                T.copy(Sin[row, 0], s_ub)
                with T.SimtVF(threads=VEC_THREADS):
                    for s, hh, j in T.Parallel(slots, heads, half):
                        base = (s * heads + hh) * dim
                        x1 = ub[base + j]
                        x2 = ub[base + j + half]
                        sn = sign * s_ub[j]
                        ub[base + j] = x1 * c_ub[j] - x2 * sn
                        ub[base + j + half] = x2 * c_ub[j] + x1 * sn
                T.copy(ub, QKV[row, 0, 0, 0])

    return rope_impl


def _plan(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, slots: int,
          sign: int, target: str) -> dict[str, Any] | None:
    """形状/位宽/连续性/就地安全性判据；不过关返回 None 让入口走回退（不硬凑）。

    白话：先核对盒子是不是"三摞、每摞一行格、格子连着摆"的标准形状，抄表也没换成省格子的记法；
    形状对、位数对、东西确实摊在一整张桌上，才值得动用模具。
    """
    name = ascend_env.normalize_target(target)
    if qkv.dim() != 4 or qkv.size(1) != 3:
        return None
    if not (qkv.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()):
        return None
    tokens, _, heads, dim = (int(v) for v in qkv.shape)
    if dim % 2 or slots not in (1, 2):
        return None
    if cos.shape != (tokens, dim // 2) or sin.shape != (tokens, dim // 2):
        return None
    if name == ascend_env.TARGET_CPU:
        if qkv.dtype != torch.float32 or cos.dtype != torch.float32 or sin.dtype != torch.float32:
            return None
    elif qkv.dtype not in (torch.float32, torch.bfloat16, torch.float16) or qkv.dtype != cos.dtype:
        return None
    kwargs = {"heads": heads, "dim": dim, "slots": slots, "sign": sign}
    impl = rope_asc_impl if name == ascend_env.TARGET_ASCEND else rope_cpu_impl
    key = f"rope[{name}]|h{heads}d{dim}s{slots}g{sign}"
    return {"target": name, "key": key, "kwargs": kwargs, "impl": impl}


def _run(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, spec: dict[str, Any]) -> bool:
    """取（或首编）内核并就地改写 qkv；False 表示方言此刻不可用，入口需要回退。

    缓存键不含 token 数：换长度不换产物（实测同一产物连跑 8/33 个 token 都精确）。

    白话：把盒子原样递进模具、原地转好了拿回来；模具开不了就说一声，让外面手工转。
    """
    kernel = ascend_env.get_compiled(
        spec["key"],
        lambda: tilelang.compile(spec["impl"](qkv, cos, sin, **spec["kwargs"]),
                                 out_idx=[], **ascend_env.compile_kwargs(spec["target"])),
        spec["target"])
    if kernel is None:
        return False
    kernel(qkv, cos, sin)
    return True


def forward(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
            rotate_slots: tuple[int, ...] = ROTATE_SLOTS, target: str | None = None) -> torch.Tensor:
    """就地旋转 qkv 的指定槽位并返回同一个对象；内核不可用时用 fp32 的普通写法就地算。

    白话：把前两摞按角度转一转、原地放回去，第三摞不动；模具不合适就手工转，转的口径一模一样。

    :raises ValueError: qkv 不是 (tokens,3,heads,dim)、或 dim 是奇数时抛出。
    """
    if qkv.dim() != 4 or qkv.size(1) != 3:
        raise ValueError(f"qkv 需 (tokens,3,heads,dim)，实得 {tuple(qkv.shape)}")
    if qkv.size(3) % 2:
        raise ValueError(f"每头宽度需为偶数（要折半旋转），实得 {qkv.size(3)}")
    name = ascend_env.normalize_target(target)
    slots = len(rotate_slots)
    if name == ascend_env.TARGET_CPU:
        q32 = qkv.to(torch.float32)
        spec = _plan(q32, cos, sin, slots, 1, name) if ascend_env.active_backend(
            name, qkv.device) == ascend_env.TILELANG else None
        if spec is not None and _run(q32, cos, sin, spec):
            if q32.data_ptr() != qkv.data_ptr():
                qkv.copy_(q32.to(qkv.dtype))
            return qkv
    return _eager(qkv, cos, sin, rotate_slots, 1)


def backward(gqkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
             rotate_slots: tuple[int, ...] = ROTATE_SLOTS,
             target: str | None = None) -> torch.Tensor:
    """反向：把 sin 取负再走一遍同一条旋转路（旋转矩阵的逆就是角度取负）。

    白话：正着转是"照抄表转一格"，倒着转就是"把带方向的那半张抄表反着念"，模具是同一个。
    """
    name = ascend_env.normalize_target(target)
    slots = len(rotate_slots)
    if name == ascend_env.TARGET_CPU:
        g32 = gqkv.to(torch.float32)
        spec = _plan(g32, cos, sin, slots, -1, name) if ascend_env.active_backend(
            name, gqkv.device) == ascend_env.TILELANG else None
        if spec is not None and _run(g32, cos, sin, spec):
            if g32.data_ptr() != gqkv.data_ptr():
                gqkv.copy_(g32.to(gqkv.dtype))
            return gqkv
    return _eager(gqkv, cos, sin, rotate_slots, -1)


def _eager(qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
           rotate_slots: tuple[int, ...], sign: int) -> torch.Tensor:
    """回退路径：fp32 累加、就地写回原对象（与内核同口径，供对拍当尺子）。

    白话：没有合适模具时按同一本账手工转一遍，转完照样放回原盒子，交出去的还是那个对象。
    """
    half = qkv.size(3) // 2
    c = cos.float().unsqueeze(1)
    s = (sign * sin).float().unsqueeze(1)
    for slot in rotate_slots:
        # x1/x2 必须先各留一份原件：qkv 已是 fp32 时 .float() 返回的是视图，
        # 直接写前半会把 x1 改掉，后半那一行就会读到被污染的值（就地语义下的经典自噬）。
        x1 = qkv[:, slot, :, :half].float().clone()
        x2 = qkv[:, slot, :, half:].float().clone()
        qkv[:, slot, :, :half] = (x1 * c - x2 * s).to(qkv.dtype)
        qkv[:, slot, :, half:] = (x2 * c + x1 * s).to(qkv.dtype)
    return qkv
