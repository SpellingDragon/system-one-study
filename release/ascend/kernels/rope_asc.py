"""rope 方言件：对打包好的 QKV 就地旋转前两个槽位（查询/键），反向用"角取负"复用同一条路。

【做什么】入口 `forward(qkv, cos, sin, rotate_slots=DEFAULT_SLOTS)`：qkv 是 (tokens,3,heads,dim)
的打包表，只转槽位 0 与 1，槽位 2（值）碰都不碰；算完原地放回，返回同一个对象（与 P1 的
`rope_kernel.forward` 同名同签名）。反向 `backward(gqkv, cos, sin)`：把同一套旋转再走一遍，
但把横向抄（sin）取负——这就是逆旋转，不需要第二个内核。
【怎么做】① 旋转口径是"前后半"（split-half）：前半减后半乘 sin、后半加前半乘 sin；
   ② token 数是动态符号，但网格上界必须是编译期常量（CPU(c) 后端实测：把动态维写进网格就一次
   都不发射），所以两份正文都用"常量网格 + 动态串行循环"：CPU 那份一个块串行扫全部 token，
   昇腾那份 NUM_BLOCKS 个向量核块按跨步领行；③ 昇腾侧是**纯 AIV 标量面**（P1-4 合流后的 910B
   可编形态）：`T.Kernel(NUM_BLOCKS)` + `T.Vector()` + 纯 `T.serial`，一格一格地从全局直接读、
   算、写回，零整块搬运、零 `alloc_shared`、零 UB 参与；角度一律从入口带进来的 cos/sin 表里
   直读，绝不在核内自算三角函数（910B 标量面根本没有 sinf/cosf 这两个件，见 attempts/E G-E1），
   也绝不引入新的中间全局缓冲（就地语义）。
【为什么】反向复用前向件是数学决定的：2x2 旋转矩阵的逆就是把角度取负，另开一模只会多一份要
同步的代码。被否方案一：为反向单独写"转置旋转"内核——两份额外的方言代码换不到任何收益；
被否方案二：把 qkv 复制一份再转（非就地）——P1 的调用方（autograd 的 _RopeFn）依赖就地语义
拿回同一个对象，改了会让上游的保存/引用全部错位；连续性判据（plan 里必须 contiguous）就是为
这条语义守门的：一旦对方递来的是 contiguous() 现场复制的临时块，转完原对象纹丝不动，"就地"
会静默失效。被否方案三（P1-4 合流时新增）：昇腾侧继续沿用 950 代际那套"整行搬进 UB → 线程级
并行转 → 整块搬出"的 SIMT 向量载体——该代际没有 SIMT 硬件模型，这一族原语在 patched 910B
tilelang 下实测编不出（D-int1），标量面逐格读写才是这台机器真正可用的件；它还顺带少了一件事：
"先把 x1/x2 各抄进手心再写回"的顺序由标量赋值天然保证，不再依赖向量件里"同一趟前后半互不
污染"这种隐性纪律。
"""
import tilelang

from typing import Any

import torch

from ascend.kernels import ascend_env

#: 昇腾侧内核位宽白名单（P1-1m）：qkv 与角度表都只收 fp32/fp16；**bf16 显式不收**（bisheng
#: dav-2201 vector 面缺 bf16 标量 cast，取证 attempts/B/b_addln_910b.py:23-24 + dump_bf16.py）。
#: 早先 _plan 收 bf16/fp16 而正文声明 fp32 ⇒ 编译按声明位宽出核、非 fp32 张量到发射期才被当 fp32
#: 解读（静默错数据，取证 reconcile/I_RESULT.md §⑥-3）；现在位宽在 plan 期就拒或按 dtype 参数化。
_ASC_DTYPES = (torch.float32, torch.float16)
#: torch dtype → DSL 位宽名
_TL_DT = {torch.float32: "float32", torch.float16: "float16"}

#: 打包布局里被旋转的槽位：0=查询、1=键；2=值 永远不动（与 P1 rope_mps.ROTATE_SLOTS 同值）
ROTATE_SLOTS = (0, 1)
#: 昇腾侧一次发射占用的向量核块数。910B 标量面件没有"每块多少线程、流水几级"这两维，
#: 原来的每块线程数与流水深度两个常量随 SIMT 载体一起撤掉，别留成假的配置项。
NUM_BLOCKS = 8


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


def rope_asc_impl(QKV, Cos, Sin, heads: int, dim: int, slots: int, sign: int,
                  qdt: str = "float32", adt: str = "float32"):
    """昇腾方言正文（910B / dav-2201 可编形态）：纯 AIV 标量面，按跨步领 token 行、就地转。

    白话：八个工人各领每隔八个的一本记录；翻到某一页某一头，先把前半后半各抄进手心（x1、x2），
    再照两张小抄算出新值原地放回；第三摞连碰都不碰。sign 为 -1 就是"横向抄倒着念"。

    四条 910B 方言纪律（P1-4 合流，凭据 attempts/E/e_rope_910b.py + RESULT.md + G-E1）：
      * 载体只有 `T.Kernel` + `T.Vector()` + 纯 `T.serial`；不碰 950 那族线程级 SIMT 并行载体，
        也不用整块搬运与流水循环——该代际无 SIMT 硬件模型，那些件实测编不出（D-int1）。
      * 角度**只从 cos/sin 表标量直读**：910B 标量面缺 `sinf/cosf`（G-E1），而表加载本就是
        本接口的既有契约（入口就收两张表），所以 rope 本体零 transcendental、零新增 compat。
      * 零 `alloc_shared`（G-C8：单死缓冲会落成命名空句柄），暂存全走 `T.alloc_var`。
      * 就地安全由标量顺序天然保证：两次读（x1、x2）都在两次写之前，不存在"写前半污染后半"
        的自噬面——这也是 attempts/E 那份"出地（另开一张 O 表）"件能收回就地口径的原因。

    领行方式：token 轴是动态符号，每核从 `bx` 起按 `NUM_BLOCKS` 跨步走，行程数写成
    `ceildiv(tokens - bx, NUM_BLOCKS)`；tokens>=1 时被除数恒正（最小 1-8+7=0），尾行天然落在
    界外，不需要再补 `if row < tokens` 守卫（发射体见 `((tokens + 7) - block_idx) >> 3`）。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")，P1-4 起本地实编判 .o）。
    """
    import tilelang.ascend.language as T

    tokens = T.dynamic("tokens")
    half = dim // 2

    # 位宽参数化（P1-1m）：qkv 与两张角度表各自按实际位宽声明（生产就是 qkv fp16 + 表 fp32 的
    # 混宽），**计算量仍恒 fp32**（alloc_var 不动，只在标量 load/store 处 cast）——与 `_eager`
    # “fp32 累加、按 qkv 位宽交货”同口径。三个 helper 在位宽为 float32 时都是恒等函数 ⇒
    # fp32 轨的生成码与参数化之前逐字相同（零回归）。
    def _lq(e):            # 标量读 qkv：qdt → fp32
        return e if qdt == "float32" else T.cast(e, "float32")

    def _la(e):            # 标量读角度表：adt → fp32
        return e if adt == "float32" else T.cast(e, "float32")

    def _sq(e):            # 标量写 qkv：fp32 结算 → qdt（就地语义、不新增全局缓冲）
        return e if qdt == "float32" else T.cast(e, qdt)

    @T.prim_func
    def rope_impl(QKV: T.Tensor((tokens, 3, heads, dim), qdt),
                  Cos: T.Tensor((tokens, half), adt),
                  Sin: T.Tensor((tokens, half), adt)):
        """被追踪的那一层：头数、每头宽度、转几摞、正转还是反转都是常量，只有条数是活的。

        白话：昇腾件这一份，多个工人按跨步各领一段记录，同一条转法照着两张小抄做——前半减去
        后半乘纵向抄，后半加上前半乘横向抄，算完原地放回；第三摞连碰都不碰。
        """
        with T.Kernel(NUM_BLOCKS) as bx:
            x1 = T.alloc_var("float32", T.float32(0.0))
            x2 = T.alloc_var("float32", T.float32(0.0))
            c = T.alloc_var("float32", T.float32(0.0))
            sn = T.alloc_var("float32", T.float32(0.0))
            with T.Vector():
                for r in T.serial(T.ceildiv(tokens - bx, NUM_BLOCKS)):
                    t = bx + r * NUM_BLOCKS
                    for s in T.serial(slots):
                        for hh in T.serial(heads):
                            for j in T.serial(half):
                                x1 = _lq(QKV[t, s, hh, j])
                                x2 = _lq(QKV[t, s, hh, j + half])
                                c = _la(Cos[t, j])
                                sn = _la(Sin[t, j])
                                if sign < 0:
                                    sn = T.float32(0.0) - sn
                                QKV[t, s, hh, j] = _sq(x1 * c - x2 * sn)
                                QKV[t, s, hh, j + half] = _sq(x2 * c + x1 * sn)

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
    else:
        # 昇腾面位宽判据（P1-1m，单点裁决）：qkv 与两张表都只收 fp32/fp16；两张表必须同宽
        # （sin 也要比——早先只比 cos，sin 换宽就是静默错数据）；qkv 与表**允许不同宽**：生产的
        # cos/sin 出自 train_step._build_params 的 torch.cos(angle) ⇒ fp32，而主干 qkv 是 fp16，
        # 早先那条 `qkv.dtype != cos.dtype` 会把生产形状直接判死、内核永不被调用。
        if (qkv.dtype not in _ASC_DTYPES or cos.dtype not in _ASC_DTYPES
                or sin.dtype not in _ASC_DTYPES or cos.dtype != sin.dtype):
            return None
    kwargs = {"heads": heads, "dim": dim, "slots": slots, "sign": sign}
    impl = rope_asc_impl if name == ascend_env.TARGET_ASCEND else rope_cpu_impl
    qdt, adt = _TL_DT.get(qkv.dtype, "float32"), _TL_DT.get(cos.dtype, "float32")
    if name == ascend_env.TARGET_ASCEND:
        kwargs.update(qdt=qdt, adt=adt)
    # 键尾位宽段：同 (h,dim,slots,sign) 换位宽必得新模具（防陈旧产物错位读内存，见 gemm_asc 同注）
    key = f"rope[{name}]|h{heads}d{dim}s{slots}g{sign}q{qdt}a{adt}"
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
    elif (name == ascend_env.TARGET_ASCEND
          and ascend_env.active_backend(name, qkv.device) == ascend_env.TILELANG):
        # P1-4 合流（路由修复）：target=ascend 现走内核；此前 forward 无 ascend 分支，
        # 卡上会永落 _eager（内核白编）。P1-1m（dtype 矩阵）：位宽判据从这里的“硬编码 fp32 三门”
        # 下放到 `_plan` **单点裁决**——正文已按 qdt/adt 参数化（fp32 累加、按实际位宽交货），
        # fp16 主干＋fp32 角度表就是生产形状；白名单外的位宽（bf16 等）由 _plan 回 None，照旧落
        # _eager，不会出现“按声明位宽出核、把 fp16 内存当 fp32 读”的静默错数据。
        spec = _plan(qkv, cos, sin, slots, 1, name)
        if spec is not None and _run(qkv, cos, sin, spec):
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
