"""ascend 九件方言内核的**最小混合栈一步冒烟件**（甲路 P2-prep）。

【做什么】把 `ascend.autograd_asc` 的八个可微门面按训练时真正的拓扑串一遍——
`linear → add_ln → rope → attn_sw → linear → gdn_delta → lora_apply → readout`——
前向出 loss、`loss.backward()`、Adam 一步更新，然后打印判决行
`P2-STEP-<target> loss0=<x> loss1=<y>`：本波只验**接线正确性**（loss 是否下降、
是否每个可训参数都拿到非 None 的梯度），**不验内核数值 rel/V***。

【怎么做】① 参数量任意小（d=32, seq=16, heads=2, dim=16），CPU 上毫秒级；② 所有
可微点走 `autograd_asc` 门面（不吃 raw torch ops），确保"如果哪件反向没接对，梯度就
是 None"这条硬判据真能触发；③ 后端由 `--target {cpu, ascend}` 或环境变量
`DMLAYA_ASCEND_TARGET` 决定，卡上只翻 device 与 `--target ascend`，代码不动；④ 用
Adam（`lr=1e-2`）——尺度不变的更新对**单步是否下降**最稳（SGD 在小图上容易因梯度
量级过大一步冲过头）；⑤ 判决前必做**全参非 None 检查**，任一 None 就直接 fail 输出
参数名，方便定位到具体门面。

【为什么】此前 `ascend/` 无任何 `autograd.Function`、`*_kernel` 无模型消费方——R11
消费者枚举缺口。本件就是把消费者接上：**host 上跑通即证明拓扑与反向公式对；卡窗只翻
`--target ascend` 就能真跑一步**。被否方案一：直接在 sys1/model.py 上换后端——把 P1
冻结件拽进 P2 波次，破坏"接口稳定 + 内核合流"的层次；被否方案二：只写单测不写可运行
脚本——上卡那天还得重搭一遍 main，等于把卡窗时间花在写脚本上。

【卡窗可信化（p2-13 代理 B）】在原有 `P2-STEP-<target> loss0=.. loss1=..` 之外再加两条判决：
`NPU-REG ok|fail|skip <version>`——H1：torch 内建清单里没有 npu，ascend 轨必须先
`import torch_npu` 把设备注册进来（cpu 轨**不得** import，无 npu 机器会当场 ImportError）；
`KERNEL-USED-PASS kernels=<n> blockers=<m> fallbacks=<k>`——P1-1l：loss 下降只证明**接线**对，
不证明**走了内核**（探针/发射一失败就回落 eager，train_step 照样打 PASS）。三证不齐即
`KERNEL-USED-FAIL`，ascend 轨非零退出（退码 5），把"跑完=成功"这条误读彻底堵死。

【产品位宽端到端（p2-13 代理 F2）】再加 `--dtype {fp32,fp16}`（默认 fp32，现行行为逐字不变）：
fp16 档把主干切成**生产位宽**——可训参数与激活 fp16、角度表保持 fp32（C2 §7 的"qkv fp16+表 fp32"
混宽形状）、loss 与归约恒钉 fp32 面、梯度出口按输入位宽交货。两条新增判决（**不动**任何原判决行
的格式与"末行"位置）：① `DTYPE-TRACE <dt> params=<n> fp16=<n> fp32_accum=<true|false> fallback=<n>`
——位宽账（fp16 档下 KERNEL-USED 三证照常生效，退码 2/3/4/5 语义归 B、原样保留）；② 归约纪律钉
——fp16 档运行期在 **Python 层**装 guard（`install_reduction_guard`）：fp16 张量不带显式 fp32 累加
dtype 直接 `sum/mean/var/std/prod/amax/amin`、或以 fp16 操作数走 `matmul/mm/bmm`，一律记违规；
违规>0 ⇒ `DTYPE-TRACE-FAIL` + 退码 `EXIT_DTYPE_DISCIPLINE=6`。这是 C2 在件级防过的"静默错数据"
风险面在训练步组合面的等价物；内核内部的归约面不经 Python，由 C2 逐件证据钉死（fp16 原生三件
UB 累加恒 fp32、六件入口 fp32 物化），guard 盯的是**组装面**（闭式漏写 `.float()`、新降位点等）。

不接的位点（本波边界，如实登记，不假绿）：
- `gdn_conv` 门面不在本件里跑——`gdn_kernel.conv_forward` 无 conv_backward 入口，
  本波 `_GdnConvFn` 的 fp32 闭式兜底已由 `test_ascend_train_step.py` 的 gradcheck
  单独覆盖；把它塞进训练图会引入一条与内核不同源的反向链，混淆"接线对不对"的判据。
- 数值精度域（rel、V*）由 `test_ascend_gradcheck.py` 覆盖，与本件解耦。
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import warnings

# 允许 `python ascend/train_step.py` 直接跑（用户交付指令里的调用形式）：
# 把 release/ 挂进 sys.path，让 import 语义与 `python -m ascend.train_step` / pytest 一致。
_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from ascend import autograd_asc as ops  # noqa: E402
from ascend.kernels import add_ln_kernel, ascend_env  # noqa: E402

# --------------------------------------------------------------------------- H1：npu 设备注册
#: torch 的内建设备清单里没有 `npu`——没注册时 `torch.device("npu")` 直接抛
#: `RuntimeError: Expected one of cpu, cuda, ... device string: npu`（凭据：
#: reconcile/A3_PREFLIGHT.md §3 与 §6，容器整链实测 [4] 腿连 P2-STEP 行都留不下）。
#: 注册它是 `torch_npu` 在 import 时的副作用；而 cpu 轨去 import 它又会在无 npu 的机器上
#: ImportError 崩掉整条链 ⇒ 只能"按 ascend 轨条件 import"，并把结果收成一行可 grep 的账。
NPU_TARGET_ALIASES = ("ascend", "asc", "npu")
#: 退出码：4=设备/注册不可用（H1），5=内核一次没走却自称通过（P1-1l），6=fp16 档归约纪律钉触发（F2）
EXIT_DEVICE_UNAVAILABLE = 4
EXIT_KERNEL_UNUSED = 5
#: 自证钩子：置 1 时连 cpu 轨也按同一口径计入门禁（负例自证用；卡窗 ascend 轨本来就在门下）
ENV_KERNEL_USAGE_STRICT = "SYS1_KERNEL_USAGE_STRICT"

#: F2：CLI 位宽档名 → torch.dtype。**没有 bf16 档**：bisheng vector 面 bf16 标量 cast 的缺件
#: 取证由 C2 §8 钉死（三件 plan 期显式拒），训练步入口同样不放开。
DTYPE_BY_NAME = {"fp32": torch.float32, "fp16": torch.float16}
#: F2：退码 6 = fp16 档出现"归约在 fp16 做"。编号续在 B 的 4/5 之后，不改它们的语义与判读。
EXIT_DTYPE_DISCIPLINE = 6

_npu_reg: dict | None = None


def npu_requested(argv: list[str] | None = None) -> bool:
    """这一趟是否该走 ascend 后端：命令行 `--target ascend`（asc/npu 一并认）或 env DMLAYA_ASCEND_TARGET。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, item in enumerate(argv):
        if item in ("-h", "--help"):
            return False
        if item.startswith("--target"):
            val = item.split("=", 1)[1] if "=" in item else (argv[i + 1] if i + 1 < len(argv) else "")
            return val.strip().lower() in NPU_TARGET_ALIASES
    raw = (os.environ.get(ascend_env.ENV_TARGET) or "").strip().lower()
    return raw in NPU_TARGET_ALIASES


def _npu_device_usable() -> bool:
    """torch 侧到底认不认识 npu 这个名字——唯一硬判据是"能造出这个 device 对象"。"""
    try:
        torch.device("npu")
        return True
    except Exception:  # noqa: BLE001  没注册就是没注册，不关心原话长什么样
        return False


def _npu_avail_text() -> str:
    """设备在场读数（字符串化，绝不因探测本身把脚本带崩）。

    **纪律（实测踩出来的）**：绝不用 `hasattr(torch, "npu")` 或 `torch.npu` 去"先看一眼"——
    torch 对后端模块做的是**惰性 import**，碰这个名字本身就会把 `torch_npu` 拉进来，失败时
    它还自己吞掉异常、只留一行 `An unexpected error occurred while checking NPU availability`
    （B_probe_h1_npu_reg.py 在 host 用"毒药 torch_npu"实测复现）。cpu 轨一次都不该惊动它，
    所以判据换成"设备名可不可造"（`_npu_device_usable()`，本函数只在 ascend 轨被调用）。
    """
    if not _npu_device_usable():
        return "False"
    try:
        return str(bool(torch.npu.is_available()))
    except Exception:  # noqa: BLE001
        return "probe-error"


def ensure_torch_npu() -> dict:
    """ascend 轨按需 `import torch_npu` 并复验注册结果；幂等（第二次直接复用上一笔结论）。

    返回 `{"status": ok|fail, "version": .., "detail": ..}`，由 `npu_reg_line` 收成判决行。
    刻意**不抛异常**：注册失败要由 main() 打成 `NPU-REG-FAIL` 并给退码，而不是把裸 traceback
    丢给卡窗日志——旧版的病灶正是"失败长得不像失败"。
    """
    global _npu_reg
    if _npu_reg is not None:
        return _npu_reg
    loaded = getattr(sys.modules.get("torch_npu"), "__version__", None)
    if _npu_device_usable():                      # 昇腾打过补丁的 torch 构建：import 前就注册好了
        _npu_reg = {"status": "ok", "version": loaded or "torch-builtin",
                    "detail": "npu 已在场（未额外 import）"}
        return _npu_reg
    try:
        import torch_npu  # noqa: F401  只用它注册设备类型，本件不碰它的任何算子
    except Exception as exc:  # noqa: BLE001
        _npu_reg = {"status": "fail", "version": "none",
                    "detail": f"import torch_npu -> {type(exc).__name__}: {exc}"}
        return _npu_reg
    if _npu_device_usable():
        _npu_reg = {"status": "ok", "version": getattr(torch_npu, "__version__", "unknown"),
                    "detail": "torch_npu 注册成功"}
    else:
        _npu_reg = {"status": "fail", "version": getattr(torch_npu, "__version__", "unknown"),
                    "detail": "import 成功但 torch.device('npu') 仍不可用"}
    return _npu_reg


def npu_reg_line(state: dict, target: str) -> str:
    """把注册状态收成一行判决（token 与 card_run.sh [0] 段探针一致：`NPU-REG ok|fail|skip <version>`）。

    skip（cpu 轨）时 `npu_avail=n/a` 而不是 False：这一趟压根没探测过，编一个 False 出来会把
    "没看"和"看了说没有"混成同一行——那是另一种假绿。
    """
    avail = "n/a" if state["status"] == "skip" else _npu_avail_text()
    return (f"NPU-REG {state['status']} {state['version']} target={target} "
            f"npu_avail={avail} {state['detail']}")


if npu_requested():          # 模块加载即注册：被 import 着用（而不只 __main__）也照样覆盖
    ensure_torch_npu()


# ------------------------------------------------------------------ F2：fp16 归约纪律钉
#: guard 的作用域 = **Python 层**。内核内部的归约面不经 Python，其位宽纪律已由 C2 逐件钉死
#: （reconcile/C2_dtype_matrix.md §2/§5：linear/dW/rope 三件 fp16 原生、UB 累加面恒 fp32，
#: 其余六件入口 fp32 物化）。本 guard 钉的是另一半风险面——**训练步组装面**上"归约被不小心留在
#: fp16 做"：闭式漏写 `.float()`、新加的归约点没升宽、拿 fp16 操作数直接做内积。判法可解释：
#: fp16 张量直接跑 _GUARDED_REDUCTIONS 且不带 dtype=fp32/fp64 关键字 ⇒ 违规；fp16 参与
#: _GUARDED_DOTS（自或对方任一为 fp16）⇒ 违规（生产纪律：内积必须先升 fp32 面，件级入口件都这么
#: 契约）。违规>0 ⇒ 判决行 `fp32_accum=false` 且 main 给退码 6；负例自证见
#: ascend/kernels/reconcile/F2_probe_dtype_guard.py。
_GUARDED_REDUCTIONS = ("sum", "mean", "var", "std", "prod", "amax", "amin")
_GUARDED_DOTS = ("matmul", "mm", "bmm")
_F2_ACC_OK = (torch.float32, torch.float64)   # fp16 归约允许显式升到的累加位宽

#: 纪律账本（violations=违规记录；fp32_acc=合规的 fp16→fp32 归约计数；installed=对称计数，
#: 嵌套安装只打一次补丁）。LAST_AUDIT 由 run() 在步末落账，main() 取去拼 DTYPE-TRACE。
_reduction_ledger: dict = {"installed": 0, "violations": [], "fp32_acc": 0}
_orig_methods: dict[str, object] = {}
LAST_AUDIT: dict = {}


def _f2_wrap(name: str, dot: bool = False) -> None:
    """包一层 torch.Tensor.<name>：只记账、不改行为（原实现原样执行，判决在步末统一给）。"""
    orig = getattr(torch.Tensor, name)

    if dot:
        def _wrapped(self, *args, **kwargs):
            other = args[0] if args else None
            if self.dtype == torch.float16 or (torch.is_tensor(other) and other.dtype == torch.float16):
                _reduction_ledger["violations"].append(
                    f"{name}(fp16 内积面: {tuple(self.shape)} × "
                    f"{tuple(other.shape) if torch.is_tensor(other) else other})")
            return orig(self, *args, **kwargs)
    else:
        def _wrapped(self, *args, **kwargs):
            if self.dtype == torch.float16:
                acc = kwargs.get("dtype", None)
                if acc in _F2_ACC_OK:
                    _reduction_ledger["fp32_acc"] += 1
                else:
                    _reduction_ledger["violations"].append(
                        f"{name}(fp16 直接归约: shape={tuple(self.shape)} dtype={acc})")
            return orig(self, *args, **kwargs)

    _wrapped.__name__ = name
    _orig_methods[name] = orig
    setattr(torch.Tensor, name, _wrapped)


def install_reduction_guard() -> int:
    """装 fp16 归约纪律钉（与 uninstall 对称；嵌套只有最外层真正打补丁）。返回当前计数。"""
    if _reduction_ledger["installed"] == 0:
        _reduction_ledger["violations"] = []
        _reduction_ledger["fp32_acc"] = 0
        for _n in _GUARDED_REDUCTIONS:
            _f2_wrap(_n)
        for _n in _GUARDED_DOTS:
            _f2_wrap(_n, dot=True)
        # `@` 运算符走类型槽 __matmul__，不经过 Tensor.matmul 方法包装（F2 实测），
        # 故把运算符面一并罩上：任一操作数为 fp16 即记违规（判据与 dot 面一致）。
        _f2_wrap("__matmul__", dot=True)
        _f2_wrap("__rmatmul__", dot=True)
    _reduction_ledger["installed"] += 1
    return int(_reduction_ledger["installed"])


def uninstall_reduction_guard() -> int:
    """卸载（计数归零才真还原原方法）。返回剩余计数。"""
    _reduction_ledger["installed"] = max(0, int(_reduction_ledger["installed"]) - 1)
    if _reduction_ledger["installed"] == 0:
        for _n, _orig in _orig_methods.items():
            setattr(torch.Tensor, _n, _orig)
        _orig_methods.clear()
    return int(_reduction_ledger["installed"])


def reduction_violations() -> list[str]:
    """当前违规清单快照。探针/负例自证入口：伪造的 fp16 归约必须出现在这里，否则纪律钉是装饰。"""
    return list(_reduction_ledger["violations"])


def dtype_trace_line(audit: dict, fallback_events: int = 0) -> str:
    """位宽判决行（纯函数，不读全局、可喂假账自证负例）：

    `DTYPE-TRACE <dt> params=<n> fp16=<n> fp32_accum=<true|false> fallback=<n>`
      - params/fp16：可训参数总数与**实际 fp16 个数**——一个数就暴露"位宽有没有真切过去"；
      - fp32_accum：归约面无违规。fp32 档平凡为 true（fp32 归约是基线，不是例外）；
      - fallback：与 B 的 KERNEL-USED 同口径的回落公告计数（fp16 档三证照常生效）。
    """
    acc_ok = not audit.get("violations")
    return (f"DTYPE-TRACE {audit.get('dtype', 'fp32')} params={audit.get('params', 0)} "
            f"fp16={audit.get('fp16', 0)} fp32_accum={'true' if acc_ok else 'false'} "
            f"fallback={int(fallback_events)}")


def _make_inputs(seq: int, d: int, device: torch.device, gen: torch.Generator,
                 dtype: torch.dtype = torch.float32):
    """造一批固定种子的输入与目标：seq 个 token、宽度 d；目标是随机投影（回归任务）。

    白话：数据不进优化器（不参与 requires_grad），只当"考卷"；标准答案也一次造好、
    训练全程不变——这样 loss 是否真的下降，只由参数与反向决定，没有噪声可推诿。
    """
    x0 = torch.randn(seq, d, device=device, generator=gen)
    target = torch.randn(seq, d, device=device, generator=gen)
    if dtype != torch.float32:
        x0 = x0.to(dtype)      # 主干输入按生产位宽；target 保持 fp32 考卷（loss 归约在 fp32 面做）
    return x0, target


def _build_params(seq: int, d: int, heads: int, dim: int, rope_theta: float,
                  device: torch.device, gen: torch.Generator,
                  dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """按拓扑所需准备所有 requires_grad 的原料（不含 cos/sin 与 ids，那三样是常量）。

    白话：`nn.Parameter` 那种"包一层类"的做法留给正式训练；本件只要一张扁平字典，
    优化器直接吃 list(d.values())，读代码时"哪一位是参数"一眼可数，不藏着掖着。
    """
    def _r(*shape, scale: float = 1.0):
        return (torch.randn(*shape, device=device, generator=gen) * scale).requires_grad_(True)

    p: dict[str, torch.Tensor] = {
        "W1": _r(d, d, scale=0.1), "b1": _r(d, scale=0.01),
        "ln_g": torch.ones(d, device=device).requires_grad_(True), "ln_b": _r(d, scale=0.01),
        "Wqkv": _r(3 * heads * dim, d, scale=0.1), "bqkv": _r(3 * heads * dim, scale=0.01),
        "Wout": _r(d, heads * dim, scale=0.1), "bout": _r(d, scale=0.01),
        # GDN 支路：从 proj 再投一次得到 q/k/v/g/beta，全走 linear 门面
        "Wgq": _r(heads * dim, d, scale=0.1), "Wgk": _r(heads * dim, d, scale=0.1),
        "Wgv": _r(heads * dim, d, scale=0.1), "Wgg": _r(heads * dim, d, scale=0.1),
        "Wgb": _r(heads, d, scale=0.1),
        "Wout_g": _r(d, heads * dim, scale=0.1),
        # LoRA 旁路
        "lora_a": _r(8, d, scale=0.05), "lora_b": _r(d, 8, scale=0.05),
    }
    if dtype != torch.float32:
        # F2 生产位宽：先按与 fp32 档**逐字同**的 RNG 造 fp32 值，再统一降位（fp16=同值取整，
        # 不是重采样），两档数字差才只反映位宽、不反映样本。cos/sin 角度表**保持 fp32**——
        # "qkv fp16 + 表 fp32" 就是 C2 §7 的生产混宽形状（键 `qfloat16afloat32`）。
        p = {k: v.detach().to(dtype).requires_grad_(True) for k, v in p.items()}
    cos = torch.ones(seq, dim // 2, device=device) * (rope_theta ** 0)   # 角度表本件不参与训练
    sin = torch.zeros(seq, dim // 2, device=device)
    # 真正的 RoPE 角度：位置 pos 与半维 j → θ·pos/(10000^(2j/dim))
    pos = torch.arange(seq, device=device).float().unsqueeze(1)
    j = torch.arange(dim // 2, device=device).float().unsqueeze(0)
    inv = rope_theta / (10000.0 ** (2.0 * j / dim))
    angle = pos * inv
    return p, torch.cos(angle), torch.sin(angle)


def _forward(x0: torch.Tensor, p: dict[str, torch.Tensor], cos: torch.Tensor,
             sin: torch.Tensor, seq: int, d: int, heads: int, dim: int,
             target: str, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """按拓扑把八个门面都过一遍，交回 loss（fp32、MSE）。

    白话：一层一层照着"乘加→残差+整形→打包 qkv→旋转→切三摞→注意力→投回 d→
    走一遍 GDN→再投回 d→合流→LoRA 旁路→读出"排下来；每一站都走门面、不走 raw
    torch，反向是否接对全看梯度能不能回来——空接会当场报错，不会悄悄通过。
    """
    # 1) linear：x0 (seq, d) → h_lin (seq, d)
    h_lin = ops.linear(x0, p["W1"], p["b1"], out_dtype=out_dtype, target=target)
    # 2) add_ln：把 x0 当残差合进来 → (y, h)
    y, _h = ops.add_ln(h_lin, x0, p["ln_g"], p["ln_b"], eps=add_ln_kernel.DEFAULT_EPS,
                       out_dtype=out_dtype, target=target)
    # 3) linear 到 packed-qkv：(seq, 3·heads·dim)
    qkv_p = ops.linear(y, p["Wqkv"], p["bqkv"], out_dtype=out_dtype, target=target)
    # 4) reshape 到 rope 契约形状 (tokens, 3, heads, dim)
    qkv = qkv_p.reshape(seq, 3, heads, dim)
    # 5) rope（内核就地改，门面内部 clone，训练原件安全）
    qkv = ops.rope(qkv, cos, sin, rotate_slots=rope_slots_default(), target=target)
    # 6) 拆三摞：(heads, seq, dim)
    q = qkv[:, 0].transpose(0, 1).contiguous()
    k = qkv[:, 1].transpose(0, 1).contiguous()
    v = qkv[:, 2].transpose(0, 1).contiguous()
    # 7) 因果滑窗注意力（window ≥ seq ⇒ 等价全因果）
    o = ops.attn_sw(q, k, v, window=seq, out_dtype=out_dtype, target=target)
    # 8) 展平 → linear → proj (seq, d)
    o_flat = o.transpose(0, 1).reshape(seq, heads * dim)
    proj = ops.linear(o_flat, p["Wout"], p["bout"], out_dtype=out_dtype, target=target)
    # 9) GDN 递推支路：从 proj 再投一次得 q/k/v/g/beta，跑 gdn_delta，再 linear 回 d
    qg = ops.linear(proj, p["Wgq"], None, out_dtype=out_dtype, target=target) \
        .reshape(seq, heads, dim).transpose(0, 1).contiguous()
    kg = ops.linear(proj, p["Wgk"], None, out_dtype=out_dtype, target=target) \
        .reshape(seq, heads, dim).transpose(0, 1).contiguous()
    vg = ops.linear(proj, p["Wgv"], None, out_dtype=out_dtype, target=target) \
        .reshape(seq, heads, dim).transpose(0, 1).contiguous()
    # 门：每键通道的对数衰减 ⇒ 取负 softplus，恒 ≤ 0
    gate = ops.linear(proj, p["Wgg"], None, out_dtype=out_dtype, target=target) \
        .reshape(seq, heads, dim).transpose(0, 1)
    gate = -torch.nn.functional.softplus(gate).contiguous()
    # beta：每 (head, token) 一个标量 ∈ (0, 1)，写成 sigmoid
    beta = ops.linear(proj, p["Wgb"], None, out_dtype=out_dtype, target=target) \
        .reshape(seq, heads).transpose(0, 1).sigmoid().contiguous()
    out_g = ops.gdn_delta(qg, kg, vg, gate, beta, out_dtype=out_dtype, target=target)
    out_g_flat = out_g.transpose(0, 1).reshape(seq, heads * dim)
    gdn_out = ops.linear(out_g_flat, p["Wout_g"], None,
                         out_dtype=out_dtype, target=target)
    mix = proj + gdn_out                              # 融合支路（普通加法，走 autograd 图）
    # 10) LoRA 旁路：delta = s·(x@Aᵀ)@Bᵀ；s=1.0
    delta = ops.lora_apply(mix, p["lora_a"], p["lora_b"], scaling=1.0, base=None,
                           out_dtype=out_dtype, target=target)
    h_final = mix + delta
    # 11) 读出：ids 从 h_final 里逐行抽出（等价 identity，但走内核 gather/scatter 路）
    ids = torch.arange(seq, dtype=torch.int64, device=h_final.device)
    logits = ops.readout(h_final, ids, out_dtype=out_dtype, target=target)
    return logits


def rope_slots_default() -> tuple[int, ...]:
    """rope 的槽位默认走 rope_kernel.DEFAULT_SLOTS，与 P1 / ascend 入口同源。"""
    from ascend.kernels import rope_kernel
    return rope_kernel.DEFAULT_SLOTS


def run(target: str, device: torch.device, seed: int = 20261010,
        lr: float = 1e-2, verbose: bool = True,
        dtype: torch.dtype = torch.float32) -> tuple[float, float, list[str]]:
    """跑完整一步：前向→backward→检查全参梯度→Adam.step→再前向一次拿 loss1。

    白话：一次前向只够算梯度，不看更新效果；判据要"更新完再前向一次"的 loss1 才算
    一步真的走下去了。全参非 None 检查放在 optimizer.step 之前——step 一旦执行就
    把参数改了，届时"某个梯度漏接"变成"某个参数被更新了一小步"，反而看不出问题。
    返回 (loss0, loss1, missing) 三元组；missing 非空即失败。

    F2 位宽参数：`dtype=torch.float32`（默认）逐字维持现行行为、不装 guard；
    `dtype=torch.float16` 按生产位宽建参数/激活（角度表 fp32 混宽），全程罩归约纪律钉，
    步末把位宽账落进模块级 `LAST_AUDIT`（main 取去拼 DTYPE-TRACE；run 的返回三元组不动，
    tests 对 `run()` 最后一行是 `P2-STEP-…` 的断言因此不受影响）。
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device=device.type)
    gen.manual_seed(seed)

    seq, d, heads, dim = 16, 32, 2, 16
    x0, target_t = _make_inputs(seq, d, device, gen, dtype)
    p, cos, sin = _build_params(seq, d, heads, dim, rope_theta=1.0, device=device, gen=gen,
                                dtype=dtype)

    guarded = dtype == torch.float16
    if guarded:
        install_reduction_guard()
    try:
        # 前向 #0：拿 loss0（loss 归约钉 fp32 面：logits 先升宽再算，target 本就是 fp32 考卷；
        # fp32 档 .to(torch.float32) 恒等返回自身、连图节点都不多，数值与改前逐字同）
        logits0 = _forward(x0, p, cos, sin, seq, d, heads, dim, target, out_dtype=dtype)
        loss0 = (logits0.to(torch.float32) - target_t).pow(2).mean()

        # 反向
        for t in p.values():
            if t.grad is not None:
                t.grad = None
        loss0.backward()

        missing = [name for name, t in p.items() if t.grad is None]
        if missing:
            raise RuntimeError(f"以下参数 grad=None，反向接线未到位：{missing}")

        # 优化器：fp32 档逐字维持原样（params 即 leaves，Adam 直接在 fp32 面走）；
        # fp16 档按生产混精口径把**优化器状态与更新放在 fp32 master 副本**上——一手
        # 复现：fp16 leaf 直喂 Adam 时 eps=1e-8 舍入成 0、v_hat 在 |grad|≈7e-6 一档
        # 下溢 ⇒ step 出 ±inf/nan（F2 host 首跑 loss1=nan 的根源）。梯度先把 fp16
        # 原件升宽交给 master（累加面在 fp32），step 完再降回 leaves（出口按输入位宽
        # 交货），与内核侧"C2 §2 归约不降位"是同一纪律的两半。
        params = list(p.values())
        if dtype == torch.float32:
            opt = torch.optim.Adam(params, lr=lr)
            opt.step()
        else:
            master = [t.detach().float().requires_grad_(True) for t in params]
            for m, t in zip(master, params):
                m.grad = None if t.grad is None else t.grad.detach().float()
            torch.optim.Adam(master, lr=lr).step()
            with torch.no_grad():
                for m, t in zip(master, params):
                    t.copy_(m.detach().to(t.dtype))

        # 前向 #1：拿 loss1（同一个 x0/target）
        logits1 = _forward(x0, p, cos, sin, seq, d, heads, dim, target, out_dtype=dtype)
        loss1 = (logits1.to(torch.float32) - target_t).pow(2).mean()

        l0, l1 = float(loss0.detach().item()), float(loss1.detach().item())
        global LAST_AUDIT
        LAST_AUDIT = {
            "dtype": "fp16" if dtype == torch.float16 else "fp32",
            "params": len(p),
            "fp16": sum(1 for t in p.values() if t.dtype == torch.float16),
            "violations": reduction_violations() if guarded else [],
            "fp32_acc": int(_reduction_ledger["fp32_acc"]) if guarded else 0,
        }
        if verbose:
            print(f"P2-STEP-{target} loss0={l0:.6f} loss1={l1:.6f} "
                  f"reduced={l1 < l0} params_grad_ok={not missing}")
        return l0, l1, missing
    finally:
        if guarded:
            uninstall_reduction_guard()


def main(argv: list[str] | None = None) -> int:
    """命令行入口：`--target {cpu, ascend}` 选后端，判决不通过 exit 1（供 CI 硬门）。"""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", default=None, choices=list(ascend_env.TARGETS),
                    help=f"编译/发射后端；未指定时读 env {ascend_env.ENV_TARGET}（默认 cpu）")
    ap.add_argument("--lr", type=float, default=1e-2, help="Adam 学习率（默认 1e-2）")
    ap.add_argument("--seed", type=int, default=20261010, help="随机种子（默认 20261010）")
    ap.add_argument("--device", default=None,
                    help="torch 设备；cpu 后端默认 cpu，ascend 后端默认 npu（可覆盖）")
    ap.add_argument("--dtype", default="fp32", choices=sorted(DTYPE_BY_NAME),
                    help="位宽档：fp32（默认，现行行为逐字不变）/ fp16（生产位宽，F2；"
                         f"bf16 缺件由 C2 §8 钉在 plan 期，不在本入口放开）")
    args = ap.parse_args(argv)

    name = ascend_env.normalize_target(args.target)
    dt = DTYPE_BY_NAME[args.dtype]
    dev_default = "npu" if name == ascend_env.TARGET_ASCEND else "cpu"

    # H1：先问"这台机器的 torch 认不认 npu"，再谈设备。cpu 轨不 import，直接记 skip。
    if name == ascend_env.TARGET_ASCEND:
        reg = ensure_torch_npu()
    else:
        reg = {"status": "skip", "version": "not-imported",
               "detail": "cpu 轨不 import torch_npu（无 npu 机器会当场 ImportError）"}
    print(npu_reg_line(reg, name))

    try:
        dev = torch.device(args.device or dev_default)
    except Exception as exc:  # noqa: BLE001  旧版正是在这一行裸崩，整腿零判决
        print(f"P2-STEP-{name} FAIL: NPU-REG-FAIL 设备名不可用 {type(exc).__name__}: {exc}"
              f"（{reg['detail']}）", file=sys.stderr)
        return EXIT_DEVICE_UNAVAILABLE
    if name == ascend_env.TARGET_ASCEND and dev.type == "npu" and reg["status"] != "ok":
        print(f"P2-STEP-{name} FAIL: NPU-REG-FAIL {reg['detail']}", file=sys.stderr)
        return EXIT_DEVICE_UNAVAILABLE
    if name == ascend_env.TARGET_ASCEND and dev.type != "npu":
        print(f"WARN: target=ascend 但 device={dev}，ascend_env.active_backend 会判成 torch_eager",
              file=sys.stderr)

    # P1-1l：旁听回落公告。只挂钩子、原样转给 stderr（不吞公告、不改 warning 过滤器 ⇒
    # 日志形态与旧版逐字一致），拿到的条数就是"内核没走成"的次数。
    seen: list[str] = []
    show = warnings.showwarning

    def _hook(message, category, filename, lineno, file=None, line=None):
        seen.append(str(message))
        show(message, category, filename, lineno, file, line)

    warnings.showwarning = _hook
    try:
        l0, l1, missing = run(target=name, device=dev, seed=args.seed, lr=args.lr, dtype=dt)
    except Exception as exc:  # noqa: BLE001  顶层脚本：把 traceback 打出来再退码
        print(f"P2-STEP-{name} FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    finally:
        warnings.showwarning = show

    fb = ops.count_fallback_events(seen)
    report = ops.kernel_usage_report(name, fallback_events=fb)
    print(report["line"])
    gating = (name == ascend_env.TARGET_ASCEND
              or os.environ.get(ENV_KERNEL_USAGE_STRICT) == "1")
    if not report["ok"]:
        where = (f"exit={EXIT_KERNEL_UNUSED}" if gating
                 else f"未计入门禁（cpu 轨的 gdn_backward[cpu] 回落是已登记事实；"
                      f"要强验就 {ENV_KERNEL_USAGE_STRICT}=1）")
        print(f"KERNEL-USED-BLOCK: 未达项={report['failed']} ⇒ {where}", file=sys.stderr)

    # F2：位宽判决行（**新增行**，原 NPU-REG/P2-STEP/KERNEL-USED 三行的格式与顺序逐字不动）。
    # 退码优先级：grad 缺失(2) > loss 未降(3) > 归约纪律(6) > 内核门禁(5)——接线与下降判据保持
    # B 的原始优先级；纪律钉排在内核门禁之前：fp16 档归约落在粗尺子上时数值不可信，"走没走内核"
    # 已无从谈起。三证（KERNEL-USED）在 fp16 档与 fp32 档同账本、同判据、同退码。
    audit = dict(LAST_AUDIT or {})
    print(dtype_trace_line(audit, fb))
    if audit.get("violations"):
        print(f"DTYPE-TRACE-FAIL: fp16 档归约违规 {audit['violations'][:4]} "
              f"⇒ exit={EXIT_DTYPE_DISCIPLINE}", file=sys.stderr)
    if missing:
        print(f"P2-STEP-{name} FAIL: grad 缺失 {missing}", file=sys.stderr)
        return 2
    if not (l1 < l0):
        print(f"P2-STEP-{name} FAIL: loss 未下降（{l0} → {l1}）", file=sys.stderr)
        return 3
    if audit.get("violations"):
        return EXIT_DTYPE_DISCIPLINE
    if gating and not report["ok"]:
        return EXIT_KERNEL_UNUSED
    return 0


if __name__ == "__main__":
    sys.exit(main())
