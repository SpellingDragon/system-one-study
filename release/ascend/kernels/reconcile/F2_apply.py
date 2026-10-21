#!/usr/bin/env python3
"""reconcile/F2_apply.py — p2-13 F2 定点补丁（train_step.py + autograd_asc.py，锚点数≠1 则整批不动）。"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]  # release/
TS = ROOT / "ascend" / "train_step.py"
AG = ROOT / "ascend" / "autograd_asc.py"

EDITS: list[tuple[str, str, str, int]] = []


def edit(path_tag: str, old: str, new: str, count: int = 1) -> None:
    EDITS.append((path_tag, old, new, count))


# ============================== train_step.py ==============================

edit("TS", '''把"跑完=成功"这条误读彻底堵死。
''', '''把"跑完=成功"这条误读彻底堵死。

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
''')

edit("TS", '''#: 退出码：4=设备/注册不可用（H1），5=内核一次没走却自称通过（P1-1l）''',
     '''#: 退出码：4=设备/注册不可用（H1），5=内核一次没走却自称通过（P1-1l），6=fp16 档归约纪律钉触发（F2）''')

edit("TS", '''ENV_KERNEL_USAGE_STRICT = "SYS1_KERNEL_USAGE_STRICT"
''', '''ENV_KERNEL_USAGE_STRICT = "SYS1_KERNEL_USAGE_STRICT"

#: F2：CLI 位宽档名 → torch.dtype。**没有 bf16 档**：bisheng vector 面 bf16 标量 cast 的缺件
#: 取证由 C2 §8 钉死（三件 plan 期显式拒），训练步入口同样不放开。
DTYPE_BY_NAME = {"fp32": torch.float32, "fp16": torch.float16}
#: F2：退码 6 = fp16 档出现"归约在 fp16 做"。编号续在 B 的 4/5 之后，不改它们的语义与判读。
EXIT_DTYPE_DISCIPLINE = 6
''')

# ---------------- guard 段（插在 _make_inputs 之前） ----------------
edit("TS", '''def _make_inputs(seq: int, d: int, device: torch.device, gen: torch.Generator):''',
'''# ------------------------------------------------------------------ F2：fp16 归约纪律钉
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


def _make_inputs(seq: int, d: int, device: torch.device, gen: torch.Generator):''')

edit("TS", '''def _make_inputs(seq: int, d: int, device: torch.device, gen: torch.Generator):''',
     '''def _make_inputs(seq: int, d: int, device: torch.device, gen: torch.Generator,
                 dtype: torch.dtype = torch.float32):''')

edit("TS", '''    x0 = torch.randn(seq, d, device=device, generator=gen)
    target = torch.randn(seq, d, device=device, generator=gen)
    return x0, target''',
     '''    x0 = torch.randn(seq, d, device=device, generator=gen)
    target = torch.randn(seq, d, device=device, generator=gen)
    if dtype != torch.float32:
        x0 = x0.to(dtype)      # 主干输入按生产位宽；target 保持 fp32 考卷（loss 归约在 fp32 面做）
    return x0, target''')

edit("TS", '''def _build_params(seq: int, d: int, heads: int, dim: int, rope_theta: float,
                  device: torch.device, gen: torch.Generator) -> dict[str, torch.Tensor]:''',
     '''def _build_params(seq: int, d: int, heads: int, dim: int, rope_theta: float,
                  device: torch.device, gen: torch.Generator,
                  dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:''')

edit("TS", '''        "lora_a": _r(8, d, scale=0.05), "lora_b": _r(d, 8, scale=0.05),
    }
    cos = torch.ones(seq, dim // 2, device=device) * (rope_theta ** 0)   # 角度表本件不参与训练''',
     '''        "lora_a": _r(8, d, scale=0.05), "lora_b": _r(d, 8, scale=0.05),
    }
    if dtype != torch.float32:
        # F2 生产位宽：先按与 fp32 档**逐字同**的 RNG 造 fp32 值，再统一降位（fp16=同值取整，
        # 不是重采样），两档数字差才只反映位宽、不反映样本。cos/sin 角度表**保持 fp32**——
        # "qkv fp16 + 表 fp32" 就是 C2 §7 的生产混宽形状（键 `qfloat16afloat32`）。
        p = {k: v.detach().to(dtype).requires_grad_(True) for k, v in p.items()}
    cos = torch.ones(seq, dim // 2, device=device) * (rope_theta ** 0)   # 角度表本件不参与训练''')

edit("TS", '''def _forward(x0: torch.Tensor, p: dict[str, torch.Tensor], cos: torch.Tensor,
             sin: torch.Tensor, seq: int, d: int, heads: int, dim: int,
             target: str) -> torch.Tensor:''',
     '''def _forward(x0: torch.Tensor, p: dict[str, torch.Tensor], cos: torch.Tensor,
             sin: torch.Tensor, seq: int, d: int, heads: int, dim: int,
             target: str, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:''')

# 14 处门面调用统一改吃 out_dtype 参数（fp32 档传 torch.float32 ⇒ 调用值与旧行完全一致）
edit("TS", "out_dtype=torch.float32", "out_dtype=out_dtype", 14)

edit("TS", '''def run(target: str, device: torch.device, seed: int = 20261010,
        lr: float = 1e-2, verbose: bool = True) -> tuple[float, float, list[str]]:''',
     '''def run(target: str, device: torch.device, seed: int = 20261010,
        lr: float = 1e-2, verbose: bool = True,
        dtype: torch.dtype = torch.float32) -> tuple[float, float, list[str]]:''')

edit("TS", '''    返回 (loss0, loss1, missing) 三元组；missing 非空即失败。''',
     '''    返回 (loss0, loss1, missing) 三元组；missing 非空即失败。

    F2 位宽参数：`dtype=torch.float32`（默认）逐字维持现行行为、不装 guard；
    `dtype=torch.float16` 按生产位宽建参数/激活（角度表 fp32 混宽），全程罩归约纪律钉，
    步末把位宽账落进模块级 `LAST_AUDIT`（main 取去拼 DTYPE-TRACE；run 的返回三元组不动，
    tests 对 `run()` 最后一行是 `P2-STEP-…` 的断言因此不受影响）。''')

edit("TS", '''    seq, d, heads, dim = 16, 32, 2, 16
    x0, target_t = _make_inputs(seq, d, device, gen)
    p, cos, sin = _build_params(seq, d, heads, dim, rope_theta=1.0, device=device, gen=gen)''',
     '''    seq, d, heads, dim = 16, 32, 2, 16
    x0, target_t = _make_inputs(seq, d, device, gen, dtype)
    p, cos, sin = _build_params(seq, d, heads, dim, rope_theta=1.0, device=device, gen=gen,
                                dtype=dtype)''')

edit("TS", '''    # 前向 #0：拿 loss0
    logits0 = _forward(x0, p, cos, sin, seq, d, heads, dim, target)
    loss0 = (logits0 - target_t).pow(2).mean()

    # 反向
    for t in p.values():
        if t.grad is not None:
            t.grad = None
    loss0.backward()

    missing = [name for name, t in p.items() if t.grad is None]
    if missing:
        raise RuntimeError(f"以下参数 grad=None，反向接线未到位：{missing}")

    opt = torch.optim.Adam(list(p.values()), lr=lr)
    opt.step()

    # 前向 #1：拿 loss1（同一个 x0/target）
    logits1 = _forward(x0, p, cos, sin, seq, d, heads, dim, target)
    loss1 = (logits1 - target_t).pow(2).mean()

    l0, l1 = float(loss0.detach().item()), float(loss1.detach().item())
    if verbose:
        print(f"P2-STEP-{target} loss0={l0:.6f} loss1={l1:.6f} "
              f"reduced={l1 < l0} params_grad_ok={not missing}")
    return l0, l1, missing''',
     '''    guarded = dtype == torch.float16
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

        opt = torch.optim.Adam(list(p.values()), lr=lr)
        opt.step()

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
            uninstall_reduction_guard()''')

# ---------------- main() ----------------
edit("TS", '''    ap.add_argument("--device", default=None,
                    help="torch 设备；cpu 后端默认 cpu，ascend 后端默认 npu（可覆盖）")
    args = ap.parse_args(argv)''',
     '''    ap.add_argument("--device", default=None,
                    help="torch 设备；cpu 后端默认 cpu，ascend 后端默认 npu（可覆盖）")
    ap.add_argument("--dtype", default="fp32", choices=sorted(DTYPE_BY_NAME),
                    help="位宽档：fp32（默认，现行行为逐字不变）/ fp16（生产位宽，F2；"
                         f"bf16 缺件由 C2 §8 钉在 plan 期，不在本入口放开）")
    args = ap.parse_args(argv)''')

edit("TS", '''    name = ascend_env.normalize_target(args.target)
    dev_default = "npu" if name == ascend_env.TARGET_ASCEND else "cpu"''',
     '''    name = ascend_env.normalize_target(args.target)
    dt = DTYPE_BY_NAME[args.dtype]
    dev_default = "npu" if name == ascend_env.TARGET_ASCEND else "cpu"''')

edit("TS", '''        l0, l1, missing = run(target=name, device=dev, seed=args.seed, lr=args.lr)''',
     '''        l0, l1, missing = run(target=name, device=dev, seed=args.seed, lr=args.lr, dtype=dt)''')

edit("TS", '''    report = ops.kernel_usage_report(name, fallback_events=ops.count_fallback_events(seen))
    print(report["line"])''',
     '''    fb = ops.count_fallback_events(seen)
    report = ops.kernel_usage_report(name, fallback_events=fb)
    print(report["line"])''')

edit("TS", '''        print(f"KERNEL-USED-BLOCK: 未达项={report['failed']} ⇒ {where}", file=sys.stderr)
    if missing:''',
     '''        print(f"KERNEL-USED-BLOCK: 未达项={report['failed']} ⇒ {where}", file=sys.stderr)

    # F2：位宽判决行（**新增行**，原 NPU-REG/P2-STEP/KERNEL-USED 三行的格式与顺序逐字不动）。
    # 退码优先级：grad 缺失(2) > loss 未降(3) > 归约纪律(6) > 内核门禁(5)——接线与下降判据保持
    # B 的原始优先级；纪律钉排在内核门禁之前：fp16 档归约落在粗尺子上时数值不可信，"走没走内核"
    # 已无从谈起。三证（KERNEL-USED）在 fp16 档与 fp32 档同账本、同判据、同退码。
    audit = dict(LAST_AUDIT or {})
    print(dtype_trace_line(audit, fb))
    if audit.get("violations"):
        print(f"DTYPE-TRACE-FAIL: fp16 档归约违规 {audit['violations'][:4]} "
              f"⇒ exit={EXIT_DTYPE_DISCIPLINE}", file=sys.stderr)
    if missing:''')

edit("TS", '''    if not (l1 < l0):
        print(f"P2-STEP-{name} FAIL: loss 未下降（{l0} → {l1}）", file=sys.stderr)
        return 3
    if gating and not report["ok"]:''',
     '''    if not (l1 < l0):
        print(f"P2-STEP-{name} FAIL: loss 未下降（{l0} → {l1}）", file=sys.stderr)
        return 3
    if audit.get("violations"):
        return EXIT_DTYPE_DISCIPLINE
    if gating and not report["ok"]:''')

# ============================== autograd_asc.py ==============================

edit("AG", '''        ctx.save_for_backward(A, W)
        ctx.has_bias = bias is not None
        ctx.target = target''',
     '''        ctx.save_for_backward(A, W)
        ctx.has_bias = bias is not None
        ctx.target = target
        ctx.bias_dtype = None if bias is None else bias.dtype   # F2：db 的交货位宽按输入原件''')

edit("AG", '''        db = dC.float().sum(dim=0) if ctx.has_bias else None
        return dA, dW, db, None, None''',
     '''        db = dC.float().sum(dim=0) if ctx.has_bias else None
        if db is not None:
            # F2 位宽传导：db 的**归约在 fp32 面做**（上行 .float()），出口按原件位宽交货；
            # bias 本就是 fp32 时 `.to()` 恒等返回 self，fp32 档行为逐字不变。
            db = db.to(ctx.bias_dtype)
        return dA, dW, db, None, None''')

edit("AG", '''            grads = gdn_kernel.backward(q, k, v, g, beta, dout, ctx.target)
        return (*grads[:5], None, None)''',
     '''            grads = gdn_kernel.backward(q, k, v, g, beta, dout, ctx.target)
        # F2 位宽传导：gdn 五梯的**内核出口恒 fp32**（C2 §2：归约不降位），交回 autograd 前
        # 按各自输入原件位宽交货。fp32 档 `.to()` 恒等返回 self ⇒ 行为与改前逐字同；
        # fp16 档不依赖引擎的静默折算，交货位宽在件上就钉死。
        dq_, dk_, dv_, dg_, db_ = grads[:5]
        return (dq_.to(q.dtype), dk_.to(k.dtype), dv_.to(v.dtype),
                dg_.to(g.dtype), db_.to(beta.dtype), None, None)''')


def main() -> int:
    tag_path = {"TS": TS, "AG": AG}
    texts = {tag: path.read_text(encoding="utf-8") for tag, path in tag_path.items()}
    miss = []
    for i, (tag, old, new, count) in enumerate(EDITS):
        n = texts[tag].count(old)
        if n != count:
            miss.append((i, tag, n, count, old.splitlines()[0][:60]))
            continue
        texts[tag] = texts[tag].replace(old, new)
    if miss:
        print(f"锚点异常 {len(miss)} 处（未做任何改动）：")
        for i, tag, n, want, head in miss:
            print(f"  #{i} {tag} 命中={n} 期望={want} :: {head}")
        return 2
    for tag, path in tag_path.items():
        path.write_text(texts[tag], encoding="utf-8")
        print(f"patched {path.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
