#!/usr/bin/env python3
"""gradcheck_card.py — 昇腾/CPU 八门面「数值梯度判决件」（P1-1y：把悬空的 gradcheck NPU 档落地）。

【做什么】对 `ascend/autograd_asc.py` 的八个可微门面（linear / add_ln / rope / attn_sw /
gdn_delta / gdn_conv / readout / lora_apply），在**同一次运行里**同时拿两样东西：
  ① 数值梯度：fp32 中心差分（ε 按件敏感性择，依据见 `reconcile/B3_GRADCHECK_DESIGN.md` §2）；
  ② 解析梯度：门面 autograd 反传（内核反向 / 闭式 / fp32 尺子，按各件既有处置）。
两路按 **Frobenius 相对误差**对拍，阈值**沿用** `tests/test_ascend_gradcheck.py` 的 TOL
（rtol=1e-4 / atol=1e-5，**不另造**）。判决行：
    GRADCHECK-<op> PASS max_rel=<v> eps=<e>
    GRADCHECK-<op> FAIL max_rel=<v> eps=<e>
    GRADCHECK-<op> INVALID reason=no-kernel | blockers:<k> | fallback
末行 `GRADCHECK-SUMMARY ops=8 pass= fail= invalid= kernel_used=<n>/<8>`；退出码：pass 全绿 0 /
有 FAIL 1 / 有 INVALID 2。

【为什么必须与 KERNEL-USED 三证同窗绑定（R14 防假绿）】内核在编译/发射失败时会**静默**落回
torch eager（`ascend_env.get_compiled` 返回 None ⇒ 入口落 `_eager`）。eager 与解析梯度数学一致，
于是「gradcheck 一致」会**伪装**成「内核 gradcheck 通过」——M2 已证 Python 层量不到内核内部。
所以每件对拍前先取三证：**本趟新出现的 `<target>|` 编译键 ≥1 ∧ 无未申报 blocker ∧ 无未申报
回落**，任一不满足判 **INVALID**（既不是 PASS 也不是 FAIL：这本就没量到内核）。
唯一已申报边界：gdn_delta 在 **cpu** target 上反向按设计落 torch 尺子并登记 `gdn_backward[cpu]`
（见 `autograd_asc._GdnFn` 与 `tests/test_ascend_gradcheck.py::test_gdn_backward_ascend_kernelized_cpu_ruler_grads_correct`），
不使该件 INVALID；ascend target 反向已内核化，无此边界。详见 `reconcile/B3_GRADCHECK_DESIGN.md` §4。

【怎么做】① 每件一个小形状（真机秒级）；② ε 逐件取「小 ε 抗截断 / 大 ε 抗舍入」的折中点；
③ 解析侧标量按门面位宽（fp32）累加——归约面沿 F2 纪律；差分本身用 fp64 累加**只作量具**，
不参与产品归约；④ `--target cpu`（host 可跑）与 `--target ascend` 走**同一判据族**；ascend 轨
在无设备容器里用 `--no-run` 只判编译键（发射桩掉，手法与 `compile_all_asc.py` 同构）。

跑法：
    .venv/bin/python ascend/gradcheck_card.py --target cpu
    python3 ascend/gradcheck_card.py --target ascend --no-run          # 卡窗/容器
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))      # .../release/ascend
_RELEASE = os.path.dirname(_HERE)                        # .../release
if _RELEASE not in sys.path:
    sys.path.insert(0, _RELEASE)

import torch  # noqa: E402

from ascend import autograd_asc as ops  # noqa: E402
from ascend.autograd_asc import count_fallback_events  # noqa: E402
from ascend.kernels import ascend_env  # noqa: E402

#: 八门面（判决面）
OPS = ("linear", "add_ln", "rope", "attn_sw", "gdn_delta", "gdn_conv", "readout", "lora_apply")

#: 逐件中心差分步长 ε（依据见 B3_GRADCHECK_DESIGN.md §2）：小 ε 抗截断误差、大 ε 抗 fp32 舍入，
#: 每件取使 Frobenius 相对误差最小的折中值（真机秒级，方向数不大）。
EPS = {
    "linear": 0.05, "add_ln": 0.02, "rope": 0.05, "attn_sw": 0.02,
    "gdn_delta": 0.01, "gdn_conv": 0.01, "readout": 0.05, "lora_apply": 0.05,
}

#: 对拍阈值：**沿用** tests/test_ascend_gradcheck.py 的 TOL（该文件 line 49），不另造。
RTOL, ATOL = 1e-4, 1e-5

#: 已申报的「设计边界」：(op, target) → 该 target 下允许出现的 blocker/回落键。
#: 不在表内的任何新增 blocker/回落一律判 INVALID（防内核静默落 eager）。
DECLARED_BLOCKERS = {
    ("gdn_delta", "cpu"): {"gdn_backward[cpu]"},
}

#: host 侧缺卡时给各代理的保底形态标记（与 compile_all_asc 同口径）
_FALLBACK_TOKEN = ops.FALLBACK_WARNING_TOKEN


class _LaunchSkip(Exception):
    """仅 `--no-run`：kernel 被"发射"时抛出——本模式只判编译，不判发射（无设备）。"""


def _rn(device, *shape):
    return torch.randn(*shape, dtype=torch.float32, device=device)


def _build(op: str, target: str | None, device: torch.device):
    """每件返回 (f, base)：f 只吃**可微输入**（常量如 rope 的 cos/sin、readout 的行号闭在 f 里）。

    形状取"真机秒级"的小件，且同时满足各件 `plan()` 接受域（小块宽整数倍 / dim%16 / fp32 连续），
    免得测到的是"plan 拒形状后静默落 eager"那条假路（B3_GRADCHECK_DESIGN.md §3）。
    """
    torch.manual_seed(20241011)
    if op == "linear":
        A, W, b = _rn(device, 6, 16), _rn(device, 16, 16), _rn(device, 16)
        f = lambda A_, W_, b_: ops.linear(A_, W_, b_, out_dtype=torch.float32, target=target)
        return f, [A, W, b]
    if op == "add_ln":
        x, r, g, b = _rn(device, 6, 16), _rn(device, 6, 16), _rn(device, 16), _rn(device, 16)
        f = lambda x_, r_, g_, b_: ops.add_ln(x_, r_, g_, b_, eps=1e-5,
                                              out_dtype=torch.float32, target=target)[0]
        return f, [x, r, g, b]
    if op == "rope":
        qkv, cos, sin = _rn(device, 4, 3, 2, 8), _rn(device, 4, 4), _rn(device, 4, 4)
        f = lambda q_: ops.rope(q_, cos, sin, rotate_slots=(0, 1), target=target)
        return f, [qkv]
    if op == "attn_sw":
        q, k, v = _rn(device, 1, 5, 16), _rn(device, 1, 5, 16), _rn(device, 1, 5, 16)
        f = lambda q_, k_, v_: ops.attn_sw(q_, k_, v_, 4, scale=16 ** -0.5,
                                           out_dtype=torch.float32, target=target)
        return f, [q, k, v]
    if op == "gdn_delta":
        q, k, v = _rn(device, 2, 4, 8), _rn(device, 2, 4, 8), _rn(device, 2, 4, 8)
        g = -torch.rand(2, 4, 8, dtype=torch.float32, device=device) * 0.2
        beta = torch.rand(2, 4, dtype=torch.float32, device=device) * 0.9 + 0.05
        f = lambda q_, k_, v_, g_, be_: ops.gdn_delta(q_, k_, v_, g_, be_,
                                                      out_dtype=torch.float32, target=target)
        return f, [q, k, v, g, beta]
    if op == "gdn_conv":
        x, w, b = _rn(device, 2, 8, 6), _rn(device, 8, 4), _rn(device, 8)
        f = lambda x_, w_, b_: ops.gdn_conv(x_, w_, b_, out_dtype=torch.float32, target=target)
        return f, [x, w, b]
    if op == "readout":
        rows = _rn(device, 8, 16)
        ids = torch.tensor([3, 0, 3, 7, 5], dtype=torch.int64, device=device)
        f = lambda r_: ops.readout(r_, ids, out_dtype=torch.float32, target=target)
        return f, [rows]
    if op == "lora_apply":
        x, a, b = _rn(device, 6, 16), _rn(device, 16, 16) * 0.2, _rn(device, 16, 16) * 0.2
        f = lambda x_, a_, b_: ops.lora_apply(x_, a_, b_, scaling=0.5,
                                              out_dtype=torch.float32, target=target)
        return f, [x, a, b]
    raise KeyError(op)


def _loss(o: torch.Tensor, dy: torch.Tensor) -> float:
    """差分标量 <o, dy>；**fp64 累加只作量具**（差分要的是一位更准的读数，非产品归约）。"""
    return float((o.double() * dy.double()).sum())


def _numeric_grad(f, base, eps: float, dy: torch.Tensor):
    """逐元素中心差分：num[i][k] = [<f(x+εe_k),dy> - <f(x-εe_k),dy>] / (2ε)。"""
    grads = []
    for t in base:
        num = torch.zeros_like(t)
        flat = t.reshape(-1)
        nflat = num.reshape(-1)
        for k in range(flat.numel()):
            orig = float(flat[k])
            flat[k] = orig + eps
            lp = f(*base)
            flat[k] = orig - eps
            lm = f(*base)
            flat[k] = orig
            nflat[k] = _loss(lp, dy) - _loss(lm, dy)
        grads.append(num / (2.0 * eps))
    return grads


def _gradcheck_op(op: str, target: str, device: torch.device,
                  n_dirs: int, inject=None) -> dict:
    """数值 vs 解析：返回 {rel, abs_err, ana_norm, ok, eps, fwd_keys, exc}。

    `fwd_keys` = **前向**这一步新编译出的 `<target>|` 键——判"内核真被前向用到"的唯一硬证据。
    回退路径不留痕；**不能拿反向键顶数**：plan 一旦拒形状，前向会静默落 eager，而反向仍可能
    编出 `gemm_dw` 之类键，只看"有没有键"就会把假绿放进来（B3_GRADCHECK_DESIGN.md §3.2）。
    """
    k0 = set(ascend_env.compiled_keys())
    f, raw = _build(op, target, device)
    eps = EPS.get(op, 0.01)

    def _fwd_keys():
        return [k for k in ascend_env.compiled_keys()
                if k not in k0 and k.startswith(f"{target}|")]

    try:
        with torch.no_grad():
            out0 = f(*raw)                                   # 前向：内核 or（静默）eager
        fwd_keys = _fwd_keys()
        torch.manual_seed(101)
        dys = [torch.randn(out0.shape, dtype=torch.float32, device=device) for _ in range(n_dirs)]
        base = [t.detach().clone().contiguous() for t in raw]    # 数值侧（独立副本，防污染）
        leaves = [t.clone().detach().requires_grad_(True) for t in raw]
        sq_err = sq_ana = 0.0
        for dy in dys:
            out = f(*leaves)
            g_ana = torch.autograd.grad((out * dy).sum(), leaves)  # 解析：门面 autograd 反传（fp32 归约）
            if inject is not None:
                g_ana = [inject(g) for g in g_ana]
            with torch.no_grad():
                g_num = _numeric_grad(f, base, eps, dy)
            for a, n in zip(g_ana, g_num):
                d = n.double() - a.double()
                sq_err += float((d ** 2).sum())
                sq_ana += float((a.double() ** 2).sum())
        abs_err, ana_norm = sq_err ** 0.5, sq_ana ** 0.5
        rel = abs_err / max(ana_norm, 1e-30)
        return {"rel": rel, "abs_err": abs_err, "ana_norm": ana_norm, "eps": eps,
                "fwd_keys": fwd_keys, "exc": None,
                "ok": abs_err <= ATOL + RTOL * ana_norm}
    except Exception as e:  # noqa: BLE001  前向/反向任何抛出都收成判决，不让单件带崩整窗
        return {"rel": None, "abs_err": None, "ana_norm": None, "eps": eps,
                "fwd_keys": _fwd_keys(), "exc": e, "ok": False}


def _fallback_keys(messages) -> list[str]:
    """从回落公告里抠出"哪个键回落了"：文案 `方言后端 [<key>] 不可用，已回退 torch eager：…`。

    注意键自身可能含方括号（如 `gdn_backward[cpu]`），故取 token 前**最后**一个 `]` 为闭合处。
    """
    out = []
    for m in messages:
        s = str(m)
        tok = s.find(_FALLBACK_TOKEN)
        i = s.find("[")
        if tok < 0 or i < 0:
            continue
        j = s.rfind("]", i + 1, tok)
        out.append(s[i + 1:j] if i < j < tok else "?")
    return out


def _binding(op: str, target: str, fwd_keys, new_blockers, fb_keys) -> tuple[bool, str]:
    """KERNEL-USED 三证（本趟增量口径）：**前向**编译键 ≥1 ∧ 无未申报 blocker ∧ 无未申报回落。"""
    declared = DECLARED_BLOCKERS.get((op, target), set())
    undeclared_blk = [k for k in new_blockers if k not in declared]
    undeclared_fb = [k for k in fb_keys if k not in declared]
    if not fwd_keys:
        return False, "no-kernel"
    if undeclared_blk:
        return False, "blockers:" + ",".join(sorted(undeclared_blk))
    if undeclared_fb:
        return False, "fallback"
    return True, ""


def _drive_no_run(op: str, target: str, device: torch.device) -> None:
    """只触发编译（不发射）：驱动门面一次，kernel 发射被替身抛 _LaunchSkip。"""
    f, raw = _build(op, target, device)
    try:
        with torch.no_grad():
            f(*raw)
    except _LaunchSkip:
        pass


def run_one(op: str, target: str, *, no_run: bool, device: torch.device,
            n_dirs: int, inject=None) -> dict:
    """跑一件：先取三证增量，再判决；返回 {op, verdict, reason, rel, eps, keys, fb, ms, err}。

    异常处置：门面**未用内核**却抛（如 lora 的 `_eager_grad` 在 no_grad 下搭不出图，见
    B3_GRADCHECK_DESIGN.md §3.3）⇒ INVALID（这本就没量到内核）；门面**已用内核**却抛 ⇒ FAIL（红）。
    """
    ascend_env.reset()
    before_keys = set(ascend_env.compiled_keys())
    before_blockers = set(ascend_env.blockers())
    t0 = time.time()
    result, exc, wl = None, None, []
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            if no_run:
                _drive_no_run(op, target, device)
            else:
                result = _gradcheck_op(op, target, device, n_dirs, inject)
    except Exception as e:  # noqa: BLE001  只可能来自 no_run 的驱动；run 模式异常已在 _gradcheck_op 内收走
        exc = e
    fb = _fallback_keys([w.message for w in wl])
    ms = time.time() - t0
    all_new = [k for k in ascend_env.compiled_keys()
               if k not in before_keys and k.startswith(f"{target}|")]
    new_blk = [k for k in ascend_env.blockers() if k not in before_blockers]
    fwd_keys = result["fwd_keys"] if result is not None else all_new
    if result is not None and result.get("exc") is not None:
        exc = result["exc"]
    ok, reason = _binding(op, target, fwd_keys, new_blk, fb)
    base = {"op": op, "reason": "", "eps": EPS.get(op), "keys": fwd_keys,
            "all_keys": all_new, "fb": fb, "ms": ms, "err": None}
    if not ok:
        return {**base, "verdict": "INVALID", "reason": reason, "rel": None}
    if exc is not None:
        head = f"{type(exc).__name__}: {(str(exc).splitlines() or [''])[0]}"
        return {**base, "verdict": "FAIL", "rel": None, "err": head}
    if no_run:
        return {**base, "verdict": "PASS", "rel": None}
    return {**base, "verdict": "PASS" if result["ok"] else "FAIL", "rel": result["rel"]}


# ────────────────────────── `--target ascend --no-run` 的编译替身（同 compile_all_asc 手法）──────
_SAVED: dict = {}


def _install_stubs() -> None:
    """强制走 tilelang 后端 + 真编译产物换成"调用即跳发射"的替身（只判编译面）。"""
    import tilelang
    for n in ("backend_available", "_npu_present", "active_backend"):
        _SAVED[n] = getattr(ascend_env, n)
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG   # 绕设备-目标匹配门
    real = tilelang.compile
    _SAVED["tilelang.compile"] = real

    def _compile(*a, **k):
        kern = real(*a, **k)            # 真编译；失败原样抛，由 get_compiled 记 blocker

        class _Dummy:
            def __call__(self, *aa, **kk):
                raise _LaunchSkip()
            def __getattr__(self, n):
                return lambda *aa, **kk: 0
        return _Dummy()

    tilelang.compile = _compile


def _restore_stubs() -> None:
    import tilelang
    for n, v in _SAVED.items():
        if n == "tilelang.compile":
            tilelang.compile = v
        else:
            setattr(ascend_env, n, v)
    _SAVED.clear()


def _line(res: dict, no_run: bool) -> str:
    op = res["op"]
    if res["verdict"] == "INVALID":
        return f"GRADCHECK-{op} INVALID reason={res['reason']}"
    if no_run:
        return f"GRADCHECK-{op} PASS max_rel=- eps={res['eps']} keys={len(res['keys'])} mode=no-run"
    rel = "n/a" if res.get("rel") is None else f"{res['rel']:.3e}"
    return f"GRADCHECK-{op} {res['verdict']} max_rel={rel} eps={res['eps']:g}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="八门面数值梯度判决件（P1-1y）")
    ap.add_argument("--target", default=None, choices=("cpu", "ascend"),
                    help="cpu（host 数值对拍）| ascend（卡窗；无设备用 --no-run 只判编译）")
    ap.add_argument("--no-run", action="store_true", help="只判编译键，桩掉发射（容器/无卡）")
    ap.add_argument("--op", default=None, help="只跑某件（缺省八件全跑）")
    ap.add_argument("--dirs", type=int, default=2, help="每件对拍方向数（缺省 2）")
    args = ap.parse_args(argv)

    target = ascend_env.normalize_target(args.target)
    if target == ascend_env.TARGET_ASCEND:
        os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
    npu_ok = hasattr(torch, "npu") and bool(getattr(torch.npu, "is_available", lambda: False)())
    device = torch.device("npu") if (target == ascend_env.TARGET_ASCEND and npu_ok) else torch.device("cpu")

    names = [args.op] if args.op else list(OPS)
    bad = [o for o in names if o not in OPS]
    if bad:
        print(f"GRADCHECK-usage FAIL 未知件 {bad}；可用：{','.join(OPS)}")
        return 1

    import tilelang
    mode = "no-run(compile-only)" if args.no_run else "run(numeric-gradcheck)"
    print(f"=== gradcheck_card target={target} mode={mode} device={device.type} "
          f"tilelang={tilelang.__version__} dirs={args.dirs} rtol={RTOL:g} atol={ATOL:g} ===",
          flush=True)

    if args.no_run:
        _install_stubs()
    results = []
    try:
        for op in names:
            res = run_one(op, target, no_run=args.no_run, device=device, n_dirs=args.dirs)
            results.append(res)
            print(_line(res, args.no_run), flush=True)
            keys = ", ".join(sorted(k.split("|", 1)[1] if "|" in k else k for k in res["keys"]))
            extra = f"fwd_keys=[{keys}]"
            nother = len(res.get("all_keys", res["keys"])) - len(res["keys"])
            if nother > 0:
                extra += f"+{nother}other"
            if res["fb"]:
                extra += f" fallback={res['fb']}"
            if res.get("err"):
                extra += f" err={res['err']}"
            print(f"    · {extra} wall={res['ms']:.2f}s", flush=True)
    finally:
        if args.no_run:
            _restore_stubs()

    n_pass = sum(1 for r in results if r["verdict"] == "PASS")
    n_fail = sum(1 for r in results if r["verdict"] == "FAIL")
    n_inv = sum(1 for r in results if r["verdict"] == "INVALID")
    n_used = sum(1 for r in results if r["keys"])
    print(f"GRADCHECK-SUMMARY ops={len(results)} pass={n_pass} fail={n_fail} invalid={n_inv} "
          f"kernel_used={n_used}/{len(results)}", flush=True)
    if n_fail:
        return 1
    if n_inv:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
