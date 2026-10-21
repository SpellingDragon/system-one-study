#!/usr/bin/env python3
"""verify_gdn.py — P1-4 GDN 三子项的 910B 编译判决 + 语义不回归闸（执行代理 J）。

判的是三件事（对应任务书三子项）：
  [FWD]  `gdn_asc.gdn_asc_impl` 的 delta-rule 前向：SimtVF 换成 910B 的 AIV 标量面后能不能出 .o
  [BWD]  `gdn_asc.gdn_bwd_asc_impl` 的反向伴随件（六梯）能不能出 .o，且 `backward` 真走模具
  [CONV] 新增的 conv 入口 `gdn_conv_asc.conv_forward`（910B 标量档）能不能出 .o
外加两条口径检查：[CPU] target=cpu 的语义不许因为本次重写而回归；[STATIC] 接口面上
`BWD_STATUS` 与实况一致、ascend 正文里不许还留着 SimtVF/Parallel 这类 950 载体。

判决纪律：
  * 每次全新 `TILELANG_CACHE_DIR`（tilelang 按 kernel 源码哈希命中缓存，不换就是陈旧 PASS）。
  * 真编后把 kernel 换成 dummy：调用只计数不真发（_STATS[launches]）—— 本件只判**编译**，
    不判发射（无设备）。反向是「前向重算落 Hist + 伴随递推」**两趟**流水，dummy 若在第一趟就抛，
    第二趟的伴随件永远进不了判决面（本波踩过：features 里那份 59 行其实是前向产物）。
  * 判据来自 `ascend_env.compiled_keys()` 里出现 `ascend|` 前缀的该件键（真编过才算走内核，R14）。
  * `active_backend` 有一道设备门（ascend target + CPU 张量 ⇒ 判 torch_eager，这是接口该有的
    安全门，不许为编译判决而拆掉），所以驱动**入口**（forward/backward/conv_forward）时把
    这道门 stub 成 tilelang；直接驱动 `plan/run` 则不需要（它们不看设备）。

用法（容器内，release 挂在 /work）：
  python3 /work/ascend/kernels/reconcile/verify_gdn.py                 # 编译判决全家（含 pure/ubstage 保留档）
  python3 /work/ascend/kernels/reconcile/verify_gdn.py --gate          # 只跑首推档（ub/scalar），退出码可用
  python3 /work/ascend/kernels/reconcile/verify_gdn.py --cpu           # 追加 target=cpu 语义对拍（需 c 后端）
"""
import argparse
import inspect
import os
import re
import sys
import tempfile
import traceback

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
os.environ.setdefault("TILELANG_CACHE_DIR", tempfile.mkdtemp(prefix="tlc_j_"))

HERE = os.path.dirname(os.path.abspath(__file__))            # .../release/ascend/kernels/reconcile
RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))   # .../release
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

import torch  # noqa: E402
import tilelang  # noqa: E402

from ascend.kernels import ascend_env, gdn_asc, gdn_conv_asc, gdn_kernel  # noqa: E402


class _LaunchSkip(Exception):
    """kernel 被"发射"时抛它：本件只判编译，不判发射（无设备）。"""


_STATS = {"src": "", "bytes": 0, "compiles": 0, "launches": 0}


def install_stubs(device_gate: bool = False):
    """强制走 tilelang 后端 + 把真编译产物换成"已编译、调用即跳发射"的替身。"""
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    if device_gate:
        # 只 stub 设备的在场性，不改接口的判据本体：ascend 产物只该在 npu 上发射
        ascend_env.active_backend = lambda target=None, device=None: ascend_env.TILELANG
    real_compile = tilelang.compile
    if getattr(real_compile, "_j_stubbed", False):
        return

    def _compile(*a, **k):
        kern = real_compile(*a, **k)                 # 真编译；失败原样抛，由 get_compiled 记 blocker
        src = kern.get_kernel_source() if hasattr(kern, "get_kernel_source") else ""
        _STATS["src"] = src
        _STATS["bytes"] = len(src)
        _STATS["compiles"] += 1

        class _Dummy:
            def __call__(self, *aa, **kk):
                _STATS["launches"] += 1
                return None

            def __getattr__(self, n):
                return lambda *aa, **kk: 0

        return _Dummy()

    _compile._j_stubbed = True
    tilelang.compile = _compile


def f32(*shape):
    return torch.randn(*shape, dtype=torch.float32)


def feats(src: str) -> dict:
    """codegen 产物体检：910B 标量面的四条特征（凭据口径与 attempts/C、F 一致）。"""
    return {
        "simt": ("asc_vf_call" in src) or ("threadIdx" in src),
        "mte_copy": bool(re.search(r"\basc_copy_[a-z0-9]+_align", src)),
        "expf": "expf" in src,
        "gm_rmw": "read_gm_bypass_dcache" in src,
        "named_null_ubuf": re.findall(
            r"__ubuf__ [A-Za-z_0-9]+ \*(?!buf_dyn_shmem)(\w+) = \(__ubuf__ [A-Za-z_0-9]+ \*\)0", src),
        "lines": src.count("\n"),
    }


def verdict(fragment: str) -> tuple[str, str]:
    """据 compiled_keys/blockers 判：真编过=PASS；有 ascend| blocker=FAIL；都没=PRECOMPILE-ERR。"""
    keys = [k for k in ascend_env.compiled_keys()
            if k.startswith("ascend|") and fragment in k]
    bl = {k: v for k, v in ascend_env.blockers().items() if k.startswith("ascend|") and fragment in k}
    if keys:
        return "PASS", f"key={keys[0]} .o~{_STATS['bytes']}B"
    if bl:
        return "FAIL", f"{next(iter(bl))} :: {next(iter(bl.values()))[:180]}"
    return "PRECOMPILE-ERR", f"未达 tilelang.compile（blockers={list(ascend_env.blockers())[:3]}）"


# --------------------------------------------------------------------------- 驱动
def drive_fwd(shape=(8, 32, 16, 16), variant="ub", gate=False, blocks=None, scalar_gate=False):
    """[FWD] delta-rule 前向：(H, T, DK, DV) 取 F 件默认形状；scalar_gate=True 传每 token 标量门。"""
    h, t, dk, dv = shape
    q, k, v = f32(h, t, dk), f32(h, t, dk), f32(h, t, dv)
    g = f32(h, t) if scalar_gate else f32(h, t, dk)
    beta = torch.rand(h, t, dtype=torch.float32)
    o = torch.empty(h, t, dv, dtype=torch.float32)
    spec = gdn_asc.plan(q, k, v, g, beta, o, "ascend", variant=variant, blocks=blocks)
    if spec is None:
        raise RuntimeError("plan=None（形状判据没过）")
    if not gdn_asc.run(q, k, v, g, beta, o, spec):
        raise RuntimeError("run=False（模具没开成）")
    return spec


def drive_bwd(shape=(8, 32, 16, 16), variant="ub", gate=False, blocks=None):
    """[BWD] 反向伴随件：走接口 `backward`（内部两趟：前向落 Hist + 伴随递推）。"""
    h, t, dk, dv = shape
    q, k, v = f32(h, t, dk), f32(h, t, dk), f32(h, t, dv)
    g, beta = f32(h, t, dk), torch.rand(h, t, dtype=torch.float32)
    dy = f32(h, t, dv)
    return gdn_asc.backward(q, k, v, g, beta, dy, "ascend", variant=variant, blocks=blocks)


def drive_conv(shape=(2048, 512), variant="scalar", gate=False, blocks=None, bias=True):
    """[CONV] 短卷积入口：C=2048/L=512 是 attempts/C 的默认（生产）形状，cores 上限 64。"""
    x = f32(*shape)            # 支持 (C,L) 与 (B,C,L) 两种入参形状
    c = int(shape[-2])
    w = f32(c, gdn_conv_asc.CONV_KERNEL)
    b1 = f32(c) if bias else None
    return gdn_conv_asc.conv_forward(x, w, b1, target="ascend", variant=variant, blocks=blocks)


# --------------------------------------------------------------------------- 用例
CASES = [
    # (标签, 判决片段, 驱动, 是否入门禁, 附加参数, 特征断言)
    ("J-GDN-FWD  ub 档（首推）", "gdn[ascend]", drive_fwd, True, {}, ("expf", "no_simt", "no_mte", "no_null")),
    ("J-GDN-FWD  pure 档（保留，G-F3 待卡证）", "gdn[ascend]", drive_fwd, False, {"variant": "pure"}, ("expf", "no_simt")),
    ("J-GDN-FWD  标量门入参 (H,T)", "gdn[ascend]", drive_fwd, True, {"scalar_gate": True}, ("expf", "no_simt")),
    ("J-GDN-FWD  生产形状 H16 T128 DK64 DV64 cores16", "gdn[ascend]", drive_fwd, False,
     {"shape": (16, 128, 64, 64), "blocks": 16}, ("expf", "no_simt", "no_mte")),
    ("J-GDN-BWD  ub 档（首推，六梯）", "gdnb[ascend]", drive_bwd, True, {}, ("expf", "no_simt", "no_mte", "no_null")),
    ("J-GDN-BWD  pure 档（保留，GM RMW 待卡证）", "gdnb[ascend]", drive_bwd, False, {"variant": "pure"}, ("expf", "no_simt")),
    ("J-GDN-CONV scalar 档（首推）", "gdnconv[ascend]", drive_conv, True, {}, ("expf", "no_simt", "no_mte", "no_null")),
    ("J-GDN-CONV 批入参 (2,C,L)", "gdnconv[ascend]", drive_conv, False, {"shape": (2, 64, 128), "blocks": 8}, ("expf", "no_simt")),
    ("J-GDN-CONV ubstage 档（保留，MTE2/MTE3 面）", "gdnconv[ascend]", drive_conv, False,
     {"shape": (64, 128), "variant": "ubstage", "blocks": 8}, ("expf",)),
]

GATE = "gate"          # 只有首推档入门禁（exit code 只看这些）


def run_case(tag, frag, driver, gated, opts, want):
    ascend_env.reset()
    install_stubs(device_gate=True)
    _STATS.update(src="", bytes=0, compiles=0, launches=0)
    try:
        driver(**opts)
    except _LaunchSkip:
        pass
    except Exception as exc:  # noqa: BLE001
        head = (traceback.format_exception_only(type(exc), exc) or [""])[0].strip()
        v, d = verdict(frag)
        if v != "PASS":
            return v, f"{d} | {head[:160]}", {}
    v, d = verdict(frag)
    f = feats(_STATS["src"])
    if v == "PASS":
        bad = []
        for cond in want:
            if cond == "expf" and not f["expf"]:
                bad.append("没发 expf（decay/silu 的数学面没走到？）")
            if cond == "no_simt" and f["simt"]:
                bad.append("落码里出现了 asc_vf_call/threadIdx（950 载体没清干净）")
            if cond == "no_mte" and f["mte_copy"]:
                bad.append("落码里出现了 asc_copy_*_align（本档应当零 MTE）")
            if cond == "no_null" and f["named_null_ubuf"]:
                bad.append(f"G-C8 命名空句柄：{f['named_null_ubuf']}")
        if bad:
            return "FAIL", f"{d} | 特征不符：" + "；".join(bad), f
    return v, d, f


# --------------------------------------------------------------------------- CPU 语义回归
def cpu_checks() -> list[tuple[str, str, str]]:
    """target=cpu 的语义不许因为 ascend 路径重写而回归（与内核不同源的独立尺子）。"""
    out = []
    torch.manual_seed(13)
    h, t, dk, dv = 2, 24, 8, 8
    q = torch.nn.functional.normalize(torch.randn(h, t, dk), p=2, dim=-1)
    k = torch.nn.functional.normalize(torch.randn(h, t, dk), p=2, dim=-1)
    v, g = torch.randn(h, t, dv), -torch.rand(h, t, dk) * 0.2
    beta = torch.rand(h, t) * 0.9 + 0.05
    dy = torch.randn(h, t, dv)

    def ruler(qq, kk, vv, gg, bb):
        heads, seq, dkw = qq.shape
        dvw = vv.size(-1)
        st = torch.zeros(heads, dkw, dvw)
        o = torch.empty(heads, seq, dvw)
        for i in range(seq):
            st = st * gg[:, i].exp().unsqueeze(-1)
            pred = torch.bmm(st.transpose(1, 2), kk[:, i].unsqueeze(-1)).squeeze(-1)
            upd = (vv[:, i] - pred) * bb[:, i].unsqueeze(-1)
            st = st + torch.bmm(kk[:, i].unsqueeze(-1), upd.unsqueeze(1))
            o[:, i] = torch.bmm(st.transpose(1, 2), qq[:, i].unsqueeze(-1)).squeeze(-1)
        return o

    try:
        got = gdn_asc.forward(q, k, v, g, beta, torch.float32, "cpu")
        torch.testing.assert_close(got, ruler(q, k, v, g, beta), rtol=1e-4, atol=1e-5)
        ran = any("gdn[cpu]" in key for key in ascend_env.compiled_keys())
        out.append(("J-GDN-FWD-CPU", "PASS" if ran else "PASS(未走内核?)", f"keys={ran}"))
    except Exception as exc:  # noqa: BLE001
        out.append(("J-GDN-FWD-CPU", "FAIL", f"{type(exc).__name__}: {str(exc)[:160]}"))

    try:
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v, g, beta)]
        want = torch.autograd.grad(ruler(*leaves), leaves, dy)
        got = gdn_asc.backward(q, k, v, g, beta, dy, "cpu")
        assert len(got) == 5, f"梯度个数 {len(got)} != 5"
        for nm, a, b in zip(gdn_asc.BWD_GRADES, got, want):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5, msg=f"{nm} 与自动微分不一致")
        out.append(("J-GDN-BWD-CPU", "PASS", "五梯 vs autograd ≤1e-4"))
    except Exception as exc:  # noqa: BLE001
        out.append(("J-GDN-BWD-CPU", "FAIL", f"{type(exc).__name__}: {str(exc)[:160]}"))

    try:
        c, length = 16, 64
        x, w, b1 = torch.randn(c, length), torch.randn(c, 4) * 0.3, torch.randn(c) * 0.1
        got = gdn_conv_asc.conv_forward(x, w, b1, target="cpu")
        torch.testing.assert_close(got, gdn_conv_asc._eager(x, w, b1), rtol=1e-4, atol=1e-5)
        ran = any("gdnconv[cpu]" in key for key in ascend_env.compiled_keys())
        out.append(("J-GDN-CONV-CPU", "PASS" if ran else "PASS(未走内核?)", f"keys={ran}"))
    except Exception as exc:  # noqa: BLE001
        out.append(("J-GDN-CONV-CPU", "FAIL", f"{type(exc).__name__}: {str(exc)[:160]}"))
    return out


def static_checks() -> list[tuple[str, str, str]]:
    """接口面与正文面的静态口径检查：partial 撤干净了吗？950 载体清干净了吗？"""
    out = []
    ok = gdn_kernel.BWD_STATUS != "partial" and gdn_asc.BWD_STATUS == gdn_kernel.BWD_STATUS
    out.append(("J-STATIC-BWD-STATUS", "PASS" if ok else "FAIL",
                f"gdn_kernel.BWD_STATUS={gdn_kernel.BWD_STATUS!r} gdn_asc={gdn_asc.BWD_STATUS!r}"))

    import inspect as _i
    bodies = {}
    for fn in (gdn_asc.gdn_asc_impl, gdn_asc.gdn_bwd_asc_impl, gdn_conv_asc.conv_asc_impl):
        bodies[fn.__name__] = _i.getsource(fn)
    dirty = {n: [tok for tok in ("T.SimtVF", "T.Parallel", "T.Pipelined", "T.annotate_buffer_versions")
                 if f"{tok}(" in src or f"with {tok}(" in src]
             for n, src in bodies.items()}
    dirty = {n: d for n, d in dirty.items() if d}
    out.append(("J-STATIC-NO-SIMTVF", "PASS" if not dirty else "FAIL",
                f"ascend 正文残留 950 载体：{dirty}" if dirty else "三份 ascend 正文零 SimtVF/Parallel/Pipelined"))

    has_conv = hasattr(gdn_kernel, "conv_forward") and hasattr(gdn_conv_asc, "conv_forward")
    out.append(("J-STATIC-CONV-ENTRY", "PASS" if has_conv else "FAIL",
                f"gdn_kernel.conv_forward={getattr(gdn_kernel, 'conv_forward', None)}"))
    return out


def print_signatures():
    for label, fn in (("gdn_asc.forward", gdn_asc.forward), ("gdn_asc.backward", gdn_asc.backward),
                      ("gdn_asc.plan", gdn_asc.plan), ("gdn_asc.run", gdn_asc.run),
                      ("gdn_asc.plan_backward", gdn_asc.plan_backward),
                      ("gdn_asc.run_backward", gdn_asc.run_backward),
                      ("gdn_conv_asc.conv_forward", gdn_conv_asc.conv_forward),
                      ("gdn_conv_asc.plan", gdn_conv_asc.plan), ("gdn_conv_asc.run", gdn_conv_asc.run)):
        print(f"  SIG {label}{inspect.signature(fn)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", action="store_true", help="只跑首推档（ub/scalar）——门禁口径")
    ap.add_argument("--cpu", action="store_true", help="追加 target=cpu 语义对拍（需 c 后端可编）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    print_signatures()
    n_fail = 0
    for tag, frag, driver, gated, opts, want in CASES:
        if args.gate and not gated:
            continue
        v, d, f = run_case(tag, frag, driver, gated, opts, want)
        if v != "PASS":
            n_fail += 1
        extra = "" if args.quiet or not f else (
            f" [{f['lines']}行 expf={f['expf']} simt={f['simt']} mte={f['mte_copy']} "
            f"gm_rmw={f['gm_rmw']} null_ubuf={f['named_null_ubuf']}]")
        print(f"{tag:52s} ASC-COMPILE-{v} {d}{extra}")

    for tag, v, d in static_checks():
        if v != "PASS":
            n_fail += 1
        print(f"{tag:52s} {v} {d}")

    if args.cpu:
        ascend_env.reset()
        for tag, v, d in cpu_checks():
            if v == "FAIL":
                n_fail += 1
            print(f"{tag:52s} {v} {d}")

    print(f"—— J 汇总 FAIL={n_fail} ——" if n_fail else f"—— J 汇总 全 PASS（FAIL=0）——")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
