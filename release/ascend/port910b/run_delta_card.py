#!/usr/bin/env python3
"""run_delta_card.py — P1-3 GDN delta rule 前向的**真机执行**收数件（910B）。

F 波已在 CPU 证真（fp32 递推 golden rel=5.76e-07 / adjoint-vs-FD 7.28e-10），但容器无
davinci 设备从未**执行**过 kernel——本件在卡上编译+跑+对拍 fwd_exact(float64 精确递推)。

C4（P1-1w）扩 `--bwd`：把训练步里那个只"执行过"、从未逐元素判过的 `gdnb` 模具（产品 home
`ascend/kernels/gdn_asc.py` 的反向）拉出来逐元素对拍——六项梯度（dq/dk/dv/dbeta/dg/dh0，
按 F 波清单）vs **同输入** fp64 golden（fwd_exact + bwd_adjoint）；前向 o/s 复用现腿。
判决行：
  NUM-delta_bwd PASS dq=<v> dk=<v> dv=<v> dbeta=<v> dg=<v> dh0=<v> adjoint=<v> rel_max=<v>
  NUM-delta_bwd FAIL first=<item> rel=<v> absdiff=<v> thresh=<v>
（六项逐项都打，不只打 max；FAIL 指认首个超阈项。假绿守卫：必须见 `gdnb[` 键，否则判 FAIL。）

纪律：不用 out_idx 魔法（本地无法 run-verify，猜错即烧卡窗）。显式分配全部 9 张量，
kernel 就地写 HIST/SOUT/O，逐一比对。镜像 run_numerics 里已验证的 num_gdn 喂参形态。

判据行：NUM-delta_fwd PASS rel_o=<x> rel_s=<y>   /   NUM-delta_fwd FAIL <reason>
"""
import os, sys, argparse
os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE = os.path.dirname(os.path.dirname(HERE))            # .../release（`import ascend.kernels`）
sys.path.insert(0, os.path.join(HERE, "attempts", "F"))
if RELEASE not in sys.path:
    sys.path.insert(0, RELEASE)
import numpy as np
import tilelang
import tilelang.language as T  # noqa: F401
# torch / torch_npu 延迟到 run 半程（device-less 容器可验编译，与 run_numerics --no-run 同纪律）

AP = argparse.ArgumentParser()
AP.add_argument("--compile-only", "--no-run", action="store_true", dest="compile_only",
                help="只编译不跑（device-less 判决）")
AP.add_argument("--bwd", action="store_true",
                help="附加 GDN delta 反向判决腿（产品 home 六项梯度 vs fp64 golden；前向 o/s 复用现腿）")
AP.add_argument("--target", choices=("ascend", "cpu"), default="ascend",
                help="ascend=真机/容器编译（默认）；cpu=cpu-alias 宿主全量自证")
AP.add_argument("--variant", choices=("ub", "pure"), default=None,
                help="gdn 档位；缺省 ascend=ub / cpu=pure（cpu 后端只允许一处 dynamic shared，ub 编不过）")
AP.add_argument("--neg-perturb", default="", metavar="ITEM",
                help="负例：把该项**实测**梯度 ×1.01，用于证 FAIL 且指认正确项（dq/dk/dv/dbeta/dg/dh0）")
ARGS = AP.parse_args()

H, TT, DK, DV, CORES = 2, 8, 8, 8, 1
DT = "float32"
TOL_F32 = 2e-4     # 沿用 F 波 TOL_F32（不另造）
TOL_ADJ = 1e-8     # fp64 伴随 vs 有限差分（沿用 F 波 TOL_ADJ）
BWD_ITEMS = ("dq", "dk", "dv", "dbeta", "dg", "dh0")


def _tgt():
    return "c" if ARGS.target == "cpu" else "ascend"


def _variant():
    if ARGS.variant:
        return ARGS.variant
    return "pure" if ARGS.target == "cpu" else "ub"


def relerr(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    d = np.linalg.norm(b) + 1e-30
    return np.linalg.norm(a - b) / d


def _fwd_inputs():
    """现腿输入口径（seed=11）：Q,Kt,V,BETA,G,H0 挨个抽，逐位不变。"""
    rng = np.random.default_rng(11)
    Q = rng.standard_normal((H, TT, DK)).astype(DT)
    Kt = (rng.standard_normal((H, TT, DK)) * 0.3).astype(DT)
    V = rng.standard_normal((H, TT, DV)).astype(DT)
    BETA = (rng.random((H, TT)) * 0.5 + 0.25).astype(DT)
    G = (-rng.random((H, TT))).astype(DT)          # decay log 门 g<=0
    H0 = (rng.standard_normal((H, DV, DK)) * 0.3).astype(DT)
    return Q, Kt, V, BETA, G, H0


def _bwd_inputs():
    """反向腿输入：前 5 量与现腿逐位同（seed=11 同抽序），第 6 抽作 dO（N(0,1)）。

    初状态不在此列——产品反向内部恒以零初态重算（`fwd_workspaces` 的 H0 恒零），
    参考侧必须同输入，故 golden 也用零初态（见 `_bwd_leg`）。
    """
    rng = np.random.default_rng(11)
    Q = rng.standard_normal((H, TT, DK)).astype(DT)
    Kt = (rng.standard_normal((H, TT, DK)) * 0.3).astype(DT)
    V = rng.standard_normal((H, TT, DV)).astype(DT)
    BETA = (rng.random((H, TT)) * 0.5 + 0.25).astype(DT)
    G = (-rng.random((H, TT))).astype(DT)
    DO = rng.standard_normal((H, TT, DV)).astype(DT)
    return Q, Kt, V, BETA, G, DO


def _install_cpu_alias():
    """cpu-alias：把 `tilelang.ascend.language` 指到 cpu 方言，给 cpu 方言缺的 frame 挂 no-op。

    必须在 import 任何 `tilelang.ascend.language` 使用方（F 件 / gdn_asc）之前调用。
    （run_numerics 的 cpu-alias 手法同源。）
    """
    import contextlib
    import types
    import tilelang.cpu.language as tcpu
    proxy = types.ModuleType("tilelang.ascend.language")
    for n in dir(tcpu):
        if not n.startswith("__"):
            setattr(proxy, n, getattr(tcpu, n))

    @contextlib.contextmanager
    def _noop(*a, **k):
        yield None

    for n in ("Vector", "SimtVF", "Cube", "SimdVF"):
        if not hasattr(proxy, n):
            setattr(proxy, n, _noop)
    sys.modules["tilelang.ascend.language"] = proxy
    setattr(tilelang.ascend, "language", proxy)
    return proxy


def _force_kernel_env():
    """让产品 home 走**内核路**（而非 torch 回退），并按 target 配好编译参数。

    * `--target cpu`：设备是 cpu、target 名却必须是 "ascend"（产品件内部口径），`active_backend`
      本会判 torch_eager（设备与 target 不搭）⇒ 强制 tilelang；编译参数换成 c 后端三件套。
    * `--no-run`（device-less 容器）：无 npu ⇒ 探测会判不可用 ⇒ 同样强制 tilelang；但**不动**
      `compile_kwargs`（仍按真实 ascend 编，才产 `gdnb[ascend]` 键）。

    产品件是禁改件，这里只做运行期目录替换（不落盘写产品源码）。
    """
    from ascend.kernels import ascend_env
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG
    if ARGS.target == "cpu":
        ascend_env.compile_kwargs = lambda t: {"target": "c", "target_host": "c",
                                               "execution_backend": "cython"}
    ascend_env.reset()
    return ascend_env


def _wrap_no_launch():
    """no-run：包住 `tilelang.compile`，交回一个"只登记键、不发射"的壳。

    为什么用 no-op 而非抛异常跳过：`run_backward` 里**前向发射在反向模具编译之前**，若前向
    那一下抛异常，反向的 `gdnb` 键就登记不上——而要的正是那个键。
    """
    real = tilelang.compile

    class _NoLaunch:
        def __init__(self, ker):
            self._ker = ker

        def __call__(self, *a, **k):
            return None

        def get_kernel_source(self):
            return self._ker.get_kernel_source()

    def _wrapped(fn, *a, **k):
        return _NoLaunch(real(fn, *a, **k))

    tilelang.compile = _wrapped


def _fwd_run(k, gd):
    """跑现腿前向 + 打 fwd 判决行；返回 (ok, ins)。ins 供 --bwd 复用同输入。"""
    import torch
    if ARGS.target == "ascend":
        import torch_npu  # noqa: F401  # device backend（卡上才有）
        dev = lambda x: torch.from_numpy(np.ascontiguousarray(x)).npu()  # noqa: E731
    else:
        dev = lambda x: torch.from_numpy(np.ascontiguousarray(x))  # noqa: E731
    Q, Kt, V, BETA, G, H0 = _fwd_inputs()
    HIST = np.zeros((H, TT, DV, DK), DT)
    # pure 档状态就地读写 SOUT（约定宿主预置 SOUT=H0）；ub 档 SOUT 只作出口
    SOUT = H0.copy() if _variant() == "pure" else np.zeros((H, DV, DK), DT)
    O = np.zeros((H, TT, DV), DT)

    inps = [dev(x) for x in (Q, Kt, V, BETA, G, H0)]
    outs = [dev(x) for x in (HIST, SOUT, O)]
    # kernel 签名序 = Q,K,V,BETA,G,H0,HIST,SOUT,O
    k(*inps, *outs)
    if ARGS.target == "ascend":
        torch.npu.synchronize()

    got_o = outs[2].cpu().numpy().astype(np.float64)
    got_s = outs[1].cpu().numpy().astype(np.float64)
    ref_o, _ref_h, ref_sf = gd.fwd_exact(H, TT, DK, DV,
                                         Q, Kt, V, BETA, G, H0)
    r_o = relerr(got_o, ref_o)
    r_s = relerr(got_s, ref_sf)
    # F CPU 口径 (a) 仅 expf，期望 rel_o 1e-6 级；阈值给 bf16/漂移余量
    if r_o < 2e-4 and r_s < 2e-4:
        print(f"NUM-delta_fwd PASS rel_o={r_o:.2e} rel_s={r_s:.2e}")
        return True, (Q, Kt, V, BETA, G, H0)
    print(f"NUM-delta_fwd FAIL rel_o={r_o:.3e} rel_s={r_s:.3e} (阈值 2e-4)")
    return False, (Q, Kt, V, BETA, G, H0)


def _adjoint_fd_check(gd):
    """参考侧自证：fp64 伴随 vs 中心差分（与推导过程**独立**的锚）；返回 worst rel。

    这就是判决行里的 `adjoint=<v>`——它证的不是 kernel，而是"我们拿来当尺子的 fp64 伴随
    本身没错"，判据行带上它，任何人看一行就知道尺子可信（F 波口径沿用）。
    """
    fdH, fdT, fdK, fdD = 2, 6, 4, 4
    Qf, Kf, Vf, Bf, Gf, H0f, wf = gd.sample(fdH, fdT, fdK, fdD, 8)
    _o, HISTf, _s = gd.fwd_exact(fdH, fdT, fdK, fdD, Qf, Kf, Vf, Bf, Gf, H0f)
    adj = gd.bwd_adjoint(fdH, fdT, fdK, fdD, Qf, Kf, Vf, Bf, Gf, HISTf, wf,
                         order=_variant())
    fdg = gd.gradcheck_fd((fdH, fdT, fdK, fdD, Qf, Kf, Vf, Bf, Gf, H0f, wf))
    return max(gd.rel(a, fdg[n]) for n, a in
               zip(("Q", "K", "V", "BETA", "G", "H0"), adj))


def _bwd_leg(gd):
    """产品 home 反向腿：`gdn_asc.backward` 真跑（或 no-run 只编模具）→ 六项梯度 vs fp64 golden。

    返回 0/1。**假绿守卫**：必须见 `gdnb[` 模具键，否则判 FAIL——torch 回退路交回的 dh0 恒零、
    数值看着也"能过"，不守就会把回退当成内核过了。
    """
    from ascend.kernels import ascend_env, gdn_asc
    import torch
    if ARGS.neg_perturb and ARGS.neg_perturb not in BWD_ITEMS:
        print(f"NUM-delta_bwd FAIL neg-perturb 项名非法: {ARGS.neg_perturb!r}"
              f"（限 {'/'.join(BWD_ITEMS)}）")
        return 1
    Q, Kt, V, BETA, G0, DO = _bwd_inputs()
    if ARGS.target == "ascend" and not ARGS.compile_only:
        import torch_npu  # noqa: F401
        to_dev = lambda x: torch.from_numpy(np.ascontiguousarray(x)).npu()  # noqa: E731
    else:
        to_dev = lambda x: torch.from_numpy(np.ascontiguousarray(x))  # noqa: E731
    tq, tk, tv, tg, tb, tdo = (to_dev(x) for x in (Q, Kt, V, G0, BETA, DO))

    # 档位显式透传（cpu 后端只认 pure；ub 档 7 个 dynamic shared 编不出）
    grads = gdn_asc.backward(tq, tk, tv, tg, tb, tdo, target="ascend",
                             return_dh0=True, variant=_variant())
    keys = list(ascend_env.compiled_keys())
    bkeys = [k for k in keys if "gdnb[" in k]
    blk = {k: v for k, v in ascend_env.blockers().items() if "gdn_backward" in k}

    if ARGS.compile_only:
        tag = bkeys[0] if bkeys else "MISSING"
        if bkeys and not blk:
            print(f"NUM-delta_bwd PASS rel=n/a (compile-only) key={tag}")
            return 0
        print(f"NUM-delta_bwd FAIL compile: key={tag} keys={keys} blockers={blk or 'none'}")
        return 1

    if not bkeys or blk:
        print(f"NUM-delta_bwd FAIL 假绿: 未见 gdnb 键 keys={keys} blockers={blk or 'none'}")
        return 1

    # 反向交回序 = (dq, dk, dv, dg, dbeta[, dh0])（见 gdn_asc.BWD_GRADES）
    dq, dk, dv, dg, db, dh0 = (g.detach().cpu().numpy().astype(np.float64) for g in grads)
    # fp64 golden（**同输入**；初态零——产品内部恒以零初态重算）
    _O64, HIST64, _Sf = gd.fwd_exact(H, TT, DK, DV, Q, Kt, V, BETA, G0, np.zeros((H, DV, DK)))
    dqa, dka, dva, dba, dga, dh0a = gd.bwd_adjoint(H, TT, DK, DV, Q, Kt, V, BETA, G0,
                                                   HIST64, DO, order=_variant())
    # g 入参是每 token 标量门，产品 `_gate3` 摊成每键通道 ⇒ dg 沿通道求和即回到每 token 标量
    meas = {"dq": dq, "dk": dk, "dv": dv, "dbeta": db, "dg": dg.sum(-1), "dh0": dh0}
    ref = {"dq": dqa, "dk": dka, "dv": dva, "dbeta": dba, "dg": dga, "dh0": dh0a}
    if ARGS.neg_perturb in meas:
        meas[ARGS.neg_perturb] = meas[ARGS.neg_perturb] * 1.01

    vals, diff = {}, {}
    for it in BWD_ITEMS:
        vals[it] = relerr(meas[it], ref[it])
        diff[it] = float(np.max(np.abs(np.asarray(meas[it]) - np.asarray(ref[it]))))
    vals["adjoint"] = _adjoint_fd_check(gd)
    diff["adjoint"] = None

    order = BWD_ITEMS + ("adjoint",)
    thr = {it: TOL_F32 for it in BWD_ITEMS}
    thr["adjoint"] = TOL_ADJ
    first = next((it for it in order if vals[it] > thr[it]), None)
    if first is None:
        rel_max = max(vals[it] for it in BWD_ITEMS)
        body = " ".join(f"{it}={vals[it]:.2e}" for it in order)
        print(f"NUM-delta_bwd PASS {body} rel_max={rel_max:.2e}")
        return 0
    d = "n/a" if diff[first] is None else f"{diff[first]:.3e}"
    print(f"NUM-delta_bwd FAIL first={first} rel={vals[first]:.3e} "
          f"absdiff={d} thresh={thr[first]:.0e}")
    return 1


def main():
    if ARGS.target == "cpu":
        _install_cpu_alias()
    if ARGS.bwd and (ARGS.target == "cpu" or ARGS.compile_only):
        _force_kernel_env()
        if ARGS.compile_only:
            _wrap_no_launch()

    # ───────────────────────── 前向腿（现腿；--bwd 复用其 o/s）─────────────────────────
    try:
        import f_gdn_delta_fwd_910b as mod
        import f_gdn_golden as gd
        fn = mod.gdn_delta_rule_fwd(H=H, T_LEN=TT, DK=DK, DV=DV, cores=CORES,
                                    dtype=DT, variant=_variant(), emit_hist=True)
        k = tilelang.compile(fn, target=_tgt())  # 显式传全部张量，不用 out_idx
        if ARGS.compile_only:
            print("NUM-delta_fwd PASS rel=n/a (compile-only)")
            if not ARGS.bwd:
                return 0
    except Exception as e:  # compile 失败
        print(f"NUM-delta_fwd FAIL compile: {type(e).__name__}: {str(e)[:200]}")
        return 1

    rc = 0
    if not ARGS.compile_only:
        try:
            ok_f, _ins = _fwd_run(k, gd)
            rc = 0 if ok_f else 1
        except Exception as e:
            print(f"NUM-delta_fwd FAIL run: {type(e).__name__}: {str(e)[:200]}")
            return 1

    if ARGS.bwd:
        try:
            rc_b = _bwd_leg(gd)
        except Exception as e:
            print(f"NUM-delta_bwd FAIL run: {type(e).__name__}: {str(e)[:200]}")
            return 1
        rc = max(rc, rc_b)
    return rc


if __name__ == "__main__":
    sys.exit(main())
