#!/usr/bin/env python3
"""reconcile/S_cpu_oracle.py — P1-1f 双方言 oracle：**同一份标量正文**，cpu 轨真跑判数值、ascend 轨判编译。

【为什么这么摆】D-cube1 改道后，`gemm_asc.gemm_scalar_body` / `gemm_bwd_dw_asc.dw_scalar_body` 是
只吃两方言公共子集（`T.serial` + `T.copy`(GM→UB) + 标量 load/store + 整数索引算术 + 常量网格）的
单一正文生成器，语言模块由 `ascend_env.dialect(target)` 注入。于是同一份算式可以：
  * **cpu 轨**：target=cpu 真编真跑，与 torch 闭式参考对拍 ⇒ cube 数值第一次能在本地闭环（本波最大
    价值——此前 cube 形态的数值只能等卡窗）；
  * **ascend 轨**：target=ascend 只做编译判决（容器有 bisheng、无设备），数值随下窗真机。
两条轨用的是**逐字同一份正文**，所以 cpu 轨的数值结论对 ascend 轨有直接解释力（差异只在存储层落点
`_alloc_tile`，那是 scope 归属、不是算式）。

【判据】
  * cpu 轨：`rel = max|got-ref| / max|ref| < 2e-5`（fp32 对 fp32，只容忍归约顺序差）；形状覆盖
    方阵/非方/relu-none/尾块（m 不整除 bm）/多块组合/转置方向判别（gemm 的 `Wᵀ` 方向、dW 的
    `dYᵀ@A` 朝向）；并与 day-1 `gemm_cpu_impl`/`dw_cpu_impl`（T.gemm 版尺子件）交叉对拍。
  * ascend 轨：`tilelang.compile(..., target="ascend", out_idx=[])` 出产物 ⇒ `S-GEMM-ASC-COMPILE-PASS`
    / `S-DW-ASC-COMPILE-PASS`；并静态扫生成码，**含 SimtVF/Cube/mad/L12L0/L0C 任一即判 FAIL**
    （改道的目的就是零 cube 依赖，混进去说明正文写歪了）。
  * 附赠（host 也能跑）：`--trace` 只把 ascend 方言的正文追到 PrimFunc（不实编），用来在本地提前
    抓 DSL 级错误——**这不是编译判决**，只登记 TRACE-OK/TRACE-FAIL。

【跑法】
  # 本地 host（cpu 数值轨 + 追溯自证）
  cd release && .venv/bin/python ascend/kernels/reconcile/S_cpu_oracle.py --cpu --trace
  # 容器（ascend 编译轨；先同步 trunk compat）
  docker cp /tilelang/src/tl_templates/ascend/port910b_compat.h \\
    cann910b-h:/usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend/
  docker exec cann910b-h bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \\
    export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \\
    cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 900 python3 \\
      ascend/kernels/reconcile/S_cpu_oracle.py --ascend'
"""
import os
import sys

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))              # .../release/ascend/kernels/reconcile
Kernels_DIR = os.path.dirname(HERE)                             # .../release/ascend/kernels
RELEASE_ROOT = os.path.dirname(os.path.dirname(Kernels_DIR))    # .../release
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

import torch           # noqa: E402
import tilelang        # noqa: E402
from ascend.kernels import ascend_env, gemm_asc, gemm_bwd_dw_asc  # noqa: E402

#: fp32 对 fp32 的相对误差门（与 tasks.md P1-1f 判据同值）
REL_TOL = 2e-5
#: 方向判别用：错方向必须与结果差出这个量级以上，否则"对拍通过"可能只是碰巧对称
WRONG_DIR_MIN = 1e-2

# (m, n, k, act, 说明)——m 走动态维，故意含不整除 BLOCK_M(=16) 的尾块
GEMM_CASES = [
    (64, 64, 64, "none", "方阵基线 bn=bk=64（单列块×单 K 块）"),
    (64, 64, 64, "relu", "同形换 relu 出口分支"),
    (32, 128, 64, "relu", "非方 n=2k ⇒ nb=2 多列块并行"),
    (8, 64, 64, "none", "尾块 m<bm=16 ⇒ 夹址填装 + 行守卫全生效"),
    (33, 64, 32, "relu", "m 不整除 bm 且 bk=32 多 K 块"),
    (16, 96, 48, "none", "n=96⇒bn=32(nb=3)、k=48⇒bk=16(nk=3) 多块组合"),
    (1, 64, 64, "relu", "批大小 1：单行也要走完整块口径"),
]

# (m, n, k, 说明)——GW 出口 (n,k)；n≠k 的形状本身就是转置朝向的判别器
DW_CASES = [
    (64, 64, 64, "方阵基线（另做 dYᵀ@A vs Aᵀ@dY 方向判别）"),
    (32, 64, 128, "非方 n≠k ⇒ 朝向唯一"),
    (8, 64, 32, "尾块 m<tc=16"),
    (33, 128, 64, "m 不整除 tc 且出口块数 >1"),
    (16, 96, 48, "bn=32/bk=16 ⇒ tiles=3×3=9 > NUM_BLOCKS=8 ⇒ iters=2 摊开"),
    (129, 64, 64, "m>tc 多轮累加（129=8×16+1）"),
]


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    """相对误差 = max|got-ref| / max(|ref|,eps)：分子分母都取最坏，不给"平均一下就行"留口子。"""
    num = (got - ref).abs().max().item()
    den = max(ref.abs().max().item(), 1e-12)
    return num / den


def _gemm_ref(A, W, bias, act):
    """闭式尺子：C = act(A @ Wᵀ + bias)，fp32（W 按 (N,K) 存 ⇒ 必须右翻）。"""
    acc = A.to(torch.float32) @ W.to(torch.float32).T + bias.to(torch.float32)
    return acc.clamp_min(0.0) if act == "relu" else acc


def _dw_ref(dY, A):
    """闭式尺子：GW = dYᵀ @ A，形状 (N,K)（与 gemm_bwd_dw_kernel.eager_backward 同式）。"""
    return dY.to(torch.float32).T @ A.to(torch.float32)


def _kwargs_gemm(n, k, act):
    """按 plan 同一套阶梯算出块宽（oracle 直接用生成器，不走 plan，故自己复现整除口径）。"""
    return {"n": n, "k": k, "bm": gemm_asc.BLOCK_M, "bn": gemm_asc._largest_block(n),
            "bk": gemm_asc._largest_block(k), "act_mode": gemm_asc.ACT_MODE[act]}


def _kwargs_dw(n, k):
    return {"n": n, "k": k, "tc": gemm_bwd_dw_asc.BLOCK_TC,
            "bn": gemm_bwd_dw_asc._largest_block(n), "bk": gemm_bwd_dw_asc._largest_block(k)}


# --------------------------------------------------------------------------- cpu 轨（真编真跑）
def _cpu_compile(body, args, kwargs):
    """cpu 方言：真编（target=c 三件套），返回可发射的 kernel。"""
    prog = body(*args, **kwargs)
    return tilelang.compile(prog, out_idx=[], **ascend_env.compile_kwargs("cpu"))


def _drive_gemm_cpu() -> int:
    T = ascend_env.dialect("cpu")
    torch.manual_seed(0)
    ok = True
    kern_cache = {}
    for m, n, k, act, note in GEMM_CASES:
        kw = _kwargs_gemm(n, k, act)
        if kw["bn"] is None or kw["bk"] is None:
            print(f"[gemm m{m} n{n} k{k} {act}] SKIP：形状不在 BLOCK_LADDER 表达窗内")
            continue
        A = torch.randn(m, k, dtype=torch.float32)
        W = torch.randn(n, k, dtype=torch.float32)
        bias = torch.randn(n, dtype=torch.float32)
        C = torch.zeros(m, n, dtype=torch.float32)
        key = (n, k, act)
        if key not in kern_cache:                          # 同一份产物服务多批（m 是动态符号）
            kern_cache[key] = _cpu_compile(
                lambda *a, **w: gemm_asc.gemm_scalar_body(T, *a, **w), (A, W, bias, C), kw)
        kern = kern_cache[key]
        C.zero_()
        kern(A, W, bias, C)
        ref = _gemm_ref(A, W, bias, act)
        rel = _rel(C, ref)
        line = f"[gemm m{m} n{n} k{k} {act}] rel={rel:.3e} {'PASS' if rel < REL_TOL else 'FAIL'} — {note}"
        # 转置方向判别：A@W（不翻）必须差出去，否则"方阵通过"可能只是碰巧
        if m == n == k:
            wrong = _rel(C, A @ W + bias)
            discrim = f" 方向判别 wrong(A@W,不翻)={wrong:.3e} (须>{WRONG_DIR_MIN:.0e})"
            ok = ok and wrong > WRONG_DIR_MIN
        else:
            discrim = ""
        if act == "relu":
            neg = int((C < 0).sum().item())
            line += f" 负数出口={neg}(须 0)"
            ok = ok and neg == 0
        print(line + discrim)
        ok = ok and rel < REL_TOL
    # 与 day-1 尺子件（T.gemm 版 gemm_cpu_impl）交叉对拍：两口径必须同解
    m, n, k = 33, 64, 32
    A = torch.randn(m, k, dtype=torch.float32)
    W = torch.randn(n, k, dtype=torch.float32)
    bias = torch.randn(n, dtype=torch.float32)
    C_new = torch.zeros(m, n, dtype=torch.float32)
    C_day1 = torch.zeros(m, n, dtype=torch.float32)
    kw = _kwargs_gemm(n, k, "relu")
    _cpu_compile(lambda *a, **w: gemm_asc.gemm_scalar_body(T, *a, **w),
                 (A, W, bias, C_new), kw)(A, W, bias, C_new)
    _cpu_compile(gemm_asc.gemm_cpu_impl, (A, W, bias, C_day1), kw)(A, W, bias, C_day1)
    rel_cross = _rel(C_new, C_day1)
    print(f"[gemm 交叉 day-1 gemm_cpu_impl m{m} n{n} k{k}] rel={rel_cross:.3e} "
          f"{'PASS' if rel_cross < REL_TOL else 'FAIL'}")
    ok = ok and rel_cross < REL_TOL
    print(f"compiled_products={len(kern_cache)}（同形状多批共用，m 不在缓存键里）")
    print("S-GEMM-CPU-ORACLE-PASS" if ok else "S-GEMM-CPU-ORACLE-FAIL")
    return 0 if ok else 1


def _drive_dw_cpu() -> int:
    T = ascend_env.dialect("cpu")
    torch.manual_seed(1)
    ok = True
    kern_cache = {}
    for m, n, k, note in DW_CASES:
        kw = _kwargs_dw(n, k)
        if kw["bn"] is None or kw["bk"] is None:
            print(f"[dw m{m} n{n} k{k}] SKIP：形状不在 BLOCK_LADDER 表达窗内")
            continue
        dY = torch.randn(m, n, dtype=torch.float32)
        A = torch.randn(m, k, dtype=torch.float32)
        GW = torch.zeros(n, k, dtype=torch.float32)
        key = (n, k)
        if key not in kern_cache:
            kern_cache[key] = _cpu_compile(
                lambda *a, **w: gemm_bwd_dw_asc.dw_scalar_body(T, *a, **w), (dY, A, GW), kw)
        kern = kern_cache[key]
        GW.zero_()
        kern(dY, A, GW)
        ref = _dw_ref(dY, A)
        rel = _rel(GW, ref)
        shape_ok = tuple(GW.shape) == (n, k)
        line = (f"[dw m{m} n{n} k{k}] rel={rel:.3e} shape_ok={shape_ok} "
                f"{'PASS' if rel < REL_TOL and shape_ok else 'FAIL'} — {note}")
        discrim = ""
        if n == k:      # 方阵：错方向（Aᵀ@dY）也是 (n,k)，必须靠数值差出去
            wrong = _rel(GW, A.T @ dY)
            discrim = f" 方向判别 wrong(Aᵀ@dY)={wrong:.3e} (须>{WRONG_DIR_MIN:.0e})"
            ok = ok and wrong > WRONG_DIR_MIN
        print(line + discrim)
        ok = ok and rel < REL_TOL and shape_ok
    # 与 day-1 尺子件（T.gemm 版 dw_cpu_impl，transpose_A=True）交叉对拍
    m, n, k = 33, 128, 64
    dY = torch.randn(m, n, dtype=torch.float32)
    A = torch.randn(m, k, dtype=torch.float32)
    GW_new = torch.zeros(n, k, dtype=torch.float32)
    GW_day1 = torch.zeros(n, k, dtype=torch.float32)
    kw = _kwargs_dw(n, k)
    _cpu_compile(lambda *a, **w: gemm_bwd_dw_asc.dw_scalar_body(T, *a, **w),
                 (dY, A, GW_new), kw)(dY, A, GW_new)
    _cpu_compile(gemm_bwd_dw_asc.dw_cpu_impl, (dY, A, GW_day1), kw)(dY, A, GW_day1)
    rel_cross = _rel(GW_new, GW_day1)
    print(f"[dw 交叉 day-1 dw_cpu_impl m{m} n{n} k{k}] rel={rel_cross:.3e} "
          f"{'PASS' if rel_cross < REL_TOL else 'FAIL'}")
    ok = ok and rel_cross < REL_TOL
    # 多批复用（同产物换批）：编译一次，连跑 m=1/8/33/64
    m0, n0, k0 = 64, 64, 64
    kw = _kwargs_dw(n0, k0)
    dY = torch.randn(m0, n0, dtype=torch.float32)
    A = torch.randn(m0, k0, dtype=torch.float32)
    GW = torch.zeros(n0, k0, dtype=torch.float32)
    kern = _cpu_compile(lambda *a, **w: gemm_bwd_dw_asc.dw_scalar_body(T, *a, **w), (dY, A, GW), kw)
    rels = []
    for mm in (1, 8, 16, 17, 33, 64):
        dYm = torch.randn(mm, n0, dtype=torch.float32)
        Am = torch.randn(mm, k0, dtype=torch.float32)
        GWm = torch.zeros(n0, k0, dtype=torch.float32)
        kern(dYm, Am, GWm)
        rels.append(_rel(GWm, _dw_ref(dYm, Am)))
    reuse_ok = max(rels) < REL_TOL
    print(f"[dw 同产物多批 m=1/8/16/17/33/33→64] rels={[f'{r:.2e}' for r in rels]} "
          f"{'PASS' if reuse_ok else 'FAIL'}")
    ok = ok and reuse_ok
    print("S-DW-CPU-ORACLE-PASS" if ok else "S-DW-CPU-ORACLE-FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------- ascend 轨（只判编译）
#: 生成码里出现即判 FAIL 的构造——改道的目的就是"零 cube 依赖、零 SIMT 载体"
BANNED = ("SimtVF", "SimdVF", "T.Parallel", "Cube", "mad", "L12L0", "L0C", "asc_fill_l1",
          "set_l1_2d", "transpose_in", "hf32")


def _drive_ascend() -> int:
    T = ascend_env.dialect("ascend")
    torch.manual_seed(2)
    rc_gemm = rc_dw = 0
    sizes = {"gemm": 0, "dw": 0}
    banned_hit = {"gemm": set(), "dw": set()}

    for m, n, k, act, note in GEMM_CASES:
        kw = _kwargs_gemm(n, k, act)
        if kw["bn"] is None or kw["bk"] is None:
            continue
        A = torch.randn(m, k, dtype=torch.float32)
        W = torch.randn(n, k, dtype=torch.float32)
        bias = torch.randn(n, dtype=torch.float32)
        C = torch.zeros(m, n, dtype=torch.float32)
        try:
            kern = tilelang.compile(gemm_asc.gemm_scalar_body(T, A, W, bias, C, **kw),
                                    out_idx=[], **ascend_env.compile_kwargs("ascend"))
            src = str(kern.get_kernel_source()) if hasattr(kern, "get_kernel_source") else ""
            sizes["gemm"] = max(sizes["gemm"], len(src))
            hit = {b for b in BANNED if b in src}
            banned_hit["gemm"] |= hit
            print(f"[asc gemm m{m} n{n} k{k} {act}] COMPILE-PASS src~{len(src)}B "
                  f"banned_in_src={sorted(hit) if hit else '[]'} — {note}")
        except Exception as exc:  # noqa: BLE001  编译失败必须如实报，不吞
            rc_gemm = 1
            msg = str(exc)
            i = msg.find("error:")
            print(f"[asc gemm m{m} n{n} k{k} {act}] COMPILE-FAIL "
                  f"{(msg[i:i + 220] if i >= 0 else msg.splitlines()[0][:220])}")
    for m, n, k, note in DW_CASES:
        kw = _kwargs_dw(n, k)
        if kw["bn"] is None or kw["bk"] is None:
            continue
        dY = torch.randn(m, n, dtype=torch.float32)
        A = torch.randn(m, k, dtype=torch.float32)
        GW = torch.zeros(n, k, dtype=torch.float32)
        try:
            kern = tilelang.compile(gemm_bwd_dw_asc.dw_scalar_body(T, dY, A, GW, **kw),
                                    out_idx=[], **ascend_env.compile_kwargs("ascend"))
            src = str(kern.get_kernel_source()) if hasattr(kern, "get_kernel_source") else ""
            sizes["dw"] = max(sizes["dw"], len(src))
            hit = {b for b in BANNED if b in src}
            banned_hit["dw"] |= hit
            print(f"[asc dw m{m} n{n} k{k}] COMPILE-PASS src~{len(src)}B "
                  f"banned_in_src={sorted(hit) if hit else '[]'} — {note}")
        except Exception as exc:  # noqa: BLE001
            rc_dw = 1
            msg = str(exc)
            i = msg.find("error:")
            print(f"[asc dw m{m} n{n} k{k}] COMPILE-FAIL "
                  f"{(msg[i:i + 220] if i >= 0 else msg.splitlines()[0][:220])}")

    if rc_gemm == 0 and not banned_hit["gemm"]:
        print(f"S-GEMM-ASC-COMPILE-PASS（标量体 {len(GEMM_CASES)} 形全编出，.src~{sizes['gemm']}B，"
              f"零 cube/零 SIMT 构造）")
    else:
        print(f"S-GEMM-ASC-COMPILE-FAIL rc={rc_gemm} banned={sorted(banned_hit['gemm'])}")
    if rc_dw == 0 and not banned_hit["dw"]:
        print(f"S-DW-ASC-COMPILE-PASS（标量体 {len(DW_CASES)} 形全编出，.src~{sizes['dw']}B，"
              f"零 cube/零 SIMT 构造，§12 依赖面=0）")
    else:
        print(f"S-DW-ASC-COMPILE-FAIL rc={rc_dw} banned={sorted(banned_hit['dw'])}")
    return 0 if (rc_gemm == 0 and rc_dw == 0 and not banned_hit["gemm"]
                 and not banned_hit["dw"]) else 1


# --------------------------------------------------------------------------- 接口路径（plan/run 真链）
def _drive_iface() -> int:
    """走真实接口 plan→run→*_asc_impl 的 ascend 编译判决（真编后换 no-op 发射替身，隔离发射）。

    为什么要单开这一段：`reconcile/verify_dw.py` 的桩是"真编后原样返回 kernel"，接口路径一旦编
    成功就会真去 launch cpu 张量 ⇒ `RuntimeError: kernel dw_impl input dY device_type mismatch,
    expected ext_dev`（tasks.md P1-4b 已登记该"小遗"：verify_dw 需补 no-op 发射桩，修桩不属本代理
    白名单）。于是 dW 的**接口路径**判决在这里补齐：判据与 verify_gemm/verify_lora 同配方
    = `ascend_env.compiled_keys()` 含 `ascend|gemm[` 与 `ascend|gemm_dw[`，且 `blockers()` 里
    没有任何 `ascend|` 条目。
    """
    from ascend.kernels import gemm_bwd_dw_kernel, gemm_kernel

    real = tilelang.compile
    slot = {"src": 0}

    def _fake(*a, **k):
        kern = real(*a, **k)                       # 编不出会在此抛 → get_compiled 记 blocker
        text = str(kern.get_kernel_source()) if hasattr(kern, "get_kernel_source") else ""
        slot["src"] = max(slot["src"], len(text))

        class _Dummy:
            def __call__(self, *aa, **kk):         # no-op 发射：无设备，本段只判编译
                return None

        return _Dummy()

    tilelang.compile = _fake
    ascend_env.reset()
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda target=None, device=None: ascend_env.TILELANG
    try:
        torch.manual_seed(3)
        m, n, k = 32, 64, 64
        A = torch.randn(m, k, dtype=torch.float32)
        W = torch.randn(n, k, dtype=torch.float32)
        b = torch.randn(n, dtype=torch.float32)
        dY = torch.randn(m, n, dtype=torch.float32)
        for act in ("none", "relu"):
            gemm_kernel.forward(A, W, b, act=act, out_dtype=torch.float32, target="ascend")
        gemm_bwd_dw_kernel.backward(dY, A, out_dtype=torch.float32, target="ascend")
        keys = sorted(x for x in ascend_env.compiled_keys() if x.startswith("ascend|"))
        bl = {x: v for x, v in ascend_env.blockers().items() if x.startswith("ascend|")}
        counted = ascend_env.compile_count()
    finally:
        tilelang.compile = real
    print(f"[iface] keys={keys}")
    print(f"[iface] blockers={bl} compile_count={counted} src~{slot['src']}B")
    g_ok = any(x.startswith("ascend|gemm[") for x in keys)
    d_ok = any(x.startswith("ascend|gemm_dw[") for x in keys)
    if g_ok and d_ok and not bl:
        print("S-GEMM-DW-IFACE-ASC-COMPILE-PASS（plan/run 真链两腿都编出产物，零 blocker）")
        return 0
    print(f"S-GEMM-DW-IFACE-ASC-COMPILE-FAIL gemm_leg={g_ok} dw_leg={d_ok} blockers={list(bl)}")
    return 1


# --------------------------------------------------------------------------- trace 自证（host 可跑，非判决）
def _drive_trace() -> int:
    """把 ascend 方言的正文只追到 PrimFunc（不实编）：本地提前抓 DSL 级错误，**不算编译判决**。"""
    try:
        import tilelang.ascend.language  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"S-TRACE-SKIP 本机 import tilelang.ascend.language 失败：{str(exc)[:120]}")
        return 0
    T = ascend_env.dialect("ascend")
    ok = True
    m, n, k = 33, 64, 32
    A = torch.randn(m, k, dtype=torch.float32)
    W = torch.randn(n, k, dtype=torch.float32)
    bias = torch.randn(n, dtype=torch.float32)
    C = torch.zeros(m, n, dtype=torch.float32)
    try:
        prog = gemm_asc.gemm_scalar_body(T, A, W, bias, C, **_kwargs_gemm(n, k, "relu"))
        print(f"[trace gemm] TRACE-OK {type(prog).__name__}")
    except Exception as exc:  # noqa: BLE001
        head = str(exc).splitlines()[0][:200] if str(exc) else repr(exc)[:200]
        if "is not registered" in head:      # 本机没注册 ascend tileop（环境缺件，非正文缺陷）
            print(f"[trace gemm] TRACE-SKIP(环境) {head}")
        else:
            ok = False
            print(f"[trace gemm] TRACE-FAIL {head}")
    dY = torch.randn(m, n, dtype=torch.float32)
    GW = torch.zeros(n, k, dtype=torch.float32)
    try:
        prog = gemm_bwd_dw_asc.dw_scalar_body(T, dY, A, GW, **_kwargs_dw(n, k))
        print(f"[trace dw] TRACE-OK {type(prog).__name__}")
    except Exception as exc:  # noqa: BLE001
        head = str(exc).splitlines()[0][:200] if str(exc) else repr(exc)[:200]
        if "is not registered" in head:
            print(f"[trace dw] TRACE-SKIP(环境) {head}")
        else:
            ok = False
            print(f"[trace dw] TRACE-FAIL {head}")
    print("S-TRACE-OK（仅追溯自证，编译判决以 --ascend 为准）" if ok else "S-TRACE-FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    argv = sys.argv[1:]
    rcs = []
    if "--ascend" in argv:
        rcs.append(_drive_ascend())
    if "--iface" in argv:
        rcs.append(_drive_iface())
    if "--trace" in argv:
        rcs.append(_drive_trace())
    if "--cpu" in argv or not rcs:
        rcs.append(_drive_gemm_cpu())
        rcs.append(_drive_dw_cpu())
    print(f"\n[summary] rc={rcs} (0=PASS)")
    sys.exit(0 if all(r == 0 for r in rcs) else 1)
