#!/usr/bin/env python3
"""reconcile/verify_dw.py — K 代理：gemm_bwd_dw_asc.py 的 910B target=ascend 编译判决。

判决主体（接口真实路径，**如实报**）：绕设备门 + 把 `tilelang.compile` 换成"真编译后发射即
no-op"的替身（**桩形态复用 verify_lora.py，不另立标准**），用真实形状（dY(m,n) A(m,k) GW(n,k)，
m 动态、GW fp32）驱动 gemm_bwd_dw_kernel.backward(target="ascend")。

【为什么现在必须有发射桩（P1-13/A2 补桩）】§12 `asc_fill_l1` 落地后本件编译不再失败，控制流
第一次推进到**设备发射门**：`gemm_bwd_dw_asc.run()` 里 `kernel(dY, A, GW)` 真被调用 →
`RuntimeError: kernel dw_impl input dY device_type mismatch, expected ext_dev`（cpu 张量发不出
ascend 产物）。本脚本当初只覆盖"编译期抛→记 blocker→报 FAIL"这一支，桩函数 `_stub_compile` 写了
却没接上（且它返回真产物、不隔离发射），于是判决行永远打不出来——这是**探测器缺桩**，不是产品缺陷。
补桩后应稳定出 `K-GEMM-BWD-DW ASC-COMPILE-PASS`。

判据**不因桩而放宽**（R14 防假绿，口径同 run_numerics._route_key / verify_lora）：
  · `ascend_env.compiled_keys()` 必须含 `ascend|gemm_dw[ascend]`（键由 get_compiled 在**真编译成功**
    后登记，回退路径不留痕迹）；
  · `ascend_env.blockers()` 里不得有 `ascend|` 项；
  · 桩自己必须**真见过 ≥1 次成功编译**且拿到**非空生成码**（编译未发生 ⇒ 一律 FAIL）。
no-op 发射 ⇒ **本波不判数值**（GW 是未写入的空表，绝不可当数值证据读）；数值判决走
`run_numerics.py --only dw_prod`（card 真机）或 cpu-alias 轨。

结构自证（旁证，非判决主体）：另编一份"静态 m 且整除 tc"的 l0tr 同型件——若无 asc_fill_l1
即编过，则证明 DSL 形态正确。桩只作用于判决主体，编完即还原 `tilelang.compile`，故此支仍真编
真取生成码。

用法（容器内，release 挂 /work）：
  source /usr/local/Ascend/cann-8.5.0/set_env.sh
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 300 python3 ascend/kernels/reconcile/verify_dw.py
"""
import os
import sys

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

import torch           # noqa: E402
import tilelang        # noqa: E402
from ascend.kernels import ascend_env, gemm_bwd_dw_kernel  # noqa: E402


class LaunchSkip(Exception):
    pass


#: 桩的自证账目：真编译成功次数 + 累计生成码（编译未发生 ⇒ 守卫把判决打成 FAIL）
COMPILED_SRC = {"acc": "", "n": 0}


def _stub_compile():
    """真编译后返回 no-op 发射替身（桩形态同 verify_lora.py）。

    编译失败仍原样抛在 `real()` 里 → `ascend_env.get_compiled` 记 blocker → 上层落 eager，
    判决如实变 FAIL；桩只吃掉"发射"这一段（无设备，本波不判数值）。
    """
    real = tilelang.compile

    def _fake(*a, **k):
        kern = real(*a, **k)   # 编不出会在此抛 → get_compiled 记 blocker → 上层回退
        src = str(kern.get_kernel_source()) if hasattr(kern, "get_kernel_source") else ""
        COMPILED_SRC["acc"] += src
        COMPILED_SRC["n"] += 1

        class _Dummy:
            def __call__(self, *aa, **kk):
                return None   # no-op 发射（无设备，不判数值）

        return _Dummy()

    tilelang.compile = _fake
    return lambda: setattr(tilelang, "compile", real)


def _gate():
    ascend_env.reset()
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda target=None, device=None: ascend_env.TILELANG


def _interface_dynamic_m() -> int:
    """判决主体：接口路径（动态 m）——编译判决，发射 no-op 桩隔离。"""
    restore = _stub_compile()
    try:
        _gate()
        m, n, k = 32, 64, 64
        dY = torch.randn(m, n, dtype=torch.float32)
        A = torch.randn(m, k, dtype=torch.float32)
        gemm_bwd_dw_kernel.backward(dY, A, out_dtype=torch.float32, target="ascend")
    finally:
        restore()
    keys = [x for x in ascend_env.compiled_keys() if x.startswith("ascend|")]
    bl = {x: v for x, v in ascend_env.blockers().items() if x.startswith("ascend|")}
    print(f"[interface·dynamic-m] keys={keys}")
    print(f"[interface·dynamic-m] blockers={bl}")
    print(f"[interface·dynamic-m] real_compiles={COMPILED_SRC['n']} "
          f"gen_src_bytes={len(COMPILED_SRC['acc'])} compile_count={ascend_env.compile_count()}")
    if not keys and bl:
        first = next(iter(bl.values()))
        print(f"[interface·dynamic-m] 首错={first[:240]}")
        if "asc_fill_l1" in first:
            print("K-GEMM-BWD-DW ASC-COMPILE-BLOCKED(§12 asc_fill_l1)")
        print("K-GEMM-BWD-DW ASC-COMPILE-FAIL")
        return 1
    # 假绿守卫：键在 + 无阻塞，还须桩真见过一次成功编译与非空生成码（编译未发生即 FAIL）
    dw_ok = any(x.startswith("ascend|gemm_dw[ascend]") for x in keys)
    guarded = dw_ok and not bl and COMPILED_SRC["n"] >= 1 and len(COMPILED_SRC["acc"]) > 0
    if guarded:
        print(f"K-GEMM-BWD-DW ASC-COMPILE-PASS .o~{len(COMPILED_SRC['acc'])}B "
              f"key={next(x for x in keys if x.startswith('ascend|gemm_dw[ascend]'))} "
              f"note=发射已 no-op 桩隔离⇒本行不判数值")
        return 0
    if dw_ok and not bl:
        print("K-GEMM-BWD-DW ASC-COMPILE-FAIL(假绿守卫：compiled_keys 有键但桩未观测到真编译/非空生成码)")
        return 1
    print("K-GEMM-BWD-DW ASC-COMPILE-FAIL")
    return 1


def _static_structure_sanity() -> int:
    """旁证：静态 m(整除 tc) 的同型 l0tr —— 若编过则结构正确、唯动态尾填充挡路。"""
    import tilelang.ascend.language as T
    n, k, tc, bn, bk = 64, 64, 16, 64, 64
    m_static = 32           # 编译期常量，32 % 16 == 0
    nb, nk = n // bn, k // bk
    tiles = nb * nk
    blocks = min(tiles, 8)
    iters = (tiles + blocks - 1) // blocks
    ks = m_static // tc     # 静态整除 → 不发射 asc_fill_l1

    @T.prim_func
    def dw_static(dY: T.Tensor((m_static, n), "float32"),
                  A: T.Tensor((m_static, k), "float32"),
                  GW: T.Tensor((n, k), "float32")):
        with T.Kernel(blocks) as bid:
            dy_l1 = T.alloc_l1((tc, bn), "float32")
            x_l1 = T.alloc_l1((tc, bk), "float32")
            a_l0 = T.alloc_l0a((bn, tc), "float32")
            b_l0 = T.alloc_l0b((bk, tc), "float32")
            acc = T.alloc_l0c((bn, bk), "float32")
            with T.Cube():
                for it in T.serial(iters):
                    tid = bid + it * blocks
                    n_tile = tid // nk
                    k_tile = tid % nk
                    for kt in T.serial(ks):
                        T.copy(dY[kt * tc, n_tile * bn], dy_l1)
                        T.copy(A[kt * tc, k_tile * bk], x_l1)
                        T.copy(dy_l1, a_l0, transpose=True)
                        T.copy(x_l1, b_l0, transpose=True)
                        T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=kt == 0)
                    T.copy(acc, GW[n_tile * bn, k_tile * bk])

    try:
        kern = tilelang.compile(dw_static, target="ascend", out_idx=[])
        src = str(kern.get_kernel_source())
        print(f"[static-sanity] PASS .o~{len(src)}B asc_fill_l1_in_src={'asc_fill_l1' in src} "
              f"mix={'__mix__' in src}")
        print("K-GEMM-BWD-DW STRUCTURE-SANITY-PASS(l0tr 形态可编，唯动态尾填充挡路)")
        return 0
    except Exception as e:
        msg = str(e)
        i = msg.find("error:")
        print(f"[static-sanity] FAIL {msg[i:i+200].replace(chr(10),' ') if i>=0 else msg.splitlines()[0][:200]}")
        return 1


if __name__ == "__main__":
    rc1 = _interface_dynamic_m()
    rc2 = _static_structure_sanity()
    # 判决主体（接口动态 m）为准：受阻即非零退出，如实报；旁证单独打印不计入主判。
    print(f"\n[summary] interface-dynamic-m rc={rc1} (0=PASS/1=FAIL); "
          f"static-sanity rc={rc2}")
    sys.exit(rc1)
