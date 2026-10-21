#!/usr/bin/env python3
"""run_numerics.py — 910B 上卡收数工装（P1-1i 改版：**双轨并列**，三条产品件 + 八条 performance-track）。

【本波改了什么（元缺陷修复）】wave2 之前本工装的 `gemm_l1`/`dw`/`readout`/`addln` 四条 case 内联的是
`attempts/` 那批 DSL 正文（alloc_l1/asc_copy_* 的 **intrinsic cube 路**，正是 507015 那条；以及 B 域
scores 投影路），**产品件**（`ascend/kernels/*_asc.py` 经 P1-1f 收敛的标量体、P1-1h 收敛的
gather/scatter）在卡上永远不会被测量——判决件量的不是产品代码（S/U 两代理各自独立撞破，wave2 已因此
误判 readout）。处置（保留双轨，不销毁 information）：
  ① 四条内联 case 改名 `*_intrinsic` 并标 `track=performance`，正文与判据**一字不动**（历史判决行仍可
     复跑对照；旧名 `gemm_l1`/`dw`/`addln`/`readout` 保留为**弃用别名**，fixup_card.sh /
     bundle_ondemand.sh 这类不在本波白名单里的调用行不会因此断掉）；
  ② 新增产品件 case：`linear_prod` / `dw_prod` / `readout_prod` / `addln_prod`，一律经**对外入口件**
     （`ascend.kernels.gemm_kernel` / `gemm_bwd_dw_kernel` / `letter_readout_kernel` /
     `add_ln_kernel`）调用、走
     `ascend_env` 路由（与 train_step 用的那条路一致），**判决前强制自证"这一趟真开了模具"**：
     `ascend_env.compiled_keys()` 里必须有 `ascend|...` 键。这一条是 load-bearing 的——入口件在
     plan 拒形状 / 后端不可用 / 方言别名未生效时会**静默落 torch eager**，rel 照样是 0.0（本波探针
     实测踩过：只改 `sys.modules['tilelang.ascend.language']` 不生效，`import a.b.c as T` 取的是父包
     属性）。守卫不响 = 假绿，直接判 FAIL。
  ③ `addln_prod` 在 cpu-alias 轨额外装一层**存储层 shim**（`alloc_shared`→`alloc_local`）：add_ln 的
     产品正文 `add_ln_asc_impl` 是 **ascend-only 写法**（4 个 `T.alloc_shared`，scope=`shared.dyn`），
     而 c 后端的 `lower_device_kernel_launch` 只允许**一个** dynamic shared 分配
     （`Only one dynamic shared memory allocation is allowed.`）⇒ 直连 cpu 方言编不过。gemm/dw 的
     产品正文靠 `_alloc_tile` **按方言分叉存储层**（cpu→`alloc_local`），add_ln 件没这层分叉，故在
     cpu-alias 轨由本工装替它做同一件事（只换存储层，**算式与判据一字不动**；ascend 发射层仍归
     card 轨/容器编译判决）。见 `_addln_cpu_alias_storage_shim`。

【三档 track 语义（同一套产品件 case，靠 --track 与 --no-run 分派）】
  card        生产 ascend 路由（target="ascend"）。有卡 ⇒ 真编真发射判数值；无卡（容器有 bisheng）⇒
              cpu 张量 + 桩掉发射，**只判编译**，打印 `Y-<case>-ASC-COMPILE-PASS`。
  cpu-alias   U_oracle.py --prod 的手法（修正版）：把 `tilelang.ascend.language` 这个名字指向 cpu
              方言模块，**生产 ascend 正文**在 host 以 target="c" 真编真跑 ⇒ 无卡条件下能量到产品代码
              的数值。这是"无卡"能拿到的最强证据，不冒充真机判决（差异只在 ascend 下发射那一层，由
              card 轨/容器生成码体检负责）。
  auto        有 npu ⇒ card，否则 cpu-alias。
注意：`--no-run` 只在 card 轨有意义（把发射桩掉，只留编译判决）；cpu-alias 轨在 host 本来就真跑，
`--no-run` 不改其行为（照样打印 rel）。

一行/件判决（R19：无输出=未完成）：
  Y-<case>-PROD-PASS rel=<v> ...          产品件数值正确（阈值见各 case docstring）
  Y-<case>-ASC-COMPILE-PASS key=<k>       产品件仅编译判（--track card --no-run）
  Y-<case>-PROD-FAIL <reason>             产品件失败（含"落 eager"假绿守卫、超差、方向判别失败）
  NUM-<name>-PASS rel=<v> track=performance   attempts/§12 intrinsic 路（非产品件）
  NUM-<name>-FAIL <reason>
用法：
  卡上（bundle 已铺好 compat+注入）：  python3 run_numerics.py --only linear_prod
  容器（有 bisheng 无设备）：          python3 run_numerics.py --only linear_prod --track card --no-run
  host（无卡，量生产正文的数值）：      python3 run_numerics.py --only readout_prod --no-run
"""
import os
import sys
import argparse
import importlib

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))
ATT = os.path.join(HERE, "attempts")
RELEASE_ROOT = os.path.dirname(os.path.dirname(HERE))       # .../release（产品件 `ascend` 包的家）
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)                        # 产品件 case 要 import ascend.kernels
# B 域不列入 sys.path：b_addln_910b.py / b_readout_910b.py 是 module 级脚本
# （末尾 sys.exit(main_())），importlib 整脚本执行会直接终止本进程；两件已按
# 其 build() 逐字内联为 num_addln_intrinsic / num_readout_intrinsic（见 §7 §8）。
for d in ("C", "D", "E"):
    sys.path.insert(0, os.path.join(ATT, d))

import numpy as np  # noqa: E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import tilelang.ascend.language as TA  # noqa: E402  # Ascend 方言（T.Kernel 无 threads=，刻意 shadow CUDA 门面）
from tilelang.ascend.language import Cube, alloc_l1, alloc_l0c  # noqa: E402

AP = argparse.ArgumentParser()
AP.add_argument("--only", default="all")
AP.add_argument("--no-run", action="store_true")
AP.add_argument("--track", default="auto", choices=("auto", "card", "cpu-alias"),
                help="card=生产 ascend 路由（有卡真发射/无卡只判编译）；"
                     "cpu-alias=host 用 cpu 方言别名注入强跑生产 ascend 正文；auto=按 npu 在场自动选")
ARGS = AP.parse_args()

DT = {"float16": "fp16", "float32": "fp32", "bfloat16": "bf16"}


def _t(shape, dtype, scale=1.0):
    import torch
    TORCH_DT = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    return (torch.randn(*shape, device="npu", dtype=torch.float32) * scale).to(TORCH_DT[DT[dtype]])


def _np(x):
    return x.detach().float().cpu().numpy()


def compile_and_time(kern, args, iters=20):
    import torch, time
    torch.npu.synchronize()
    out = kern(*args)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        kern(*args)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    return out, ms


def relerr(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


# ── 1. gemm direct（bf16 2048³，e2e_cube 形态 + 数值与 tflops）────────────────
# [track=performance] 本件正文由本文件自写（T.Cube + alloc_shared + T.gemm），量的是 **cube 载体的
# 吞吐/编译面**，不是 ascend/kernels/gemm_asc.py 那套标量体产品件（产品件见 prod_linear）。
def num_gemm_direct():
    def gemm(M, N, K, blk):
        @T.prim_func
        def main(A: T.Tensor((M, K), "bfloat16"), B: T.Tensor((K, N), "bfloat16"),
                 C: T.Tensor((M, N), "bfloat16")):
            with T.Kernel(T.ceildiv(N, blk), T.ceildiv(M, blk), threads=1) as (bx, by):
                A_s = T.alloc_shared((blk, K), "bfloat16")
                B_s = T.alloc_shared((K, blk), "bfloat16")
                C_l = T.alloc_shared((blk, blk), "float")
                with Cube():
                    T.copy(A[by * blk, 0], A_s)
                    T.copy(B[0, bx * blk], B_s)
                    T.gemm(A_s, B_s, C_l, clear_accum=True)
                    for i in T.serial(blk):
                        for j in T.serial(blk):
                            C[by * blk + i, bx * blk + j] = T.cast(C_l[i, j], "bfloat16")
        return main
    M = N = K = 2048
    k = tilelang.compile(gemm(M, N, K, 128), target="ascend", out_idx=-1)
    if ARGS.no_run:
        return "NUM-gemm_direct-PASS rel=n/a (compile-only)"
    import torch
    A = torch.randn(M, K, dtype=torch.bfloat16, device="npu")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="npu")
    out, ms = compile_and_time(k, (A, B))
    ref = (A.float() @ B.float()).to(torch.bfloat16)
    r = relerr(_np(out), _np(ref))
    tf = 2 * M * N * K / (ms * 1e-3) / 1e12
    assert r < 0.02, f"rel={r}"
    return f"NUM-gemm_direct-PASS rel={r:.5f} ms={ms:.3f} tflops={tf:.2f}"


# ── 2. gemm_l1_intrinsic（shared.l1 模板路 64³；V1/V2/V3 单位证真载体）─────────
# [track=performance] 正文逐字来自 attempts/§12 的 intrinsic 路（alloc_l1 / alloc_l0c /
# asc_copy_* 的 **cube 路**，正是真机 507015 那条）。它**不是**产品件：`ascend/kernels/gemm_asc.py`
# 早在 P1-1f（D-cube1 改道）就换成可移植标量体，本 case 与它零重叠。保留价值只有两条：
# ① V1/V2/V3 的 L1/L0C stride 单位与 GM→L1 分形布局仍要靠这条探针证真（performance-track 考古）；
# ② 与 prod_linear 同窗对照，才能说清"cube 路的失败/成功与产品件无关"。
# 产品件的判决请看 `--only linear_prod`。
def num_gemm_l1_intrinsic():
    R = C64 = 64

    def kern():
        @T.prim_func
        def main(A: T.Tensor((R, C64), "bfloat16"), B: T.Tensor((C64, C64), "bfloat16"),
                 O: T.Tensor((R, C64), "float32")):
            with T.Kernel(1):
                a_l1 = alloc_l1((R, C64), "bfloat16")
                b_l1 = alloc_l1((C64, C64), "bfloat16")
                c_l0 = alloc_l0c((R, C64), "float32")
                with Cube():
                    T.copy(A, a_l1)
                    T.copy(B, b_l1)
                    T.gemm(a_l1, b_l1, c_l0, transpose_B=True, clear_accum=True)
                    T.copy(c_l0, O)
        return main
    k = tilelang.compile(kern(), target="ascend", out_idx=-1)
    if ARGS.no_run:
        return "NUM-gemm_l1_intrinsic-PASS rel=n/a (compile-only) track=performance"
    A = _t((R, C64), "bfloat16")
    Bm = _t((C64, C64), "bfloat16")
    out, _ = compile_and_time(k, (A, Bm), iters=5)
    ref = A.float() @ Bm.float()
    r = relerr(_np(out), ref.cpu().numpy())
    assert r < 0.02, f"rel={r} —— V1/V2/V3 任一单位错都体现在此"
    return f"NUM-gemm_l1_intrinsic-PASS rel={r:.5f} track=performance"


# ── 3. gdn short conv（import C 波 kernel+golden）─────────────────────────────
# [track=performance] 正文来自 attempts/C/c_gdn_conv_910b.py（C 波研究件），非 gdn_conv_asc.py 产品件。
def num_gdn():
    mod = importlib.import_module("c_gdn_conv_910b")
    gd = importlib.import_module("c_gdn_golden")
    C_, L_, Kk = 512, 512, 4          # 缩容快跑（全 2048×512 可 --only gdn 复跑）
    fn = mod.gdn_short_conv_fwd(C=C_, L=L_, K=Kk, silu="exp")
    k = tilelang.compile(fn, target="ascend", out_idx=-1)
    if ARGS.no_run:
        return "NUM-gdn-PASS rel=n/a (compile-only)"
    rng = np.random.default_rng(7)
    X = rng.standard_normal((C_, L_)).astype(np.float32)
    W = rng.standard_normal((C_, Kk)).astype(np.float32) * 0.5
    BI = rng.standard_normal((C_,)).astype(np.float32) * 0.1
    import torch
    args = tuple(torch.from_numpy(x).cuda() if False else torch.from_numpy(x).npu() for x in (X, W, BI))
    out, _ = compile_and_time(k, args, iters=5)
    ref = gd.short_conv_compat(X, W, BI, silu="exp")
    r = relerr(_np(out), ref)
    assert r < 2e-5, f"rel={r} (期望 ≤ C 波实测 3.3e-7 级)"
    return f"NUM-gdn-PASS rel={r:.2e}"


# ── 4. dw_intrinsic（l0tr npz 判据：P0 方向判别 + P1 阈值）────────────────────
# [track=performance] 正文来自 attempts/D/d_dw_910b.py（L0tr/intrinsic 路），**不是**产品件
# `ascend/kernels/gemm_bwd_dw_asc.py` 的 dw_scalar_body。保留价值：npz 那对 exact/exact_T 判据是
# 方向判别的现成尺子；产品件的判决请看 `--only dw_prod`。
def num_dw_intrinsic():
    z = np.load(os.path.join(ATT, "D", "dw_pairs.npz"))
    dY, X, exact, exact_T = z["dY"], z["X"], z["exact"], z["exact_T"]
    Tv, Nv, Kv, TT, BN, BK = int(z["T"]), int(z["N"]), int(z["K"]), int(z["TT"]), int(z["BN"]), int(z["BK"])
    mod = importlib.import_module("d_dw_910b")
    fn = mod.dw_l0tr(Tv, Nv, Kv, BN, BK, TT, min(Nv // BN * (Kv // BK), 64), "float16", "float32")
    k = tilelang.compile(fn, target="ascend", out_idx=-1)
    if ARGS.no_run:
        return "NUM-dw_intrinsic-PASS rel=n/a (compile-only) track=performance"
    import torch
    a1 = torch.from_numpy(dY).npu(); a2 = torch.from_numpy(X).npu()
    out, _ = compile_and_time(k, (a1, a2), iters=5)
    got = _np(out)
    r_ok, r_swap = relerr(got, exact), relerr(got, exact_T)
    assert r_ok < 0.02 and r_ok < r_swap - 0.5, f"rel={r_ok:.4f} rel_swap={r_swap:.4f}（P0 方向判别失败）"
    return f"NUM-dw_intrinsic-PASS rel={r_ok:.5f} dir_disc={r_swap:.3f} track=performance"


# ── 5. rope（表加载 fwd，f64 精确式对拍）─────────────────────────────────────
# [track=performance] 正文来自 attempts/E/e_rope_910b.py，非 rope_asc.py 产品件。
def num_rope():
    mod = importlib.import_module("e_rope_910b")
    eg = importlib.import_module("e_golden")
    tk, hd, dm = 64, 8, 64
    fn = mod.rope_fwd(tokens=tk, heads=hd, dim=dm, mode="table", sign=1)
    k = tilelang.compile(fn, target="ascend", out_idx=1)
    if ARGS.no_run:
        return "NUM-rope-PASS rel=n/a (compile-only)"
    rng = np.random.default_rng(7)
    qkv = rng.standard_normal((tk, 3, hd, dm)).astype(np.float32)
    ang = rng.uniform(0, 3.14, (tk, dm // 2)).astype(np.float32)
    cos, sin = np.cos(ang), np.sin(ang)
    freq = (1.0 / (10000 ** (2 * np.arange(dm // 2) / dm))).astype(np.float32)
    import torch
    ts = [torch.from_numpy(x).npu() for x in (qkv, cos, sin, freq)]
    out = k(*ts)
    ref = eg.rope_ref_f64(qkv, cos, sin, sign=1)
    r = relerr(_np(out).astype(np.float32), ref.astype(np.float32))
    assert r < 5e-6, f"rel={r}"
    return f"NUM-rope-PASS rel={r:.2e}"


# ── 6. attn_sw（滑窗 softmax，numpy 参考）─────────────────────────────────────
# [track=performance] 正文来自 attempts/E/e_attn_sw_910b.py，非 attn_sw_asc.py 产品件。
def num_attnsw():
    mod = importlib.import_module("e_attn_sw_910b")
    h, s, d, w = 4, 64, 32, 8
    fn = mod.attn_sw_fwd(heads=h, seq=s, dim=d, window=w, variant="stage")
    k = tilelang.compile(fn, target="ascend", out_idx=3)
    if ARGS.no_run:
        return "NUM-attnsw-PASS rel=n/a (compile-only)"
    rng = np.random.default_rng(7)
    Q = rng.standard_normal((h, s, d)).astype(np.float32)
    K = rng.standard_normal((h, s, d)).astype(np.float32)
    V = rng.standard_normal((h, s, d)).astype(np.float32)
    import torch
    out = k(*[torch.from_numpy(x).npu() for x in (Q, K, V)])
    scale = d ** -0.5
    ref = np.zeros_like(Q)
    for hh in range(h):
        for t in range(s):
            lo = max(0, t - w + 1)
            sc = (Q[hh, t] @ K[hh, lo:t + 1].T) * scale
            sc -= sc.max()
            e = np.exp(sc); e /= e.sum()
            ref[hh, t] = e @ V[hh, lo:t + 1]
    r = relerr(_np(out).astype(np.float32), ref)
    assert r < 1e-4, f"rel={r}"
    return f"NUM-attnsw-PASS rel={r:.2e}"


# ── 7. addln_intrinsic（残差相加 + 层内整形）─────────────────────────────────
# [track=performance] DSL 逐字内联自 attempts/B/b_addln_910b.py::build()（静态 rows 分支，即 --dyn
# 关闭），位宽/形状取该文件的环境变量缺省：B_ROWS=64 B_DIM=512 B_BM=8 B_RES_FP32=1 B_OUT_FP16=1
# B_EPS=1e-5 ⇒ RDT=float32、ODT=float16、UB~28KB。out_idx=-1 与该文件 main_() 的调用一致。
# 这条 case 与产品件 `add_ln_asc.add_ln_asc_impl` **不是同一份正文**（产品件有 bm 收缩、fp32 交货口径、
# 反向 ln_bwd 腿）——**它的判决地位已由 `addln_prod`（产品接口件）取代**：要拿 add_ln 的正确性结论请读
# `Y-addln_prod-PROD-PASS`；本条留作 §12/intrinsic 历史对拍（results 不变、不删，避免历史判决行断链）。
# 数值口径照其 docstring：三趟扫描、**有偏方差**（/DIM 非 /(DIM-1)）、Hout 全程 fp32 不降位宽、Y 落 fp16。
# 阈值来自宿主机仿真（照 DSL 逐趟 fp32 标量算术重放，/tmp/g_probe/emu.py）：
#   r_h：正确=0.0（单次 fp32 加，与参考逐位同）/ 残差路降 fp16=2.1e-4 ⇒ 1e-5 精确判别"不许降位宽"
#   r_y：正确=1.5e-5 / 方差错用无偏=7.4e-4 ⇒ 3e-4 判别"有偏"口径（原拟 5e-3 判别不了，已收紧）
def num_addln_intrinsic():
    ROWS, DIM, BM = 64, 512, 8
    F16, F32 = "float16", "float32"
    RDT, ODT = F32, F16                 # RES_FP32=1 / OUT_FP16=1
    EPS = 1e-5

    def build():
        @TA.prim_func
        def main(X: TA.Tensor((ROWS, DIM), F16), Res: TA.Tensor((ROWS, DIM), RDT),
                 G: TA.Tensor((DIM,), F32), Bt: TA.Tensor((DIM,), F32),
                 Y: TA.Tensor((ROWS, DIM), ODT), Hout: TA.Tensor((ROWS, DIM), F32)):
            with TA.Kernel(TA.ceildiv(ROWS, BM)) as bx:
                x_ub = TA.alloc_shared((BM, DIM), F16)
                r_ub = TA.alloc_shared((BM, DIM), RDT)
                g_ub = TA.alloc_shared((DIM,), F32)
                b_ub = TA.alloc_shared((DIM,), F32)
                TA.copy(X[bx * BM, 0], x_ub)
                TA.copy(Res[bx * BM, 0], r_ub)
                TA.copy(G[0:DIM], g_ub)
                TA.copy(Bt[0:DIM], b_ub)
                for i in TA.serial(BM):
                    row = bx * BM + i
                    val = TA.alloc_var(F32)
                    acc = TA.alloc_var(F32)
                    mu = TA.alloc_var(F32)
                    dv = TA.alloc_var(F32)
                    rs = TA.alloc_var(F32)
                    # 第一趟：残差相加，fp32 原样交出口，同时攒行和
                    acc = 0.0
                    for j in TA.serial(DIM):
                        val = TA.cast(x_ub[i, j], F32) + TA.cast(r_ub[i, j], F32)
                        Hout[row, j] = val
                        acc = acc + val
                    mu = acc / DIM
                    # 第二趟：偏差平方和（有偏方差）
                    acc = 0.0
                    for j in TA.serial(DIM):
                        dv = TA.cast(x_ub[i, j], F32) + TA.cast(r_ub[i, j], F32) - mu
                        acc = acc + dv * dv
                    rs = TA.rsqrt(acc / DIM + EPS)
                    # 第三趟：缩放 + 仿射，按 ODT 交货
                    for j in TA.serial(DIM):
                        dv = TA.cast(x_ub[i, j], F32) + TA.cast(r_ub[i, j], F32) - mu
                        Y[row, j] = TA.cast(dv * rs * g_ub[j] + b_ub[j], ODT)

        return main

    k = tilelang.compile(build(), target="ascend", out_idx=-1)
    if ARGS.no_run:
        return "NUM-addln_intrinsic-PASS rel=n/a (compile-only) track=performance"
    import torch
    X = _t((ROWS, DIM), F16)
    Res = _t((ROWS, DIM), RDT) + 4.0    # 大共模/小波动：贴生产残差流量级，也是三趟式的动机
    G = _t((DIM,), F32)
    Bt = _t((DIM,), F32)
    Y = torch.zeros((ROWS, DIM), dtype=torch.float16, device="npu")   # 非 out 槽 ⇒ 就地写回
    out, _ = compile_and_time(k, (X, Res, G, Bt, Y), iters=5)         # out == Hout
    # 参考实现：入参先按 kernel 实际位宽量化（X 是 fp16），再全程 fp32；算式顺序照 DSL
    xv, rv = _np(X).astype(np.float32), _np(Res).astype(np.float32)
    gv, bv = _np(G).astype(np.float32), _np(Bt).astype(np.float32)
    v = (xv + rv).astype(np.float32)                        # Hout：fp32 残差流
    mu = v.mean(axis=1, keepdims=True)
    dev = (v - mu).astype(np.float32)
    var_b = (dev * dev).mean(axis=1, keepdims=True).astype(np.float32)      # 有偏（kernel 口径）
    var_u = (dev * dev).sum(axis=1, keepdims=True) / (DIM - 1)              # 无偏（错误口径，仅作判别）
    ref_h = v
    ref_y = ((dev * (1.0 / np.sqrt(var_b + EPS)).astype(np.float32)) * gv + bv).astype(np.float16)
    alt_y = ((dev * (1.0 / np.sqrt(var_u + EPS)).astype(np.float32)) * gv + bv).astype(np.float16)
    got_y, got_h = _np(Y).astype(np.float32), _np(out).astype(np.float32)
    r_h = relerr(got_h, ref_h.astype(np.float32))
    r_y = relerr(got_y, ref_y.astype(np.float32))
    r_unb = relerr(got_y, alt_y.astype(np.float32))          # 与"无偏"错口径的距离（须明显更远）
    assert r_h < 1e-5, f"Hout rel={r_h:.3e}（fp32 残差路应逐位同；~2e-4 即降了位宽）"
    assert r_y < 3e-4, (f"Y rel={r_y:.3e}（仿真底噪 1.5e-5；若 Y 全零则 rel≈1 ⇒ 就地写回未生效）"
                        f" r_unb={r_unb:.3e}")
    assert r_y < r_unb - 2e-4, f"有偏/无偏口径判别失败 r_y={r_y:.3e} r_unb={r_unb:.3e}"
    return (f"NUM-addln_intrinsic-PASS rel_y={r_y:.2e} rel_h={r_h:.2e} "
            f"unb_disc={r_unb:.2e} track=performance")


# ── 8. readout_intrinsic（末位读出 × 字母行投影 = scores）─────────────────────
# [track=performance][语义分叉] DSL 逐字内联自 attempts/B/b_readout_910b.py::build()，取其环境变量
# 缺省：R_BATCH=8 R_SEQ=32 R_DIM=512 R_K=26 R_BM=4 R_HDT=float16（UB~60KB），out_idx=-1 同该文件。
# **它算的是 scores（末位散取 + 字母行投影点积），而产品件 `letter_readout_asc` 是纯 gather/scatter
# （index_select / index_add），两者语义不同** ⇒ wave2 那条 `NUM-readout-FAIL rel=0.1777896` 判的
# 是 b 域投影路，不能读成"产品件读出错了"（U 的 U_readout_diag.md §1 已把这处错位写实）。
# 范围声明照其 docstring：只出 scores，不含 softmax/to_probs（dav-2201 vector 面无 expf/log）。
# 阈值来自宿主机仿真：512 项 fp32 顺序累加 vs f64 参考 rel=4.0e-7 ⇒ 1e-5 留 25x 余量。
# 产品件的判决请看 `--only readout_prod`。
def num_readout_intrinsic():
    BATCH, SEQ, DIM, KK, BM = 8, 32, 512, 26, 4
    F32, I32 = "float32", "int32"
    HDT = "float16"
    VOCAB = 4096

    def build():
        @TA.prim_func
        def main(HID: TA.Tensor((BATCH, SEQ, DIM), HDT), POS: TA.Tensor((BATCH,), I32),
                 W: TA.Tensor((VOCAB, DIM), HDT), IDS: TA.Tensor((KK,), I32),
                 S: TA.Tensor((BATCH, KK), F32)):
            with TA.Kernel(TA.ceildiv(BATCH, BM)) as bx:
                h_ub = TA.alloc_shared((BM, DIM), F32)   # 本块每行的"末位数字"，已升 fp32
                w_ub = TA.alloc_shared((KK, DIM), F32)    # k 个字母行，已升 fp32
                # 1) 逐行按 pos 取末位（动态下标散取；界守卫由 codegen 生成）
                for i in TA.serial(BM):
                    for j in TA.serial(DIM):
                        h_ub[i, j] = TA.cast(HID[bx * BM + i, POS[bx * BM + i], j], F32)
                # 2) 按 letter_ids 取字母行
                for t in TA.serial(KK):
                    for j in TA.serial(DIM):
                        w_ub[t, j] = TA.cast(W[IDS[t], j], F32)
                # 3) 点积出 (BM, K) 分数，标量落 GM
                for i in TA.serial(BM):
                    acc = TA.alloc_var(F32)
                    for t in TA.serial(KK):
                        acc = 0.0
                        for j in TA.serial(DIM):
                            acc = acc + h_ub[i, j] * w_ub[t, j]
                        S[bx * BM + i, t] = acc

        return main

    k = tilelang.compile(build(), target="ascend", out_idx=-1)
    if ARGS.no_run:
        return "NUM-readout_intrinsic-PASS rel=n/a (compile-only) track=performance"
    import torch
    rng = np.random.default_rng(7)
    HID = _t((BATCH, SEQ, DIM), HDT)
    W = _t((VOCAB, DIM), HDT)
    pos = rng.integers(0, SEQ, size=BATCH).astype(np.int32)   # 动态下标必须落在 [0,SEQ)
    ids = np.arange(KK, dtype=np.int32)                       # A..Z 取词表前 26 行
    # 前置守卫：散取下标必须"非平凡"，否则本 case 会退化成没测动态寻址的空测
    assert pos.min() >= 0 and pos.max() < SEQ, f"POS 越界 [{pos.min()},{pos.max()}] vs SEQ={SEQ}"
    assert np.unique(pos).size > 1, "POS 全同 ⇒ 动态散取未被激励"
    assert np.unique(ids).size == KK and ids.max() < VOCAB, "IDS 退化/越界"
    POS = torch.from_numpy(pos).npu()
    IDS = torch.from_numpy(ids).npu()
    out, _ = compile_and_time(k, (HID, POS, W, IDS), iters=5)   # out == S (BATCH,KK) fp32
    hid_np, w_np = _np(HID).astype(np.float32), _np(W).astype(np.float32)
    rows = hid_np[np.arange(BATCH), pos, :].astype(np.float64)   # 逐行按 pos 散取
    wr = w_np[ids, :].astype(np.float64)
    ref = (rows @ wr.T).astype(np.float32)                       # .float() 语义 = fp32 累加
    bad = (hid_np[:, 0, :].astype(np.float64) @ wr.T).astype(np.float32)  # 错取首位（不随 pos）
    r = relerr(_np(out).astype(np.float32), ref)
    r_bad = relerr(_np(out).astype(np.float32), bad)
    assert r < 1e-5, f"rel={r}（仿真底噪 4.0e-7；若 S 全零则 rel≈1）"
    assert r < r_bad - 0.5, f"末位散取方向判别失败 rel={r:.3e} rel_firstpos={r_bad:.3e}"
    return f"NUM-readout_intrinsic-PASS rel={r:.2e} gather_disc={r_bad:.2e} track=performance"


# ═══════════════════════════════════════════════════════════════════════════ 产品件轨
# 下面三件量的才是 **产品代码**：`ascend/kernels/gemm_kernel.py` / `gemm_bwd_dw_kernel.py` /
# `letter_readout_kernel.py` → `*_asc.py` 正文 → `ascend_env` 路由与编译缓存（与
# `ascend/train_step.py` 消费的是同一条路）。三件都遵守两条共同纪律：
#   A. 判决前必过 `_route_key()` 假绿守卫（入口件在模具没开成时静默落 eager，rel=0.0 也能骗人）；
#   B. 每个形状都带"错参考"判别（方向/朝向/覆盖 vs 累加），只 rel 小不算过。
REL_TOL_LIN = 2e-5        # 与 reconcile/S_cpu_oracle.py 同尺（fp32 对 fp32，只容忍归约顺序差）
REL_TOL_DW = 2e-5         # 同上
REL_TOL_READOUT = 1e-5    # 与 reconcile/U_oracle.py 的 REL_TOL 同尺（纯搬运应逐位同）
WRONG_DIR_MIN = 1e-2      # 与 S_cpu_oracle.py 同尺：错方向必须差出这个量级，否则"通过"可能只是碰巧
DISC_MIN = 0.5            # 方向判别下限（与 §7/§8 两条 intrinsic case 同尺：错参考是"全表级"的）
#: 反向"累加 vs 覆盖"判别专用下限：这条错只落在重复行上，量级天然比"全表级"错小。
#: 实测（host cpu-alias 轨，正参考恒 rel=0.0）：dup-heavy 子例 0.46 / odd-dim 0.39 /
#: dup-dominant 子例 0.8 ⇒ 取 0.3 作下限（留 1.3x 余量），判别比仍 ≥1e4。
DISC_MIN_SCATTER = 0.3

#: add_ln 产品件前向判据（**逐字沿用 §7 `num_addln_intrinsic` 的宿主机仿真口径，一字未改**）：
#:   r_h：正确=0.0（单次 fp32 加，与参考逐位同）/ 残差路降 fp16=2.1e-4 ⇒ 1e-5 精确判别"不许降位宽"；
#:   r_y：正确=1.5e-5 / 方差错用无偏=7.4e-4 ⇒ 3e-4 判别"有偏"口径；
#:   有偏/无偏分离差须 > 2e-4（原拟 5e-3 判别不了，已收紧）。
ADDLN_REL_H = 1e-5
ADDLN_REL_Y = 3e-4
ADDLN_DISC_UNB = 2e-4
ADDLN_EPS = 1e-5

#: (m, n, k, act, 说明)——m 是产品件的动态维，故意含不整除 BLOCK_M(=16) 的尾块与批 1
LIN_SHAPES = (
    (33, 64, 32, "none", "尾块 m=33 不整除 bm=16 ⇒ 夹址填装 + 行号守卫真生效（507015 那条防线）"),
    (8, 64, 32, "none", "同 (n,k,act) 换批 ⇒ 必须复用同一份产物（m 不在缓存键里）"),
    (64, 64, 64, "none", "方阵 ⇒ 可做 W 不翻（A 乘 W 不转置）转置朝向判别"),
    (16, 96, 48, "relu", "K 非对齐档（k=48⇒bk=16、nk=3）⇒ 本格唯一编成**多突发跨步 GM→UB(MTE2) "
                         "DMA** 的档（bk==k 的档被 plan_single_row 折叠成单发连续、步长槽不参与）。"
                         "真机 P1-1q 定位点：错值/NaN 出在装填面，非 relu 出口、非 n 向多块；"
                         "host cpu/alias 两轨均不复现。同型的 k=96（bk=32）在归档生成码里另占 2 枚"
                         "（fp32/fp16，训练步 dA 必经），本本格未覆盖——见 reconcile/U2_matrix_k48.md。"
                         "注：本格是否真编出跨步取决于 plan 的 bk 取法；若 plan 改选整行连续"
                         "（_pick_bk_ascend，P1-1p 在途）则 bk==k⇒单发、本格退化为对照档，"
                         "届时步长语义须用 reconcile/U2_probe_k48_dma_semantics.py --dev npu"
                         "（强钉 bk，不经 plan）才测得到"),
    (1, 64, 64, "relu", "批大小 1：单行也要走完整块口径"),
)
#: (m, n, k, 说明)——出口 GW 是 (n,k)；m 是动态 token 轴
DW_SHAPES = (
    (33, 64, 32, "尾块 m=33 不整除 tc=16 ⇒ 行号守卫真生效"),
    (129, 64, 64, "m>tc 多轮累加（129=8×16+1）"),
    (64, 32, 64, "n≠k ⇒ 出口 (N,K) 朝向唯一，可判 Aᵀ@dY 反朝向"),
    (8, 96, 48, "小批 + 多出口块（bn=32/bk=16 ⇒ tiles=9 > NUM_BLOCKS=8 ⇒ 摊开累加）"),
)
#: (行数 total, 行宽 dim, 行号表, 说明)——形状口径照 reconcile/U_oracle.py cases()：
#: 乱序、重复行号（同行多次命中）、尾块（picked 不整除 GATHER_ROWS_PER_BLOCK=8）、行宽非 32B 整倍
READOUT_SHAPES = (
    (128, 64, [7, 0, 3, 3, 11, 5, 1, 127, 64, 64, 64, 2, 91, 33],
     "picked=14 ⇒ 2 块含尾块（6<RPB=8）+ 三重命中 + 首/末行"),
    (12, 26, [7, 0, 3, 3, 11, 5, 1],
     "U 的 odd-dim 口径：行宽 26（非 32B 整倍）+ picked=7<RPB=8 + 重复行号"),
    (8, 32, [5], "picked=1 ⇒ 网格 1 块、7 格守卫全生效"),
    (9, 32, [2, 2, 2, 8, 0, 0, 5, 5, 5, 1, 2, 8],
     "dup-heavy：同行最多命中 4 次 ⇒ scatter 必须累加，不许覆盖"),
    (4, 32, [1, 1, 1, 1, 3, 3],
     "dup-dominant：4 行表 6 次抽行 ⇒ 覆盖式错参考的偏离被推到最大（判别的标定子例）"),
)
#: (行数 rows, 行宽 dim, 说明)——形状档**沿用 §7 addln 的档**（rows=64、dim=512；bm 由 `_pick_bm`
#: 在 ascend 侧收缩为 16）。行数 row 不进编译键（add_ln 缓存键只含 dim/eps/bm）⇒ 两档复用同一份产物
#: 是**产品行为**（由 reuse 判别）；尾块档（rows=33 不整除 bm=16）专激励产品件的 `if row < rows`
#: 行号守卫——常量形状的 intrinsic case 永远量不到这条。
ADDLN_SHAPES = (
    (64, 512, "现 addln 档：rows=64=4×bm(bm=16)、dim=512（与 num_addln_intrinsic 同形）"),
    (33, 512, "尾块：rows=33 不整除 bm=16 ⇒ 行号守卫真生效"),
)


class _LaunchSkip(RuntimeError):
    """card 轨 `--no-run`：编译已成、发射被桩掉时抛它（与 compile_all_asc.py 同一条隔离纪律）。"""


def _npu_present():
    import torch
    return hasattr(torch, "npu") and bool(torch.npu.is_available())


def _track():
    """解析本次跑的轨：card＝生产 ascend 路由（有卡真发射，无卡只判编译）；cpu-alias＝host 强跑生产正文。"""
    if ARGS.track == "card":
        return "card"
    if ARGS.track == "cpu-alias":
        return "cpu-alias"
    return "card" if _npu_present() else "cpu-alias"


def _route_key(sub):
    """假绿守卫（**load-bearing**）：产品件必须真在编译缓存里留下 `ascend|...` 键。

    入口件（`*_kernel.py`）在 plan 拒形状 / 后端不可用 / 方言别名没生效时会静默落 torch eager 并
    照样交出正确的数——本波探针实测：只改 `sys.modules["tilelang.ascend.language"]` 而没改父包属性
    ⇒ `dialect("ascend")` 仍取到真 ascend 方言 ⇒ 编译失败 ⇒ 落 eager ⇒ rel=0.0 且
    `compiled_keys()==()`。没有这道守卫，产品件 case 会退化成"量 torch 的假绿件"。
    """
    from ascend.kernels import ascend_env
    keys = [k for k in ascend_env.compiled_keys() if k.startswith("ascend|") and sub in k]
    if not keys:
        raise AssertionError(
            f"假绿守卫：compiled_keys 无 ascend|*{sub}* ⇒ 本趟走的是 torch eager，不是产品件"
            f"（方言别名未生效 / plan 拒形状 / 后端不可用）；"
            f"blockers={list(ascend_env.blockers().items())[:2]}")
    return keys[0]


def _install_launch_skip():
    """把 `tilelang.compile` 换成"真编译 + 调用即跳发射"的替身（card 轨无设备/只判编译时用）。"""
    import tilelang as _tl
    real = _tl.compile
    box = {"src": 0}

    def _c(*a, **k):
        kern = real(*a, **k)                     # 真编译：失败原样抛，由 get_compiled 记 blocker
        box["src"] = len(str(kern.get_kernel_source())) if hasattr(kern, "get_kernel_source") else 0

        class _NoLaunch:
            def __call__(self, *aa, **kk):
                raise _LaunchSkip("card 轨只判编译，不发射")

            def __getattr__(self, n):
                return getattr(kern, n)          # 其余属性照真产物（本工装只用 __call__）

        return _NoLaunch()

    _tl.compile = _c
    return (lambda: setattr(_tl, "compile", real)), box


def _prod_setup():
    """按 track 装配运行面，返回 `(device, restore, skip_launch, box)`；box['src'] 是最后一次真编译产物源码字节数。

    - `cpu-alias`（host 无卡）：`tilelang.ascend.language` 这个名字**两处一起**指向 cpu 方言模块
      （`sys.modules` + 父包属性，只改一处不生效），并把 `ascend_env.compile_kwargs` 翻成 c 后端三件套
      ⇒ 递进去的 target 仍是 "ascend"、路由/缓存/键名全是产品件的，正文是生产 ascend 正文。
      手法出处 `reconcile/U_oracle.py::run_prod`（本波修正版，见模块 docstring ②）。
    - `card` + 有 npu：什么都不桩，真编真发射（下窗就是这个形态）。
    - `card` + 无 npu（容器有 bisheng 无设备）：只桩"设备-目标匹配门"三量，并**强制只判编译**
      ——张量是 cpu 的，压根发不出去；这一档的产出是 `Y-<case>-ASC-COMPILE-PASS`，不是数值判决。
    """
    import types
    import tilelang as _tl
    from ascend.kernels import ascend_env

    dev = "npu" if _npu_present() else "cpu"
    tk = _track()
    skip = (tk == "card") and (dev == "cpu" or ARGS.no_run)
    saved_env = {n: getattr(ascend_env, n) for n in
                 ("compile_kwargs", "backend_available", "_npu_present", "active_backend")}
    saved_alias = None
    if tk == "cpu-alias":
        import tilelang.cpu.language as tcpu
        pkg = sys.modules.get("tilelang.ascend")
        if pkg is None:
            pkg = types.ModuleType("tilelang.ascend")
            pkg.__path__ = []                                  # 伪装成包，够 import 机制用
            sys.modules["tilelang.ascend"] = pkg
            setattr(_tl, "ascend", pkg)
        saved_alias = (sys.modules.get("tilelang.ascend.language"), getattr(pkg, "language", None))
        sys.modules["tilelang.ascend.language"] = tcpu
        pkg.language = tcpu
        ascend_env.compile_kwargs = lambda t: {"target": "c", "target_host": "c",
                                               "execution_backend": "cython"}
    # 两档共同的"设备门"桩：card 无设备时不桩就直接被判 eager（want npu ≠ cpu）
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG

    restore_compile = box = None
    if skip:
        restore_compile, box = _install_launch_skip()

    def restore():
        if saved_alias is not None:
            mod, attr = saved_alias
            if mod is None:
                sys.modules.pop("tilelang.ascend.language", None)
            else:
                sys.modules["tilelang.ascend.language"] = mod
            if attr is None:
                try:
                    delattr(sys.modules["tilelang.ascend"], "language")
                except AttributeError:
                    pass
            else:
                sys.modules["tilelang.ascend"].language = attr
        for n, v in saved_env.items():
            setattr(ascend_env, n, v)
        if restore_compile is not None:
            restore_compile()

    ascend_env.reset()                    # 每件从零：compiled_keys/blockers 不许串件
    return dev, restore, skip, (box if box is not None else {"src": 0})


def _addln_cpu_alias_storage_shim():
    """cpu-alias 轨专用（**只服务 `prod_addln`**）：把 UB 落点 `alloc_shared`→`alloc_local`。

    机制（本波实测，探针 `reconcile/A4_probe_addln_compile.py`）：
      `add_ln_asc_impl` 是 **ascend-only 正文**，用 4 个 `T.alloc_shared`（scope=`shared.dyn`，
      tilelang/language/allocate.py:49）。c 后端的 `lower_device_kernel_launch`（src/transform/
      lower_device_kernel_launch.cc:173）只允许**一个** dynamic shared 分配：
      `InternalError: Only one dynamic shared memory allocation is allowed.` ⇒ 直连 cpu 方言编不过。
      gemm/dw 的产品正文靠 `gemm_asc._alloc_tile` **按方言分叉存储层**（cpu→alloc_local），add_ln
      件没有这层分叉（它只为昇腾写）。故在 cpu-alias 轨由本工装替它做同一件事。

    边界（如实声明，不夸大）：只换**存储层**（UB vs local），算式、形状、判据一字不动；cpu-alias
    轨本就只判"无卡时能拿到的最强数值证据"，ascend 发射层差异（真 UB）由 card 轨/容器编译判决负责。
    本 shim 只在 `prod_addln` 内、且只在 cpu-alias 轨安装；`_prod_setup().restore()` 会把它一并还原。
    """
    import types
    import tilelang.cpu.language as tcpu
    shim = types.ModuleType("tilelang.ascend.language")
    for nm in dir(tcpu):
        setattr(shim, nm, getattr(tcpu, nm))
    shim.alloc_shared = tcpu.alloc_local          # c 后端无 UB：shared.dyn → local（同 _alloc_tile 口径）
    shim.__tilelang_dialect__ = "cpu"
    sys.modules["tilelang.ascend.language"] = shim
    sys.modules["tilelang.ascend"].language = shim
    return shim


def _call(skippable, fn):
    """跑一次产品件调用；card 轨只判编译时把"发射被桩掉"收下来（编译已成、键已入缓存）。"""
    try:
        return fn(), False
    except _LaunchSkip:
        if not skippable:
            raise
        return None, True


def prod_linear():
    """产品件 `gemm_kernel.forward → gemm_asc.gemm_scalar_body`（P1-1f 改道后的标量体）。

    判据：① `rel < 2e-5`（S_cpu_oracle 同尺）②方阵子例做"不翻"朝向判别（差出 > 1e-2）
    ③relu 子例先自证"relu 分支真被激励"（前级确有负数）再判出口零负数
    ④尾块子例（m=33/8/1）激励夹址填装 + 行号守卫——这正是产品件从 cube 路改道后要防的那条
    ⑤同 (n,k,act) 换批不重编：`compile_count == distinct(n,k,act)`，证明 m 真的落在动态上界上
      （intrinsic case 是常量形状，永远量不到这条产品行为）。
    """
    import torch
    from ascend.kernels import ascend_env, gemm_kernel
    dev, restore, skip, box = _prod_setup()
    tk, worst, dirs, hit, seen = _track(), 0.0, None, 0, set()
    try:
        torch.manual_seed(0)
        for m, n, k, act, note in LIN_SHAPES:
            A = torch.randn(m, k, device=dev, dtype=torch.float32)
            W = torch.randn(n, k, device=dev, dtype=torch.float32)
            b = torch.randn(n, device=dev, dtype=torch.float32)
            got, only = _call(skip, lambda: gemm_kernel.forward(A, W, b, act, torch.float32, "ascend"))
            if only:
                seen.add(_route_key("gemm[ascend]")); continue
            pre = A @ W.T + b
            ref = pre.clamp_min(0.0) if act == "relu" else pre
            r = relerr(_np(got), _np(ref))
            assert r < REL_TOL_LIN, f"m{m}n{n}k{k}/{act} rel={r:.3e} ≥ {REL_TOL_LIN:g}（{note}）"
            assert float(np.linalg.norm(_np(ref))) > 0, f"m{m}n{n}k{k} 参考全零 ⇒ 空测"
            if act == "relu":
                neg_in = int((pre < 0).sum())
                assert neg_in > 0, f"m{m}n{n}k{k} relu 未被激励（前级无负数）"
                assert int((got < 0).sum()) == 0, f"m{m}n{n}k{k} relu 出口仍有负数 ⇒ act_mode 未生效"
                hit += neg_in
            if m == n == k:                     # 方阵才谈得上"A@W 不翻"这个错方向
                wrong = relerr(_np(got), _np(A @ W + b))
                assert wrong > WRONG_DIR_MIN, (f"m{m} 朝向判别失败：与 A@W（不翻）的距离 "
                                               f"{wrong:.3e} ≤ {WRONG_DIR_MIN:g}")
                dirs = wrong
            seen.add(_route_key("gemm[ascend]"))
            worst = max(worst, r)
        if skip:
            assert len(seen) >= 1, "一件都没编成"
            return (f"Y-linear_prod-ASC-COMPILE-PASS keys={len(seen)} src~{box['src']}B "
                    f"track=card(no-run) note=无设备⇒只判编译")
        want = len({(n, k, act) for _m, n, k, act, _s in LIN_SHAPES})
        got_n = ascend_env.compile_count()
        assert got_n == want, f"换批复用失败：compile_count={got_n} ≠ distinct(n,k,act)={want}"
        return (f"Y-linear_prod-PROD-PASS rel={worst:.2e} dir_disc={dirs:.2e} reuse={got_n}/{want} "
                f"relu_neg_clipped={hit} shapes={len(LIN_SHAPES)} track={tk}")
    finally:
        restore()


def prod_dw():
    """产品件 `gemm_bwd_dw_kernel.backward → gemm_bwd_dw_asc.dw_scalar_body`（dW = dYᵀ @ A）。

    判据：① `rel < 2e-5`；②朝向判别——把出口按 `Aᵀ@dY`（即 refᵀ 的内存序）比对，必须差出 > 1e-2
    （两参考 numel 相同，n≠k 时形状也相同 ⇒ 这一条对任何 (n,k) 都成立，比"方阵不翻"更严）；
    ③尾块（m=33）与多轮累加（m=129=8×16+1）都真激励到行号守卫；④同 (n,k) 换批不重编。
    """
    import torch
    from ascend.kernels import ascend_env, gemm_bwd_dw_kernel
    dev, restore, skip, box = _prod_setup()
    tk, worst, dirs, seen = _track(), 0.0, None, set()
    try:
        torch.manual_seed(1)
        for m, n, k, note in DW_SHAPES:
            dY = torch.randn(m, n, device=dev, dtype=torch.float32)
            X = torch.randn(m, k, device=dev, dtype=torch.float32)
            got, only = _call(skip, lambda: gemm_bwd_dw_kernel.backward(
                dY, X, out_dtype=torch.float32, target="ascend"))
            if only:
                seen.add(_route_key("gemm_dw[ascend]")); continue
            ref = dY.T @ X
            r = relerr(_np(got), _np(ref))
            assert r < REL_TOL_DW, f"m{m}n{n}k{k} rel={r:.3e} ≥ {REL_TOL_DW:g}（{note}）"
            wrong = relerr(_np(got), _np((X.T @ dY).contiguous()))     # 反朝向（=refᵀ 的内存序）
            assert wrong > WRONG_DIR_MIN, (f"m{m}n{n}k{k} dW 朝向判别失败：与 Aᵀ@dY 的距离 "
                                           f"{wrong:.3e} ≤ {WRONG_DIR_MIN:g}")
            dirs = wrong if dirs is None else min(dirs, wrong)
            seen.add(_route_key("gemm_dw[ascend]"))
            worst = max(worst, r)
        if skip:
            assert len(seen) >= 1, "一件都没编成"
            return (f"Y-dw_prod-ASC-COMPILE-PASS keys={len(seen)} src~{box['src']}B "
                    f"track=card(no-run) note=无设备⇒只判编译")
        want = len({(n, k) for _m, n, k, _s in DW_SHAPES})
        got_n = ascend_env.compile_count()
        assert got_n == want, f"换批不重编失败：compile_count={got_n} ≠ distinct(n,k)={want}"
        return (f"Y-dw_prod-PROD-PASS rel={worst:.2e} dir_disc={dirs:.2e} reuse={got_n}/{want} "
                f"shapes={len(DW_SHAPES)} track={tk}")
    finally:
        restore()


def prod_readout():
    """产品件 `letter_readout_kernel.forward/backward`（纯 gather + scatter_add，P1-1h 收敛后正文）。

    判据口径照 `reconcile/U_oracle.py`：
      ① 前向 = `out[i] = rows[ids[i]]`，纯搬运应逐位同 ⇒ `rel < 1e-5`（另报 bitexact）；
      ② 前向方向判别：忽略行号表、按顺序取行（`rows[arange(picked) % total]`）必须差出 > 0.5
         ——这一条防的是"动态散取被下射成顺序搬运"那种看着像的错；
      ③ 反向 = 按同一张行号表**加**回原表，与 `index_add_` 尺子 `rel < 1e-5`；
      ④ 反向口径判别：与"覆盖式写回"（同一行只留最后一次）必须差出 > DISC_MIN_SCATTER=0.3
         （行号有重复时才有效；这条错只落在重复行上，量级天然小于"全表级"错，故单列下限并给实测）
         ——这一条防的是"add 写成 put"，dup-heavy 子例（同行命中 4 次）专打它；
      ⑤ 形状覆盖尾块（picked<RPB=8）、行宽非 32B 整倍（dim=26）、picked=1、重复行号；
      ⑥ 前向/反向都过 `_route_key` 守卫（gather 与 scatter_add 是两份正文，缺一个都不算过）。
    """
    import torch
    from ascend.kernels import ascend_env, letter_readout_kernel
    dev, restore, skip, box = _prod_setup()
    tk, worst, gfwd, gback, seen = _track(), 0.0, None, None, set()
    try:
        rng = np.random.default_rng(5)
        for total, dim, id_list, note in READOUT_SHAPES:
            ids = torch.tensor(id_list, dtype=torch.int64, device=dev)
            rows = torch.from_numpy(rng.standard_normal((total, dim)).astype(np.float32)).to(dev)
            picked = len(id_list)
            got, only = _call(skip, lambda: letter_readout_kernel.forward(
                rows, ids, torch.float32, "ascend"))
            if only:
                # 只判编译也必须把**反向那份正文**真编一次：gather 与 scatter_add 是两份独立
                # 正文，只调 forward 的话 scatter_add 键永远不存在（本波容器实测踩到，守卫如实报 FAIL）
                dy0 = torch.from_numpy(rng.standard_normal((picked, dim)).astype(np.float32)).to(dev)
                _call(skip, lambda: letter_readout_kernel.backward(total, ids, dy0, "ascend"))
                seen.add(_route_key("gather[ascend]")); seen.add(_route_key("scatter_add[ascend]"))
                continue
            ref = rows[ids]
            r = relerr(_np(got), _np(ref))
            assert r < REL_TOL_READOUT, f"total{total} dim{dim} 前向 rel={r:.3e} ≥ " \
                                        f"{REL_TOL_READOUT:g}（{note}）"
            seq_ref = rows[torch.arange(picked, device=dev) % total]     # 忽略行号表、顺序取行
            bad = relerr(_np(got), _np(seq_ref))
            assert bad > DISC_MIN, f"散取方向判别失败（与顺序取行的距离 {bad:.3e} ≤ {DISC_MIN:g}）"
            bitexact = bool(torch.equal(got, ref))

            dy = torch.from_numpy(rng.standard_normal((picked, dim)).astype(np.float32)).to(dev)
            gb, _ = _call(skip, lambda: letter_readout_kernel.backward(total, ids, dy, "ascend"))
            acc = torch.zeros(total, dim, device=dev, dtype=torch.float32)
            acc.index_add_(0, ids, dy)
            r2 = relerr(_np(gb), _np(acc))
            assert r2 < REL_TOL_READOUT, f"total{total} dim{dim} 反向 rel={r2:.3e}（{note}）"
            over = torch.zeros(total, dim, device=dev, dtype=torch.float32)
            for i, ii in enumerate(id_list):                 # 覆盖式（add 写成 put 的那条错路）
                over[ii] = dy[i]
            if len(set(id_list)) != picked:                   # 有重复行号才有判别力
                b2 = relerr(_np(gb), _np(over))
                assert b2 > DISC_MIN_SCATTER, (f"累加/覆盖口径判别失败（dup 子例，距离 "
                                              f"{b2:.3e} ≤ {DISC_MIN_SCATTER:g}）")
            seen.add(_route_key("gather[ascend]"))
            seen.add(_route_key("scatter_add[ascend]"))
            worst = max(worst, r, r2)
            gfwd = r if gfwd is None else max(gfwd, r)
            gback = r2 if gback is None else max(gback, r2)
        if skip:
            assert len(seen) >= 2, f"gather/scatter_add 两份正文没都编成（seen={seen}）"
            return (f"Y-readout_prod-ASC-COMPILE-PASS keys={len(seen)} src~{box['src']}B "
                    f"track=card(no-run) note=无设备⇒只判编译")
        want = len({dim for _t, dim, _i, _s in READOUT_SHAPES}) * 2
        got_n = ascend_env.compile_count()
        assert got_n == want, f"编译次数对不上：compile_count={got_n} ≠ distinct_dim×2={want}"
        return (f"Y-readout_prod-PROD-PASS rel={worst:.2e} fwd={gfwd:.2e} bwd={gback:.2e} "
                f"bitexact={bitexact!s} reuse={got_n}/{want} shapes={len(READOUT_SHAPES)} track={tk}")
    finally:
        restore()


def prod_addln():
    """产品件 `add_ln_kernel.forward → add_ln_asc.add_ln_asc_impl`（残差相加 + 层内整形）。

    判据**逐字沿用 §7 `num_addln_intrinsic` 的宿主机仿真口径（阈值一字未改）**，只把正文从
    attempts/B 内联 DSL 换成产品接口件：
      ① `r_h < 1e-5`：Hout（残差流）全程 fp32 不降位宽（降到 fp16 即 ~2e-4，精确判别）；
      ② `r_y < 3e-4`：Y 有偏方差口径（仿真底噪 1.5e-5；Y 全零则 rel≈1 ⇒ 顺带验就地写回）；
      ③ 有偏/无偏分离判别：`r_y < r_unb - 2e-4`（防"碰巧对"，错口径=无偏方差 ~7.4e-4）；
      ④ 形状覆盖现 addln 档（rows=64,dim=512）+ 尾块（rows=33 ⇒ 产品件 `if row < rows` 行号守卫）；
      ⑤ 判决前过 `_route_key("add_ln")` 假绿守卫（入口件落 eager 时 rel 也会是 0.0）；
      ⑥ 两档同 dim ⇒ 断言 `compile_count == distinct(dim)`，证"行数不进缓存键、换批复用同一产物"
         （产品行为；常量形状的 intrinsic case 量不到）。

    与 §7 的分别：那条量的是 attempts/B 内联 DSL（`track=performance`，**不是产品件**）；旧 case
    保留不删（并存迁移，见其 docstring 的 `[note]`），本件才是 add_ln 的**产品件判决行**。
    """
    import torch
    from ascend.kernels import ascend_env, add_ln_kernel
    dev, restore, skip, box = _prod_setup()
    tk = _track()
    if tk == "cpu-alias":
        _addln_cpu_alias_storage_shim()       # add_ln 是 ascend-only 正文，host 需把 UB 映射到 local
    worst_h, worst_y, min_unb, seen = 0.0, 0.0, None, set()
    try:
        torch.manual_seed(0)
        for rows, dim, note in ADDLN_SHAPES:
            X = torch.randn(rows, dim, device=dev, dtype=torch.float32)
            Res = torch.randn(rows, dim, device=dev, dtype=torch.float32) + 4.0   # 大共模/小波动
            G = torch.randn(dim, device=dev, dtype=torch.float32)
            Bt = torch.randn(dim, device=dev, dtype=torch.float32)
            res, only = _call(skip, lambda: add_ln_kernel.forward(
                X, Res, G, Bt, ADDLN_EPS, torch.float16, "ascend"))       # -> (y=fp16, h=fp32)
            if only:
                seen.add(_route_key("add_ln"))
                continue
            y, h = res
            # 参考：入参先按 kernel 实际位宽取，再全程 fp32；算式顺序照产品正文（有偏方差、三趟式）
            xv, rv = _np(X).astype(np.float32), _np(Res).astype(np.float32)
            gv, bv = _np(G).astype(np.float32), _np(Bt).astype(np.float32)
            v = (xv + rv).astype(np.float32)                       # Hout：fp32 残差流
            mu = v.mean(axis=1, keepdims=True)
            dv = (v - mu).astype(np.float32)
            var_b = (dv * dv).mean(axis=1, keepdims=True).astype(np.float32)   # 有偏（kernel 口径）
            var_u = (dv * dv).sum(axis=1, keepdims=True) / (dim - 1)           # 无偏（错误口径，仅判别）
            ref_h = v
            ref_y = ((dv * (1.0 / np.sqrt(var_b + ADDLN_EPS)).astype(np.float32)) * gv + bv).astype(np.float16)
            alt_y = ((dv * (1.0 / np.sqrt(var_u + ADDLN_EPS)).astype(np.float32)) * gv + bv).astype(np.float16)
            got_y, got_h = _np(y).astype(np.float32), _np(h).astype(np.float32)
            r_h = relerr(got_h, ref_h.astype(np.float32))
            r_y = relerr(got_y, ref_y.astype(np.float32))
            r_unb = relerr(got_y, alt_y.astype(np.float32))
            assert r_h < ADDLN_REL_H, (f"rows{rows}dim{dim} Hout rel={r_h:.3e} ≥ {ADDLN_REL_H:g}"
                                       f"（fp32 残差路应逐位同；~2e-4 即降了位宽）（{note}）")
            assert r_y < ADDLN_REL_Y, (f"rows{rows}dim{dim} Y rel={r_y:.3e} ≥ {ADDLN_REL_Y:g}"
                                       f"（仿真底噪 1.5e-5；若 Y 全零则 rel≈1 ⇒ 就地写回未生效）（{note}）")
            assert r_y < r_unb - ADDLN_DISC_UNB, (f"rows{rows}dim{dim} 有偏/无偏口径判别失败 "
                                                  f"r_y={r_y:.3e} r_unb={r_unb:.3e}（{note}）")
            seen.add(_route_key("add_ln"))
            worst_h = max(worst_h, r_h)
            worst_y = max(worst_y, r_y)
            min_unb = r_unb if min_unb is None else min(min_unb, r_unb)
        if skip:
            assert len(seen) >= 1, "一件都没编成"
            return (f"Y-addln_prod-ASC-COMPILE-PASS keys={len(seen)} key={sorted(seen)[0]} "
                    f"src~{box['src']}B track=card(no-run) note=无设备⇒只判编译")
        want = len({dim for _r, dim, _n in ADDLN_SHAPES})
        got_n = ascend_env.compile_count()
        assert got_n == want, f"换批复用失败：compile_count={got_n} ≠ distinct(dim)={want}"
        return (f"Y-addln_prod-PROD-PASS rel_y={worst_y:.2e} rel_h={worst_h:.2e} "
                f"unb_disc={min_unb:.2e} reuse={got_n}/{want} shapes={len(ADDLN_SHAPES)} track={tk}")
    finally:
        restore()


# ── 注册表：产品件在前（下窗要量的就是它们），performance-track 在后（对照/考古）────────
PROD_TARGETS = {"linear_prod": prod_linear, "dw_prod": prod_dw, "readout_prod": prod_readout,
                "addln_prod": prod_addln}
PERF_TARGETS = {"gemm_direct": num_gemm_direct, "gemm_l1_intrinsic": num_gemm_l1_intrinsic,
                "gdn": num_gdn, "dw_intrinsic": num_dw_intrinsic, "rope": num_rope,
                "attnsw": num_attnsw, "addln_intrinsic": num_addln_intrinsic,
                "readout_intrinsic": num_readout_intrinsic}
#: 旧名 → 新名。`fixup_card.sh` / `bundle_ondemand.sh` 不在本波白名单里，仍在用旧名点 case，
#: 故保留别名（打印一行 note 后转发到新实现，判决行本身不变格式）。
DEPRECATED = {"gemm_l1": "gemm_l1_intrinsic", "dw": "dw_intrinsic",
              "addln": "addln_intrinsic", "readout": "readout_intrinsic"}
TARGETS = {**PROD_TARGETS, **PERF_TARGETS}

if __name__ == "__main__":
    names = [ARGS.only] if ARGS.only != "all" else list(TARGETS)
    rc = 0
    for nm in names:
        canon = DEPRECATED.get(nm)
        if canon:
            print(f"[note] `--only {nm}` 是弃用别名 ⇒ {canon}（attempts 内联正文，"
                  f"performance-track，非产品件）", file=sys.stderr, flush=True)
            nm = canon
        fn = TARGETS.get(nm)
        if fn is None:
            print(f"NUM-{nm}-FAIL unknown target；可用：{','.join(TARGETS)}", flush=True)
            rc = 1
            continue
        try:
            print(fn(), flush=True)
        except Exception as e:  # noqa: BLE001
            prefix = f"Y-{nm}-PROD-FAIL" if nm in PROD_TARGETS else f"NUM-{nm}-FAIL"
            print(f"{prefix} {type(e).__name__}: {str(e)[:300]}", flush=True)
            rc = 1
    sys.exit(rc)
