#!/bin/bash
# window5_ta_card.sh — 第五窗（tilelang-ascend / ascendc_pto，B 路线）卡上判决件
# ============================================================================
# 目的：卡上**免编**——解包 J3 预构建的 tilelang-ascend 可运行包，真跑 GDN 六件（chunk fwd 族）
#       数值对拍 + 性能（p50/p95，预热 + 多 iter + sync 夹计），gemm 同采（自带 1024³ + 512³ fp16/fp32）。
# 用法：宿主 run_window5_local.sh 上传本脚本 + /tmp/ta_pkg.tar.gz 后 setsid 起跑；
#       本脚本内联驱动（heredoc→/tmp/j3_ta_run.py），单文件自足，不再依赖第二份上传。
#
# ── 环境与差异风险（卡 CANN 8.5.2 vs 构建容器 8.5.0）─────────────────────────
#   预编译 .so（libtilelang/libtilelang_module/libtvm）由容器 CANN 8.5.0 链出，卡上 8.5.2 加载。
#   风险面：① 动态库 NEEDED/RUNPATH（libtilelang.so RUNPATH 硬编码 /tlasc/build/tvm → 用
#   LD_LIBRARY_PATH 兜底）；② glibc/GLIBCXX（容器 glibc2.35/GLIBC_2.34、GLIBCXX_3.4.29）；
#   ③ bisheng 头文件 ABI（编译期）。
#   10 分钟止损方案：本脚本 [0] 先做**单件 cumsum 编译快检**（≤300s）——
#     · 快检编译 OK  ⇒ ABI 兼容，继续全量（[1]/[2]/[3]）。
#     · 快检 compile-error / wrong-tilelang / 卡上意外 no-device ⇒ TA-ABORT reason=cann-abi，
#       打印原始报错后**立即停**，不烧 17 分钟。
#   · 若确证 ABI 不兼容：唯一修复是卡上重编（≈17.1min / ≈¥6）——本窗**不做**，如实报 TA-ABORT，
#     交编排者定夺（这正是 B 路线要证的"可迁移包是否免编"的负例出口）。
#
# ── 体积 / 上传（估）───────────────────────────────────────────────────────
#   ta_pkg.tar.gz = 36.5 MB（sha256 d5607ec6…；解包 ~118 MB / 1996 文件）。
#   上传：SSH ≥1 MB/s ⇒ ≤40s；≥0.3 MB/s ⇒ ≤2min。宿主以 REMOTE_BYTES + sha 核对完整性。
#
# 判决行前缀 TA-*（宿主回读）；性能行只标 track=ta-ascendc，**不出结论**（结论归编排者）。
# ============================================================================
set -u

TARPATH=/tmp/ta_pkg.tar.gz
UNP=/tmp/ta_unpack
DRIVER=/tmp/j3_ta_run.py
export TA_ROOT=$UNP/ta_pkg
export TILELANG_CACHE_DIR=/tmp/ta_w5_cache
EXPECT_SHA=d5607ec6f9c321c9652fa7d25f234d2e17a074cf308938fa2c4b522830b02172

echo "=== [0] 解包 + env + CANN/ABI 快检 ==="

# ---- tar 完整性核对 ----
BYTES=$(wc -c < "$TARPATH" 2>/dev/null || echo 0)
SHA=$(sha256sum "$TARPATH" 2>/dev/null | cut -d' ' -f1)
echo "TA0-TAR file=$TARPATH bytes=$BYTES sha256=${SHA:-none}"
if [ "$SHA" != "$EXPECT_SHA" ]; then
  echo "TA-ABORT reason=tar-sha expect=$EXPECT_SHA got=${SHA:-none}"; echo "TA-DONE"; exit 1
fi

rm -rf "$UNP"; mkdir -p "$UNP"
tar xzf "$TARPATH" -C "$UNP" || { echo "TA-ABORT reason=untar"; echo "TA-DONE"; exit 1; }
[ -f "$TA_ROOT/tilelang/__init__.py" ] \
  || { echo "TA-ABORT reason=layout missing=$TA_ROOT/tilelang/__init__.py"; echo "TA-DONE"; exit 1; }
echo "TA0-UNPACK root=$TA_ROOT files=$(find "$TA_ROOT" -type f 2>/dev/null | wc -l)"

# ---- CANN env（优先卡上 8.5.2；回退通用 prefix）----
# set_env.sh 会引用未定义变量，故 source 期间临时关 set -u
set +u
for envf in /usr/local/Ascend/cann-8.5.2/set_env.sh /usr/local/Ascend/ascend-toolkit/set_env.sh; do
  [ -f "$envf" ] && { source "$envf"; echo "TA0-CANN-env $envf"; break; }
done
set -u
: "${ASCEND_HOME_PATH:=}"
: "${LD_LIBRARY_PATH:=}"
: "${PYTHONPATH:=}"
export ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-$(ls -d /usr/local/Ascend/ascend-toolkit/latest 2>/dev/null)}
export ASCEND_NPU_ARCH=dav-2201
export TL_PLATFORM=A2
BISHENG_BIN=$(command -v bisheng || echo none)
export BISHENG_HOME=$(dirname "$(dirname "$BISHENG_BIN")" 2>/dev/null)
echo "TA0-ENV ASCEND_HOME_PATH=$ASCEND_HOME_PATH arch=$ASCEND_NPU_ARCH bisheng=$BISHENG_BIN"

# ---- python 运行时依赖（core 硬校验；optional best-effort，缺亦不 abort）----
# core = tilelang 导入 + JIT 编译本路径实测所需（容器以 core 子集跑通六件编译）；
# optional = upstream requirements.txt 里本路径未触达的条目，best-effort 补，缺只 WARN。
CORE="Cython,cloudpickle,ml_dtypes,psutil,tqdm,typing_extensions,numpy,torch"
OPT="attrs,decorator,scipy,tornado,yaml,pybind11"
if ! python3 -c "import $CORE" 2>/tmp/ta_deps_missing.log; then
  echo "TA0-DEPS core 缺失，pip 安装中（best-effort）"
  pip install -q Cython cloudpickle ml_dtypes psutil tqdm typing_extensions numpy 2>&1 | tail -3 \
    || echo "TA0-DEPS-WARN pip 失败，见 /tmp/ta_deps_missing.log"
fi
python3 -c "import $CORE" \
  || { echo "TA-ABORT reason=py-deps"; sed -n '1,6p' /tmp/ta_deps_missing.log; echo "TA-DONE"; exit 1; }
python3 -c "import $OPT" 2>/dev/null \
  || { pip install -q attrs decorator scipy tornado pyyaml pybind11 2>&1 | tail -2; \
       python3 -c "import $OPT" 2>/dev/null && echo "TA0-DEPS-OPT ok" || echo "TA0-DEPS-OPT-WARN 缺 optional（本路径不需，继续）"; }

# ---- PYTHONPATH / 动态库 rpath 兜底 ----
export PYTHONPATH="$TA_ROOT:$TA_ROOT/3rdparty/tvm/python"
export LD_LIBRARY_PATH="$TA_ROOT/build/tvm:$TA_ROOT/build:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

# ---- 内联驱动（单文件自足）----
cat > "$DRIVER" <<'PYEOF'
#!/usr/bin/env python3
"""J3 窗五卡上驱动 —— GDN 六件（chunk 前向族）真跑对拍 + 性能；gemm 同采。

不占卡验证（本文件在容器里以「无设备形态」跑 ⇒ 必须显式 SKIP reason=no-device）。

判决行契约（供 run_window5_local.sh grep）：
  TA-DEVICE   torch=... torch_npu=... npu_available=...
  TA-REFCHOICE ref=... reason=...
  TA-GDN-<name> PASS rel=<v> ref=<ref> p50_ms=<x> p95_ms=<y> dtype=fp16 iters=<n> warmup=<w> src_bytes=<b> track=ta-ascendc
  TA-GDN-<name> SKIP reason=no-device compile=<ok|fail> src_bytes=<b>
  TA-GEMM dtype=<dt> shape=<s> blk=<bm>/<bn>/<kl1> PASS rel=<v> ref=torch p50_ms=<x> p95_ms=<y> ... track=ta-ascendc
  TA-GEMM dtype=<dt> shape=<s> blk=<..> SKIP reason=no-device src_bytes=<b>
  TA-KERNELPROOF bisheng_calls=<n> cache_so=<m> verdict=<real-kernel|eager-suspect|compile-error>
  TA-SUMMARY six=<n>/6 gemm=<n> fail=<k> skip=<j> mode=<full|compile-only>

纪律：
 - 性能行只标 track=ta-ascendc，不出结论（结论归编排者）。
 - H2 的 torch_npu shim 只在「无 torch_npu」时启用（卡上有 torch_npu ⇒ 去 shim）。
 - 对拍参考＝tilelang-ascend 自带 ref_*（内部 fp32 累加、末转 fp16；跑在 npu）⇒ 满足「fp16 数值对 fp32 参考」。
 - 计时沿 E2 纪律：输入构造在计时区外，预热 + 多 iter + sync 前后夹计。
"""
import glob
import math
import os
import sys
import time

TA_ROOT = os.environ["TA_ROOT"]
TA_EX = os.path.join(TA_ROOT, "examples", "linear_attention_and_rnn")
GDN_DIR = os.path.join(TA_EX, "gdn")
CACHE = os.environ.get("TILELANG_CACHE_DIR", "")

ITERS = int(os.environ.get("TA_ITERS", "20"))
WARMUP = int(os.environ.get("TA_WARMUP", "3"))
REL_TOL = float(os.environ.get("TA_REL_TOL", "2e-3"))  # 我们既有口径（fp16）

# ---------------------------------------------------------------- device gate
import torch

TORCH_V = torch.__version__
HAVE_TORCH_NPU = hasattr(torch, "npu")
NPU_AVAIL = False
if HAVE_TORCH_NPU:
    try:
        NPU_AVAIL = bool(torch.npu.is_available())
    except Exception:  # noqa: BLE001
        NPU_AVAIL = False

if not HAVE_TORCH_NPU:
    # 仅无 torch_npu 时启用最小 shim（H2 配方：torch.device("npu")->cpu + 假 torch.npu）
    _orig_device = torch.device

    def _device(s=None, *a, **k):
        if isinstance(s, str) and s.startswith("npu"):
            return _orig_device("cpu")
        return _orig_device(s, *a, **k)

    torch.device = _device

    class _NpuShim:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def current_stream():
            class _S:
                npu_stream = 0
            return _S()

        @staticmethod
        def synchronize():
            pass

        @staticmethod
        def set_device(*a, **k):
            pass

        @staticmethod
        def get_device_name():
            return "j3-shim"

    torch.npu = _NpuShim
    print("[driver][shim] torch.npu ABSENT -> 装最小 shim（H2 配方）", flush=True)

MODE = "full" if NPU_AVAIL else "compile-only"
print(f"TA-DEVICE torch={TORCH_V} torch_npu={HAVE_TORCH_NPU} npu_available={NPU_AVAIL}", flush=True)
print(f"TA-MODE {MODE}", flush=True)

# ---------------------------------------------------------------- tilelang
import tilelang
import tilelang.language  # noqa: F401  (供 gemm example 段 exec 使用)

if not os.path.realpath(tilelang.__file__).startswith(os.path.realpath(TA_ROOT) + os.sep):
    print(f"TA-ABORT reason=wrong-tilelang file={tilelang.__file__} TA_ROOT={TA_ROOT}", flush=True)
    sys.exit(3)
print(f"TA-TILELANG file={tilelang.__file__} ver={getattr(tilelang, '__version__', '?')}", flush=True)

_orig_jit = tilelang.jit


def _jit(*a, **k):
    k.setdefault("target", "ascendc")
    k.setdefault("platform", os.environ.get("TA_PLATFORM", "A2"))
    return _orig_jit(*a, **k)


tilelang.jit = _jit

# bisheng 调用计数（真走内核自证）
import subprocess as _sp

_orig_run = _sp.run
BISHENG_RCS = []


def _run(cmd, *a, **k):
    r = _orig_run(cmd, *a, **k)
    if isinstance(cmd, (list, tuple)) and cmd and str(cmd[0]).endswith("bisheng"):
        BISHENG_RCS.append(r.returncode)
    return r


_sp.run = _run

# ---------------------------------------------------------------- 六件装载
sys.path.insert(0, TA_EX)
import importlib

M = {n: importlib.import_module("gdn." + n) for n in [
    "gdn_chunk_cumsum", "gdn_chunk_scaled_dot_kkt", "gdn_solve_tril",
    "gdn_wy_fast", "gdn_chunk_h", "gdn_chunk_o",
]}
import torch.nn.functional as F

print("TA-REFCHOICE ref=ta-repo-ref reason='自带 ref 内部 fp32 累加、末转 fp16（满足 fp16 对 fp32 参考）；"
      "跑在 npu（torch_npu ops）；系上游自测契约(rtol=atol=1e-5)'", flush=True)

B, H, L, DK, DV, C, BK, BV = 2, 32, 256, 64, 64, 32, 32, 32
CN = (L + C - 1) // C


def _sync():
    if NPU_AVAIL:
        torch.npu.synchronize()


def _pct(samples, q):
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    idx = (len(s) - 1) * q
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _measure(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    _sync()
    out = []
    for _ in range(iters):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def _maxrel(a, b):
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    den = b.abs().max().item()
    num = (a - b).abs().max().item()
    return (num / den if den > 0 else float("nan")), num


def _fmt(v):
    return f"{v:.4g}"


# ---------------------------------------------------------------- 每件：build / kernel / ref
# BUILD 逐一镜像各 example 自带 __main__ 的输入构造（含 seed=0），使对拍＝上游自测契约。
def build_cumsum():
    torch.manual_seed(0)
    g = torch.randn((B, H, L)).npu().to(torch.float)
    return (g,)


def build_kkt():
    torch.manual_seed(0)
    k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    beta = torch.randn((B, H, L)).npu().to(torch.float16)
    g = torch.randn((B, H, L)).npu().to(torch.float)
    return k, beta, g


def build_tril():
    torch.manual_seed(0)
    S = M["gdn_solve_tril"]
    k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    beta = torch.rand((B, H, L)).npu().to(torch.float16)
    g = torch.randn((B, H, L)).npu().to(torch.float)
    k = F.normalize(k, dim=-1, p=2)
    g = F.logsigmoid(g)
    g = S.ref_chunk_cumsum(g, C)
    a = S.ref_kkt(k, beta, g, C)
    return (a,)


def build_wy():
    torch.manual_seed(0)
    k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    v = torch.randn((B, H, L, DV)).npu().to(torch.float16)
    beta = torch.randn((B, H, L)).npu().to(torch.float16)
    g = torch.randn((B, H, L)).npu().to(torch.float)
    a = torch.randn((B, H, L, C)).npu().to(torch.float16)
    return k, v, beta, g, a


def build_chunk_h():
    torch.manual_seed(0)
    k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    w = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    u = torch.randn((B, H, L, DV)).npu().to(torch.float16)
    g = torch.randn((B, H, L)).npu().to(torch.float)
    g = F.logsigmoid(g)
    k, w = F.normalize(k, dim=-1, p=2), F.normalize(w, dim=-1, p=2)
    g = M["gdn_chunk_h"].ref_chunk_cumsum(g, C)
    return k, w, u, g


def build_chunk_o():
    torch.manual_seed(0)
    q = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    k = torch.randn((B, H, L, DK)).npu().to(torch.float16)
    v = torch.randn((B, H, L, DV)).npu().to(torch.float16)
    s = torch.randn((B, H, CN, DK, DV)).npu().to(torch.float16)
    g = torch.randn((B, H, L)).npu().to(torch.float)
    q, k = F.normalize(q, dim=-1, p=2), F.normalize(k, dim=-1, p=2)
    return q, k, v, s, g


BUILD = {"cumsum": build_cumsum, "kkt": build_kkt, "solve_tril": build_tril,
         "wy_fast": build_wy, "chunk_h": build_chunk_h, "chunk_o": build_chunk_o}

KERNEL = {
    "cumsum": lambda i: (M["gdn_chunk_cumsum"].chunk_cumsum(i[0], C),),
    "kkt": lambda i: (M["gdn_chunk_scaled_dot_kkt"].kkt(i[0], i[1], i[2], C, BK),),
    "solve_tril": lambda i: (M["gdn_solve_tril"].solve_tril(i[0]),),
    "wy_fast": lambda i: M["gdn_wy_fast"].wy_fast(i[0], i[1], i[2], i[3], i[4], C, BK, BV),
    "chunk_h": lambda i: M["gdn_chunk_h"].chunk_h(i[0], i[1], i[2], i[3], C, BK, BV),
    "chunk_o": lambda i: (M["gdn_chunk_o"].chunk_o(i[0], i[1], i[2], i[3], i[4], C, BK, BV),),
}

REF = {
    "cumsum": lambda i: (M["gdn_chunk_cumsum"].ref_chunk_cumsum(i[0], C),),
    "kkt": lambda i: (M["gdn_chunk_scaled_dot_kkt"].ref_kkt(i[0], i[1], i[2], C),),
    "solve_tril": lambda i: (M["gdn_solve_tril"].ref_solve_tril(i[0]),),
    "wy_fast": lambda i: M["gdn_wy_fast"].ref_wy_fast(i[0], i[1], i[2], i[3], i[4], C),
    "chunk_h": lambda i: M["gdn_chunk_h"].ref_chunk_h(i[0], i[1], i[2], i[3], C),
    "chunk_o": lambda i: (M["gdn_chunk_o"].ref_chunk_o(i[0], i[1], i[2], i[3], i[4], C),),
}

CK = {
    "cumsum": lambda: M["gdn_chunk_cumsum"].cumsum_ker(B, H, L, C),
    "kkt": lambda: M["gdn_chunk_scaled_dot_kkt"].kkt_ker(B, H, L, DK, C, BK),
    "solve_tril": lambda: M["gdn_solve_tril"].solve_tril_ker(B, H, L, C),
    "wy_fast": lambda: M["gdn_wy_fast"].wy_fast_ker(B, H, L, DK, DV, C, BK, BV),
    "chunk_h": lambda: M["gdn_chunk_h"].chunk_h_ker(B, H, L, DK, DV, C, BK, BV),
    "chunk_o": lambda: M["gdn_chunk_o"].chunk_o_ker(B, H, L, DK, DV, C, BK, BV),
}
ORDER = ["cumsum", "kkt", "solve_tril", "wy_fast", "chunk_h", "chunk_o"]

# TA_ONLY=cumsum,...：仅跑指定件（用于 CANN/ABI 快检，10 分钟止损用）
ONLY = {x for x in os.environ.get("TA_ONLY", "").split(",") if x}
if ONLY:
    ORDER = [n for n in ORDER if n in ONLY]
    print(f"TA-ONLY {sorted(ONLY)}", flush=True)

six_pass = six_fail = six_skip = 0

for name in ORDER:
    src_bytes = -1
    ck_ok = True
    try:
        ker = CK[name]()
        try:
            src_bytes = len(ker.get_kernel_source())
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        ck_ok = False
        print(f"TA-GDN-{name} FAIL reason=compile-error {type(exc).__name__}: {str(exc)[:160]}", flush=True)

    if MODE != "full":
        six_skip += 1
        print(f"TA-GDN-{name} SKIP reason=no-device compile={'ok' if ck_ok else 'fail'} "
              f"src_bytes={src_bytes}", flush=True)
        continue

    if not ck_ok:
        six_fail += 1
        continue

    try:
        inp = BUILD[name]()          # 输入构造在计时区外
        outs = KERNEL[name](inp)
        refs = REF[name](inp)
        if not isinstance(refs, tuple):
            refs = tuple(refs)
        rel, absmax, upstream_ok = 0.0, 0.0, True
        for a, b in zip(outs, refs):
            r, num = _maxrel(a, b)
            rel = float("nan") if not (r == r) else max(rel, r)
            absmax = max(absmax, num)
            try:
                torch.testing.assert_close(a.cpu(), b.cpu(), rtol=1e-5, atol=1e-5)
            except AssertionError:
                upstream_ok = False
        samples = _measure(lambda: KERNEL[name](inp))
        p50, p95 = _pct(samples, 0.50), _pct(samples, 0.95)
        ok = (rel == rel) and (rel < REL_TOL)
        six_pass += 1 if ok else 0
        six_fail += 0 if ok else 1
        print(f"TA-GDN-{name} {'PASS' if ok else 'FAIL'} rel={_fmt(rel)} ref=torch-fp32acc "
              f"p50_ms={_fmt(p50)} p95_ms={_fmt(p95)} dtype=fp16 iters={ITERS} warmup={WARMUP} "
              f"src_bytes={src_bytes} absmax={_fmt(absmax)} "
              f"upstream1e-5={'ok' if upstream_ok else 'over'} track=ta-ascendc", flush=True)
    except Exception as exc:  # noqa: BLE001
        six_fail += 1
        print(f"TA-GDN-{name} FAIL reason=run-error {type(exc).__name__}: {str(exc)[:200]}", flush=True)

# ---------------------------------------------------------------- gemm
gemm_pass = gemm_fail = gemm_skip = 0


def _load_gemm():
    """只取 example 的 matmul 定义段（跳过 top-level clear_cache 与 __main__ 采样段）。
    行号用前导换行对齐 gpath，满足 T.prim_func 解析所需的 inspect.findsource。
    dtype 契约：example 默认 'float16'（**不是** 'fp16'，后者解析出的 dtype 非法）。
    """
    gpath = os.path.join(TA_ROOT, "examples", "gemm", "example_gemm.py")
    src = open(gpath).read()
    a = src.index("@tilelang.jit")
    b = src.index("func = matmul")  # 跳过模块级 `func = matmul(M,N,K,...)`（引用 argparse 变量）
    body = "\n" * src[:a].count("\n") + src[a:b]
    ns = {"__name__": "j3_gemm", "tilelang": tilelang, "tilelang.language": tilelang.language,
          "T": tilelang.language}
    exec(compile(body, gpath, "exec"), ns)
    return ns


try:
    _G = _load_gemm()
    GEMM_OK = True
    GEMM_MATMUL = _G["matmul"]
except Exception as exc:  # noqa: BLE001
    GEMM_OK = False
    print(f"TA-GEMM FAIL reason=load {type(exc).__name__}: {str(exc)[:160]}", flush=True)

if GEMM_OK and not ONLY:
    # (dtype, block=(bm,bn,kl1), shape=(m,n,k))
    # fp16 用 example 自带 block (128,256,64)；fp32 因 L0 预算收紧到 (64,128,64)
    # —— 实测 (128,256,64)/(128,128,64)/(128,128,32) 在 fp32 下 bisheng 编不过。
    for dt, (bm, bn, kl1), (m, n, k) in [
        ("float16", (128, 256, 64), (1024, 1024, 1024)),   # 自带 example 形状
        ("float16", (128, 256, 64), (512, 512, 512)),      # 与标量数 linear 512³ 对位
        ("float32", (64, 128, 64), (512, 512, 512)),        # fp32 同采（收紧 block）
    ]:
        tag = f"dtype={dt} shape={m}x{n}x{k} blk={bm}/{bn}/{kl1}"
        try:
            fn = GEMM_MATMUL(m, n, k, bm, bn, kl1, dtype=dt)
            src_bytes = -1
            try:
                src_bytes = len(fn.get_kernel_source())
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            gemm_fail += 1
            print(f"TA-GEMM {tag} FAIL reason=compile-error "
                  f"{type(exc).__name__}: {str(exc)[:160]}", flush=True)
            continue
        if MODE != "full":
            gemm_skip += 1
            print(f"TA-GEMM {tag} SKIP reason=no-device src_bytes={src_bytes}", flush=True)
            continue
        try:
            tdt = torch.float16 if dt == "float16" else torch.float32
            torch.manual_seed(0)
            a = torch.randn(m, k).to(tdt).npu()
            b = torch.randn(k, n).to(tdt).npu()
            c = fn(a, b)
            rel, absmax = _maxrel(c, a @ b)
            samples = _measure(lambda: fn(a, b))
            p50, p95 = _pct(samples, 0.50), _pct(samples, 0.95)
            tol = 1e-2 if dt == "float16" else 2e-3   # fp16 沿 example 自带 rtol=atol=1e-2
            ok = (rel == rel) and (rel < tol)
            gemm_pass += 1 if ok else 0
            gemm_fail += 0 if ok else 1
            print(f"TA-GEMM {tag} {'PASS' if ok else 'FAIL'} rel={_fmt(rel)} "
                  f"ref=torch p50_ms={_fmt(p50)} p95_ms={_fmt(p95)} iters={ITERS} warmup={WARMUP} "
                  f"src_bytes={src_bytes} absmax={_fmt(absmax)} tol={tol} track=ta-ascendc", flush=True)
        except Exception as exc:  # noqa: BLE001
            gemm_fail += 1
            print(f"TA-GEMM {tag} FAIL reason=run-error "
                  f"{type(exc).__name__}: {str(exc)[:200]}", flush=True)

# ---------------------------------------------------------------- 真走内核自证
try:
    cache_so = len(glob.glob(os.path.join(CACHE, "**", "kernel_lib.so"), recursive=True))
except Exception:  # noqa: BLE001
    cache_so = -1
bk = len(BISHENG_RCS)
if bk > 0 and all(rc == 0 for rc in BISHENG_RCS):
    verdict = "real-kernel"
elif bk == 0:
    verdict = "eager-suspect"
else:
    verdict = "compile-error"
print(f"TA-KERNELPROOF bisheng_calls={bk} cache_so={cache_so} rc_all_zero={all(rc == 0 for rc in BISHENG_RCS) if bk else False} "
      f"judge='bisheng_calls>0 且 cache_so>0 ⇒ 走 JIT 内核非 eager' verdict={verdict}", flush=True)

TOTAL = len(ORDER)
print(f"TA-SUMMARY six={six_pass}/{TOTAL} gemm={gemm_pass} fail={six_fail + gemm_fail} "
      f"skip={six_skip + gemm_skip} mode={MODE} iters={ITERS} warmup={WARMUP} rel_tol={REL_TOL}", flush=True)

if MODE != "full":
    sys.exit(4)
sys.exit(0 if (six_pass == TOTAL and gemm_fail == 0) else 1)
PYEOF

python3 -c "import ast; ast.parse(open('$DRIVER').read())" \
  || { echo "TA-ABORT reason=driver-parse"; echo "TA-DONE"; exit 1; }

# ---- CANN/ABI 快检：单件 cumsum 编译，≤300s ----
rm -rf "$TILELANG_CACHE_DIR"
TA_ONLY=cumsum timeout 300 python3 "$DRIVER" > /tmp/ta_smoke.log 2>&1; SMOKE=$?
grep -E "^TA-" /tmp/ta_smoke.log
SMOKE_OK=1; SMOKE_REASON=""
grep -q "SKIP reason=no-device" /tmp/ta_smoke.log && { SMOKE_OK=0; SMOKE_REASON=no-device; }
grep -q "TA-ABORT reason=wrong-tilelang" /tmp/ta_smoke.log && { SMOKE_OK=0; SMOKE_REASON=wrong-tilelang; }
grep -q "FAIL reason=compile-error" /tmp/ta_smoke.log && { SMOKE_OK=0; SMOKE_REASON=cann-abi; }
if [ "$SMOKE_OK" -eq 0 ]; then
  if [ "${TA_ALLOW_NODEVICE:-0}" = "1" ]; then
    echo "TA0-SMOKE-WARN reason=$SMOKE_REASON（TA_ALLOW_NODEVICE=1：仅容器自证用，继续；驱动会把每件打成 SKIP reason=no-device，非假绿。卡上禁止设此变量）"
  else
    echo "TA-ABORT reason=$SMOKE_REASON smoke_exit=$SMOKE（no-device=跑在无设备形态；wrong-tilelang=PYTHONPATH 未生效；cann-abi=预编译 .so/bisheng 头 与卡 CANN 8.5.2 不兼容⇒需卡上重编，本窗不做）"
    grep -iE "error|undefined|symbol|Traceback|abort|Fatal" /tmp/ta_smoke.log | head -15
    echo "TA-DONE"; exit 2
  fi
else
  echo "TA0-SMOKE compile-ok exit=$SMOKE（CANN/ABI 兼容，继续全量）"
fi

echo "=== [1] GDN 六件真跑对拍 + 性能；[2] gemm 同采；[3] 汇总/内核自证 ==="
timeout 2400 python3 "$DRIVER" > /tmp/ta_run.log 2>&1; RUNEXIT=$?
echo "TA1-EXIT=$RUNEXIT"
grep -E "^TA-(DEVICE|MODE|TILELANG|REFCHOICE|GDN|GEMM|KERNELPROOF|SUMMARY)" /tmp/ta_run.log

echo "=== [4] 现场日志打包 ==="
tar czf /tmp/ta_w5_logs.tgz -C /tmp ta_smoke.log ta_run.log ta_deps_missing.log 2>/dev/null
echo "TA-TGZ size=$(wc -c </tmp/ta_w5_logs.tgz 2>/dev/null || echo 0)B"
echo "TA-DONE"
