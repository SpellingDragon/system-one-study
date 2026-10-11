"""B_probe_h1_npu_reg.py — H1（torch_npu 设备注册）分支矩阵自证（无卡可跑：host 与容器同一条命令）。

【证什么】`train_step.py` 的 H1 修法在四种机器形态下的**分支走向**，其中最重要的一条是
"**cpu 轨一次都不会 import torch_npu**"——否则无 npu 的机器（本机、以及所有 CI）连
`import ascend.train_step` 都会 ImportError，把 cpu 轨整条砸掉。

【不证什么】卡上"注册真成功"。本机与 cann910b-* 容器都没有 npu、也没有 torch_npu（容器实测
`run_delta_card.py` 就报 `No module named 'torch_npu'`），所以 ok 分支只能靠**假件**把
`torch.device('npu')` 的可造性桩出来，验的是判据与代码路径，不是卡。卡窗下窗逐字对这一行：
    NPU-REG ok <torch_npu.__version__> target=ascend npu_avail=True torch_npu 注册成功

【手法】① 进程内直接调 `train_step.npu_requested/ensure_torch_npu`（每次把模块级账本
`_npu_reg` 归零，桩掉 `_npu_device_usable` 造出"import 前不可用 / import 后可用"两种机器）；
② 端到端用**子进程** + PYTHONPATH 里放一枚"毒药 torch_npu"（import 即抛 `POISON`）——
cpu 轨的输出里出现 POISON 就说明它 import 了（必红），ascend 轨出现 POISON 则应被收成
`NPU-REG fail ... POISON` + 退码 4，而不是裸 traceback。

【环境前提（B 代理实测登记）】端到端 E1/E2（以及 H1-5 的 cpu 轨那两条）依赖 **host 的
release/.venv**。容器 `cann910b-h`（python3.10 dist-packages 那套 torch）里 cpu 轨一步训练本身
就崩——`ascend/kernels/lora_asc.py:137 _eager_grad` 抛 `RuntimeError: element 0 of tensors does
not require grad and does not have a grad_fn`，**origin/main 的原始 train_step.py 同崩（退 1）**，
与本件改动无关（复跑见 B_RESULT.md）。⇒ 这两枚探针在容器里 E1/E2 会 MISS，那是环境差异，
不是判据坏；要跑就 host 跑。卡窗侧不受影响：ascend 轨走方言路，不经 `_eager_grad`。

用法：
    cd release && .venv/bin/python ascend/kernels/reconcile/B_probe_h1_npu_reg.py
    # 容器里（配方见 A3_PREFLIGHT §4）：
    # docker exec cann910b-h bash -c 'cd /tmp/a3sys1/release && A3_ROOT=$PWD python3 \
    #   ascend/kernels/reconcile/B_probe_h1_npu_reg.py'
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import types

FAILS: list[str] = []


def _release_root() -> str:
    env = os.environ.get("A3_ROOT") or os.environ.get("SYS1_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))        # .../release


ROOT = _release_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ascend import train_step as ts  # noqa: E402

TRAIN_STEP = os.path.join(ROOT, "ascend", "train_step.py")


def check(label: str, got, want) -> None:
    hit = str(got) == str(want) if not isinstance(want, str) else want in str(got)
    print(f"  {'ok  ' if hit else 'MISS'} {label}: {got}")
    if not hit:
        FAILS.append(label)


# ---------------------------------------------------------------- ① npu_requested 真值表
print("B-H1-1 npu_requested 判定表（只有 ascend 轨才动手）")
for argv, want in (
    (["--target", "ascend"], True), (["--target=ascend"], True),
    (["--target", "cpu"], False), ([], False), (["--help"], False),
    (["--target", "ascend", "--device", "cpu"], True),
):
    check(f"argv={argv}", ts.npu_requested(argv), want)
os.environ["DMLAYA_ASCEND_TARGET"] = "asc"
check("env=asc(别名)", ts.npu_requested([]), True)
os.environ.pop("DMLAYA_ASCEND_TARGET", None)

# ---------------------------------------------------------------- ② 无 torch_npu 的真实机器
print("B-H1-2 本机形态：torch_npu 不在场 ⇒ status=fail 且不抛（旧版在这里裸崩）")
ts._npu_reg = None
st = ts.ensure_torch_npu()
check("status", st["status"], "fail")
check("detail 含 ModuleNotFoundError", st["detail"], "ModuleNotFoundError")
try:
    torch_npu_absent = "torch_npu" not in sys.modules
except Exception:  # noqa: BLE001
    torch_npu_absent = False
check("探测后 sys.modules 仍无 torch_npu", torch_npu_absent, True)

# ---------------------------------------------------------------- ③ ok 分支（假件造出"注册成功"）
print("B-H1-3 卡上形态（假件桩出 npu 可造性）：ascend 轨必须真 import torch_npu")
fake = types.ModuleType("torch_npu")
fake.__version__ = "9.9.9-fake"
sys.modules["torch_npu"] = fake
calls = {"n": 0}


def _usable_after_import() -> bool:
    calls["n"] += 1
    return calls["n"] > 1          # 第一次问=没注册；import 之后再问=注册好了


_orig_usable = ts._npu_device_usable
ts._npu_device_usable = _usable_after_import
ts._npu_reg = None
st = ts.ensure_torch_npu()
check("status", st["status"], "ok")
check("version 取自 torch_npu", st["version"], "9.9.9-fake")
check("detail", st["detail"], "torch_npu 注册成功")
check("确实问过两次可造性（import 前后各一次）", calls["n"], 2)
# 预览行里 npu_avail 会是 probe-error：本机没有 torch.npu 这个属性，is_available() 读不到。
# 卡上同一支应当读到 npu_avail=True——这一行的逐字对照留给下窗（见文件头的"卡窗预期行"）。
print(f"  判决行预览: {ts.npu_reg_line(st, 'ascend')}")

print("B-H1-4 import 成功但设备仍不可造（昇腾包坏掉）⇒ status=fail，不许冒充 ok")
ts._npu_device_usable = lambda: False
ts._npu_reg = None
st = ts.ensure_torch_npu()
check("status", st["status"], "fail")
check("detail", st["detail"], "仍不可用")
ts._npu_device_usable = _orig_usable
ts._npu_reg = None
sys.modules.pop("torch_npu", None)

# ---------------------------------------------------------------- ④ 端到端：毒药 torch_npu
print("B-H1-5 端到端：PYTHONPATH 放一枚 import 即抛的假 torch_npu（POISON）")
poison_dir = tempfile.mkdtemp(prefix="b_poison_")
with open(os.path.join(poison_dir, "torch_npu.py"), "w", encoding="utf-8") as fh:
    fh.write('raise RuntimeError("POISON cpu 轨不得 import torch_npu")\n')


def _run(*argv: str) -> tuple[int, str]:
    p = subprocess.run([sys.executable, TRAIN_STEP, *argv], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=poison_dir))
    return p.returncode, (p.stdout or "") + (p.stderr or "")


rc, out = _run("--target", "cpu", "--device", "cpu")
# 判据要分清两件事（B 代理第一轮在这里自己绊了自己一下，值得登记）：
#   · "import torch_npu ->" 这一截**只可能由本件的 ensure_torch_npu() 打出来** ⇒ cpu 轨必须没有；
#   · "An unexpected error occurred while checking NPU availability" 是 **torch 的惰性后端探测**
#     打的，实测 `import tilelang` 一句就会触发（与 train_step 无关，属上游既有行为）。
#     所以这一截出现在 cpu 轨输出里**不算违规**，但它是重要判读口径：卡上若装了 torch_npu,
#     tilelang 的 import 可能已经把 npu 注册好了——**能依赖，也不能依赖**（版本/顺序一变即失效），
#     显式 import 才是确定性做法，也正是 H1 的修法。
check("cpu 轨没有本件发起的 import（无 'import torch_npu ->'）", "import torch_npu ->" in out, False)
check("cpu 轨 NPU-REG skip", out, "NPU-REG skip not-imported target=cpu npu_avail=n/a")
check("cpu 轨仍然跑完（退码 0，内核自证不进门禁）", rc, 0)
if "checking NPU availability" in out:
    print("  note   tilelang 的惰性 npu 探测在场（上游行为，已登记为判读口径，不计违规）")

rc, out = _run("--target", "ascend", "--device", "npu")
check("ascend 轨由本件发起 import", out, "import torch_npu -> RuntimeError: POISON")
check("ascend 轨收成 NPU-REG fail（不是裸 traceback）", out, "NPU-REG fail")
check("ascend 轨退码 4=EXIT_DEVICE_UNAVAILABLE", rc, ts.EXIT_DEVICE_UNAVAILABLE)

print("B-H1-DONE", "PASS" if not FAILS else f"FAIL {FAILS}")
sys.exit(1 if FAILS else 0)
