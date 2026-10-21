"""B_probe_kernel_usage_negative.py — P1-1l「内核启用自证」判决矩阵（含"伪造回退必红"负例）。

【要回答的问题】`train_step.py` 原来只判"接线对不对"（loss 降 + 全参有梯度）。探针/发射在卡上
一失败，`ascend_env.get_compiled()` 返回 None、入口层落回普通 torch 写法，**train_step 照样
打 P2-STEP PASS、照样退 0**——这是最后一道静默假绿。本探针把 `autograd_asc.kernel_usage_report()`
的三路证据逐个单独打断，验"每一路都 load-bearing"，再端到端验"红了一定改退码"。

【三路与对应负例】
    M1 合成干净账本          → KERNEL-USED-PASS（并钉死判决行格式）
    M2 拔掉 ①（compiled 为空）→ FAIL why=kernels_absent
    M3 伪造 ②（blockers 非空）→ FAIL why=blockers_present
    M4 伪造 ③（真发一条与 record_blocker 同形的回退公告，用 showwarning 钩子数出来）
                             → FAIL why=fallback_events（且 ①② 单独保持绿，证明 ③ 独立生效）
    M5 前缀错位（只有 cpu| 键却问 ascend）→ FAIL why=kernels_absent + foreign_keys 指出来
    E1 端到端 cpu 轨（默认）  → KERNEL-USED-FAIL 但退 0（cpu 轨不在门下：gdn_backward[cpu]
                               回落是本波**已登记事实**，见 tests/test_ascend_train_step.py）
    E2 端到端 cpu 轨 + SYS1_KERNEL_USAGE_STRICT=1 → 同一份真账本 ⇒ 退 5（门确实咬得住）
    E3 端到端 ascend 轨（本机无 npu）→ 应先在 H1 处停住（退 4），压根不给"假绿"机会

【不证什么】卡上 `KERNEL-USED-PASS kernels>=1`——本机无 npu，ascend 前缀的键不可能真产生
（入口层 `active_backend("ascend", A.device)` 的设备门就把内核挡住了）。下窗要逐字看到的形是：
    KERNEL-USED-PASS kernels=13 blockers=0 fallbacks=0 target=ascend
（13 = A3_PREFLIGHT §3 实测的 train_step ascend 唯一键数；少于 13 说明有件回落，等于 0 说明整步没走内核。）

【环境前提（B 代理实测登记）】端到端 E1/E2（以及 H1-5 的 cpu 轨那两条）依赖 **host 的
release/.venv**。容器 `cann910b-h`（python3.10 dist-packages 那套 torch）里 cpu 轨一步训练本身
就崩——`ascend/kernels/lora_asc.py:137 _eager_grad` 抛 `RuntimeError: element 0 of tensors does
not require grad and does not have a grad_fn`，**origin/main 的原始 train_step.py 同崩（退 1）**，
与本件改动无关（复跑见 B_RESULT.md）。⇒ 这两枚探针在容器里 E1/E2 会 MISS，那是环境差异，
不是判据坏；要跑就 host 跑。卡窗侧不受影响：ascend 轨走方言路，不经 `_eager_grad`。

用法：
    cd release && .venv/bin/python ascend/kernels/reconcile/B_probe_kernel_usage_negative.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings

FAILS: list[str] = []


def _release_root() -> str:
    env = os.environ.get("A3_ROOT") or os.environ.get("SYS1_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


ROOT = _release_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
from ascend import autograd_asc as ops  # noqa: E402
from ascend import train_step as ts  # noqa: E402
from ascend.kernels import ascend_env  # noqa: E402

TRAIN_STEP = os.path.join(ROOT, "ascend", "train_step.py")
FAKE_KEYS = ("ascend|gemm[ascend]|n32k32b16x32x32anone",
             "ascend|gemm_dw[ascend]|n32k32tc16b32x32",
             "ascend|add_ln[ascend]|d32e1e-05b16")


def check(label: str, got, want) -> None:
    """want 是字符串 ⇒ 子串命中；否则 ⇒ 逐字相等。写错了必须 MISS，不许把非空当真。"""
    hit = (want in str(got)) if isinstance(want, str) else (str(got) == str(want))
    print(f"  {'ok  ' if hit else 'MISS'} {label}: {got}")
    if not hit:
        FAILS.append(label)


# ---------------------------------------------------------------- M1 干净账本 ⇒ PASS
print("B-KU-M1 合成干净账本（三证齐）⇒ PASS + 钉死判决行格式")
rep = ops.kernel_usage_report("ascend", fallback_events=0, compiled=FAKE_KEYS, blockers={})
check("ok=True", rep["ok"], rep["ok"])
check("kernels=3", rep["n_kernels"], 3)
check("判决行逐字", rep["line"],
      "KERNEL-USED-PASS kernels=3 blockers=0 fallbacks=0 target=ascend")

# ---------------------------------------------------------------- M2 拔掉 ①
print("B-KU-M2 ①拔掉（没有任何 ascend| 键）⇒ 必红")
rep = ops.kernel_usage_report("ascend", fallback_events=0, compiled=(), blockers={})
check("ok=False", rep["ok"], False)
check("why=kernels_absent", rep["line"], "KERNEL-USED-FAIL")
check("未达项只有 ①", rep["failed"], "kernels_absent")

# ---------------------------------------------------------------- M3 伪造 ②
print("B-KU-M3 ②伪造（阻塞账里塞一条）⇒ 必红")
rep = ops.kernel_usage_report("ascend", fallback_events=0, compiled=FAKE_KEYS,
                              blockers={"ascend|gemm[ascend]|x": "RuntimeError: boom"})
check("ok=False", rep["ok"], False)
check("未达项只有 blockers", rep["failed"], "blockers_present")
check("判决行带出 blocker 键名", rep["line"], "blocker_keys=")

# ---------------------------------------------------------------- M4 伪造 ③（真发公告）
print("B-KU-M4 ③伪造（真发一条与 record_blocker 同形的回退公告，钩子数出来）⇒ 必红")
seen: list[str] = []
_show = warnings.showwarning
warnings.showwarning = lambda m, c, f, l, file=None, line=None: seen.append(str(m))
with warnings.catch_warnings():
    warnings.simplefilter("always")
    ascend_env.record_blocker("ascend|gemm[ascend]|probe-fake", "simulated: 发射桩失败")
warnings.showwarning = _show
fb = ops.count_fallback_events(seen)
check("count_fallback_events 数到 1", fb, 1)
rep = ops.kernel_usage_report("ascend", fallback_events=fb, compiled=FAKE_KEYS, blockers={})
check("①② 都绿、只 ③ 红 ⇒ 三路各自独立 load-bearing", rep["failed"], ["fallback_events"])

print("B-KU-M4b 反向：无关 warning 不计入回落次数（不许把噪声算成回落）")
check("噪声文案不命中 token",
      ops.count_fallback_events(["UserWarning: something unrelated",
                                 "torch.cuda 之类"]), 0)
ascend_env.reset()      # 探针自己造的 blocker 归零，别留给后续 leg

# ---------------------------------------------------------------- M5 前缀错位
print("B-KU-M5 前缀错位（只有 cpu| 键，问 ascend）⇒ 红，并把外来键指出来")
rep = ops.kernel_usage_report("ascend", fallback_events=0,
                              compiled=("cpu|gemm[cpu]|n32k32b16x32x32anone",), blockers={})
check("ok=False", rep["ok"], False)
check("line 带 foreign_keys（可读回到底走的是哪一路）", rep["line"], "foreign_keys=")


# ---------------------------------------------------------------- E1/E2/E3 端到端
def _run(env_extra: dict | None, *argv: str) -> tuple[int, str]:
    env = dict(os.environ, **(env_extra or {}))
    p = subprocess.run([sys.executable, TRAIN_STEP, *argv], capture_output=True, text=True, env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


print("E1 端到端 cpu 轨（默认门，不门禁）——同一趟训练里**确实发生过一次回落**")
rc, out = _run(None, "--target", "cpu", "--device", "cpu")
check("接线判据仍是 PASS（说明旧 PASS 根本不排除回落）", out, "reduced=True")
check("内核自证行如实报红", out, "KERNEL-USED-FAIL")
check("真实回落被数到（fallbacks=1，来自 gdn_backward[cpu]）", out, "fallbacks=1")
check("cpu 轨退码不变（不越权改既有 host 行为）", rc, 0)

print("E2 端到端 cpu 轨 + SYS1_KERNEL_USAGE_STRICT=1 ⇒ 同一份真账本必须改退码")
rc, out = _run({"SYS1_KERNEL_USAGE_STRICT": "1"}, "--target", "cpu", "--device", "cpu")
check("退码 5=EXIT_KERNEL_UNUSED", rc, ts.EXIT_KERNEL_UNUSED)
check("stderr 给出未达项", out, "KERNEL-USED-BLOCK")

print("E3 端到端 ascend 轨（本机无 torch_npu）⇒ 先撞 H1，不给假绿留机会")
rc, out = _run(None, "--target", "ascend")
check("退码 4=EXIT_DEVICE_UNAVAILABLE（不是 traceback 裸崩）", rc, ts.EXIT_DEVICE_UNAVAILABLE)
check("有 NPU-REG fail 行", out, "NPU-REG fail")

print("B-KU-DONE", "PASS" if not FAILS else f"FAIL {FAILS}")
sys.exit(1 if FAILS else 0)
