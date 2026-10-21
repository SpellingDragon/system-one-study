#!/usr/bin/env python3
"""reconcile/F2_probe_dtype_guard.py — 位宽纪律钉的**负例自证**（p2-13 F2，任务书纪律第 2 条）。

判什么：`train_step` 的 fp16 归约 guard 不是装饰——
  N1 负例（伪造 fp16 归约）：装 guard 后伪造 `.sum()`/`@` 各一次，违规账必须红、
     `dtype_trace_line` 必须给 `fp32_accum=false`；再走**真 CLI**（env 注入伪造违规入口）
     判退码 == EXIT_DTYPE_DISCIPLINE(6) 且 `DTYPE-TRACE-FAIL` 落 stderr。
  N2 合规正例（不误伤）：同样形状但先 `.float()` 升宽再归约/内积 ⇒ 违规 0、
     `fp32_accum=true`、退码链不误触。
  N3 显式 dtype=fp32 归约：记合规计数、不记违规。
  N4 纯函数账：`dtype_trace_line` 喂假账逐字段核对（params/fp16/fallback 直通）。
  N5 对称卸载：uninstall 后伪造归约不再记账（不污染同进程后续 fp32 训练步）。

用法：cd release && .venv/bin/python ascend/kernels/reconcile/F2_probe_dtype_guard.py
判决行：`F2-GUARD-<用例>-PASS/FAIL …`，末行 `F2-GUARD-SUMMARY n/m`。
"""
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
RELEASE = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RELEASE))

import torch  # noqa: E402
from ascend import train_step as ts  # noqa: E402

RESULTS = []


def verdict(name: str, ok: bool, detail: str = "") -> None:
    print(f"F2-GUARD-{name}-{'PASS' if ok else 'FAIL'} {detail}", flush=True)
    RESULTS.append(ok)


# ---------------- N1/N2/N3：guard 记账行为 ----------------
ts.install_reduction_guard()
a = torch.randn(8, 8, dtype=torch.float16)
b = torch.randn(8, 8, dtype=torch.float16)

base = len(ts.reduction_violations())
_ = a.sum()          # 伪造违规 1：fp16 直接归约
_ = a @ b            # 伪造违规 2：fp16 运算符面内积
viol = ts.reduction_violations()
hit_direct = any(v.startswith("sum(fp16 直接归约") for v in viol[base:])
hit_at = any("__matmul__" in v and "内积面" in v for v in viol[base:])
audit_bad = {"dtype": "fp16", "params": 16, "fp16": 16, "violations": viol, "fp32_acc": 0}
line_bad = ts.dtype_trace_line(audit_bad, fallback_events=0)
verdict("N1-负例-记账", hit_direct and hit_at and len(viol) - base == 2,
        f"新增违规={len(viol) - base}/2 :: {viol[base:]}")
verdict("N1-负例-判决行", "fp32_accum=false" in line_bad, line_bad)

base = len(ts.reduction_violations())
acc_before = ts._reduction_ledger["fp32_acc"]
_ = a.float().sum()          # 合规：先升 fp32 面再归约
_ = a.float() @ b.float()    # 合规：fp32 面内积
_ = a.sum(dtype=torch.float32)  # 合规：显式累加位宽
viol2 = ts.reduction_violations()
acc_after = ts._reduction_ledger["fp32_acc"]
verdict("N2-合规不误伤", len(viol2) == len(viol),
        f"合规三连后违规增量={len(viol2) - base}（应为 0），当前共 {len(viol2)} 条")
verdict("N3-显式dtype升宽计数", acc_after - acc_before == 1,
        f"fp32_acc 计数 +{acc_after - acc_before}（应 +1）")
ts.uninstall_reduction_guard()

# ---------------- N5：卸载对称 ----------------
base = len(ts.reduction_violations())
_ = a.sum()
_ = a @ b
verdict("N5-卸载后不记", len(ts.reduction_violations()) == base,
        f"卸载后伪造两笔，违规增量={len(ts.reduction_violations()) - base}（应为 0）")

# ---------------- N4：纯函数字段直通 ----------------
line = ts.dtype_trace_line({"dtype": "fp16", "params": 16, "fp16": 16, "violations": []},
                           fallback_events=3)
want = "DTYPE-TRACE fp16 params=16 fp16=16 fp32_accum=true fallback=3"
verdict("N4-纯函数格式", line == want, f"got={line!r}")

# ---------------- N1-CLI：真退码 6 链路 ----------------
# 用 sitecustomize 太重，改为临时 wrapper：`-c` 里先 import train_step、装 guard 后把
# 伪造违规钉进 LAST_AUDIT（等价于 run() 步末落账的结果），再调 main() 走完整判据链。
harness = '''
import sys
sys.path.insert(0, {root!r})
from ascend import train_step as ts
ts.LAST_AUDIT = {{"dtype": "fp16", "params": 16, "fp16": 16,
                 "violations": ["sum(fp16 直接归约: shape=(16, 32) dtype=None)"],
                 "fp32_acc": 0}}
_orig = ts.run
def run(*a, **k):
    out = _orig(*a, **k)
    # run() 步末会覆写 LAST_AUDIT（fp32 档平凡干净）——这里模拟"fp16 档步末落账即违规"
    ts.LAST_AUDIT = {{"dtype": "fp16", "params": 16, "fp16": 16,
                     "violations": ["sum(fp16 直接归约: shape=(16, 32) dtype=None)"],
                     "fp32_acc": 0}}
    return out
ts.run = run
rc = ts.main(["--target", "cpu", "--device", "cpu"])
print(f"HARNESS_RC={{rc}}")
'''
proc = subprocess.run(
    [sys.executable, "-c", harness.format(root=str(RELEASE))],
    capture_output=True, text=True, cwd=str(RELEASE), env=os.environ.copy())
rc_line = [x for x in proc.stdout.splitlines() if x.startswith("HARNESS_RC=")]
rc = int(rc_line[0].split("=")[1]) if rc_line else -1
dt_line = [x for x in proc.stdout.splitlines() if x.startswith("DTYPE-TRACE ")]
fail_line = "DTYPE-TRACE-FAIL" in proc.stderr
verdict("N1-CLI退码6", rc == ts.EXIT_DTYPE_DISCIPLINE and fail_line
        and dt_line and "fp32_accum=false" in dt_line[0],
        f"exit={rc}(应={ts.EXIT_DTYPE_DISCIPLINE}) TRACE-FAIL落stderr={fail_line} "
        f"判决行={dt_line[0] if dt_line else '无'!r}")

# 对照：同样 main 链、无违规账（fp32 档真实 LAST_AUDIT）⇒ 退码 0，纪律钉不干扰原链
proc0 = subprocess.run([sys.executable, str(RELEASE / "ascend" / "train_step.py"),
                        "--target", "cpu", "--device", "cpu"],
                       capture_output=True, text=True, cwd=str(RELEASE), env=os.environ.copy())
verdict("N2-CLI原链无扰", proc0.returncode == 0
        and "DTYPE-TRACE fp32 params=16 fp16=0 fp32_accum=true" in proc0.stdout,
        f"exit={proc0.returncode}")

n_ok = sum(RESULTS)
print(f"F2-GUARD-SUMMARY {n_ok}/{len(RESULTS)}", flush=True)
sys.exit(0 if n_ok == len(RESULTS) else 1)
