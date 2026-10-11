"""B_probe_cardrun_verdict_matrix.py — 卡窗执行器（H2/H4 + 增采 C + BENCH 门）判决矩阵自证。

【要回答的问题】`card_run.sh` 改完以后，"跑到底"和"过了"还是不是两件事？旧版只要脚本没被
kill，最后一定打一行裸 `RUNNER-DONE`，人看一眼就当成验收通过（A3 §5 H4 的病灶）。新版所有腿
走 `run_leg` + `runner_done`，本探针**不修改卡窗件**：只把它头部的执行器、和 [4b]/[4c] 两段
按行**原样摘录**到沙盒里，喂进"判决行形状逐字对齐真件"的假命令，看分类与计数是否如预期。

【矩阵】
    S1  Y-<t>-PROD-PASS rc=0            ⇒ VERDICT-PASS   pass=1 fail=0 exit=0
    S2  Y-<t>-ASC-COMPILE-PASS note=无设备 ⇒ VERDICT-DEGRADED **fail=1 exit=1** ← H4 核心负例
    S3  Y-<t>-PROD-FAIL                 ⇒ VERDICT-FAIL   exit=1
    S4  啥判决行都没有（rc=0）           ⇒ VERDICT-FAIL-NO-VERDICT（不许空转）
    S5  打了 PASS 行但退码非 0           ⇒ VERDICT-FAIL-EXIT1（PASS 行 + rc≠0 不算过）
    S6  rc=124                          ⇒ VERDICT-FAIL-TIMEOUT
    S7  advisory 腿失败 + gating 腿通过   ⇒ fail=0 advisory_fail=1 exit=0（对照轨不顶退码，但可见）
    S8  缓存目录里有 device_kernel.asc    ⇒ KERNELS-TGZ 行 + VERDICT-PASS（真的 tar 一遍）
    S9  缓存目录空（本窗没编过东西）      ⇒ KERNELS-TGZ-FAIL ⇒ VERDICT-FAIL
    S10 SYS1_BENCH 未设                  ⇒ CASE-BENCH SKIP + skipped_legs=1（默认不进窗）
    S11 SYS1_BENCH=1 但交付件不在         ⇒ CASE-BENCH SKIP（E2 未入库时静默跳，不红）
    S12 trainstep 腿：P2-STEP PASS 但 KERNEL-USED-FAIL ⇒ VERDICT-FAIL（本元静默假绿被腿级抓住）

【不证什么】真机上这些判决行的**内容**是否正确（那是 [3]/[4] 各件的事，A2/A3 已有干跑结论）。
本探针只证"窗子的读法与计数不会被 PASS 形状的降级骗过去"。

【host 替身声明】darwin 没有 GNU `timeout(1)` ⇒ 沙盒里定义了同名 bash 函数（shift 掉秒数直接
执行）。这只影响探针沙盒，**card_run.sh 本身一个字没为此改**；卡上（Linux）用的是真 timeout。

用法：
    cd release && .venv/bin/python ascend/kernels/reconcile/B_probe_cardrun_verdict_matrix.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

FAILS: list[str] = []

SRC = None
ROOT = os.environ.get("SYS1_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
# CARDRUN_SRC 只为"变异体自证"留的口子（sabotage：故意把 DEGRADED 不计 fail，本探针必须响）
CARD_RUN = os.environ.get("CARDRUN_SRC") or os.path.join(ROOT, "ascend", "port910b", "card_run.sh")


def check(label: str, got, want) -> None:
    """want 是字符串 ⇒ 子串命中；否则 ⇒ 逐字相等。写错了必须 MISS，不许把非空当真。"""
    hit = (want in str(got)) if isinstance(want, str) else (str(got) == str(want))
    print(f"  {'ok  ' if hit else 'MISS'} {label}: {str(got)[:400]}")
    if not hit:
        FAILS.append(label)


# ------------------------------------------------------------------ 摘录 card_run.sh 的既有区段
lines = open(CARD_RUN, encoding="utf-8").read().splitlines()


def find_one(prefix: str) -> int:
    hits = [i for i, ln in enumerate(lines) if ln.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit(f"B-CR-DONE FAIL 摘录锚点不唯一/缺失：{prefix!r} → {hits}")
    return hits[0]


def slice_to(prefix: str) -> list[str]:
    """从某行起取到"下一个顶格 echo "=== " 或顶格 runner_done"为止（不含）。"""
    i = find_one(prefix)
    out = []
    for ln in lines[i:]:
        out.append(ln)
        if len(out) > 1 and (re.match(r'^echo "=== ', ln) or ln.strip() == "runner_done"):
            out.pop()
            break
    return out


i_ctr = find_one("PASS_N=0;")
i_end = find_one('echo "=== [0] env ==="')
EXECUTOR = "\n".join(lines[i_ctr:i_end])                       # 计数器 + run_leg + runner_done + 腿数
BENCH_BLOCK = "\n".join(slice_to('if [ -z "${SYS1_BENCH:-}" ]; then'))   # [4b] 门（原样）
TGZ_LINE = [ln for ln in lines if ln.lstrip().startswith("bash -c 'd=\"$1\"")]
if len(TGZ_LINE) != 1:
    raise SystemExit(f"B-CR-DONE FAIL 增采 C 载荷行没抓到：{len(TGZ_LINE)} 行")
TGZ_CMD = TGZ_LINE[0].strip()


def harness(legs: list[str], envs: dict | None = None, tail: str = "") -> tuple[int, str]:
    """把执行器 + 假腿拼成沙盒脚本跑一遍，返回 (退码, 输出)。"""
    sh = ["#!/bin/bash", "set -uo pipefail",
          'timeout() { local t="$1"; shift; "$@"; }   # host 替身（见文件头声明）',
          f'SYS1="{SANDBOX}"', "RUNLOG=${CARDRUN_LOGDIR}", EXECUTOR, ""]
    sh += legs
    sh += [tail, "runner_done"]
    path = os.path.join(SANDBOX, "harness.sh")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sh) + "\n")
    p = subprocess.run(["bash", path], capture_output=True, text=True,
                       env=dict(os.environ, CARDRUN_LOGDIR=os.path.join(SANDBOX, "legs"),
                                **(envs or {})))
    return p.returncode, (p.stdout or "") + (p.stderr or "")


SANDBOX = tempfile.mkdtemp(prefix="b_cardrun_")
P = lambda *s: list(s)     # noqa: E731  可读性：一条腿 = 一行

# ------------------------------------------------------------------ S1 PROD-PASS ⇒ PASS
print("S1 产品件数值判（Y-*-PROD-PASS + rc=0）⇒ VERDICT-PASS，窗子退 0")
rc, out = harness(P('run_leg "A" 5 "^Y-a" "^Y-a-PROD-FAIL" "^Y-a-PROD-PASS" "^Y-a-ASC-COMPILE-PASS" \\',
                    '  bash -c \'echo "Y-a-PROD-PASS rel=1.0e-03 reuse=13/13"; exit 0\''))
check("腿判 PASS", out, "CASE-A VERDICT-PASS rc=0")
check("窗退 0", out, "RUNNER-DONE SUMMARY pass=1 fail=0 skipped_legs=0 degraded=0 advisory_fail=0 exit=0")
check("退码", rc, 0)

# ------------------------------------------------------------------ S2 降级骗不过去（H4 核心负例）
print("S2 `note=无设备⇒只判编译`（PASS 形状的降级）⇒ 必须 DEGRADED 且**把整窗顶成 exit 1**")
rc, out = harness(P('run_leg "A" 5 "^Y-a" "^Y-a-PROD-FAIL" "^Y-a-PROD-PASS" "^Y-a-ASC-COMPILE-PASS" \\',
                    '  bash -c \'echo "Y-a-ASC-COMPILE-PASS keys=13 src~9000B track=card(no-run) note=无设备⇒只判编译"; exit 0\''))
check("腿判 DEGRADED", out, "CASE-A VERDICT-DEGRADED rc=0")
check("降级计进 fail（旧版这里就是 pass=1 的错觉）", out, "SUMMARY pass=0 fail=1 skipped_legs=0 degraded=1")
check("整窗 exit 1", rc, 1)

# ------------------------------------------------------------------ S3 PROD-FAIL
print("S3 产品件数值失败 ⇒ VERDICT-FAIL")
rc, out = harness(P('run_leg "A" 5 "^Y-a" "^Y-a-PROD-FAIL" "^Y-a-PROD-PASS" "-" \\',
                    '  bash -c \'echo "Y-a-PROD-FAIL AssertionError: rel=0.17" ; exit 1\''))
check("FAIL", out, "CASE-A VERDICT-FAIL rc=1")
check("第二趟把错误行也带出来（H2 两趟，判决不再被挤掉）", out, "AssertionError")
check("exit 1", rc, 1)

# ------------------------------------------------------------------ S4 空转
print("S4 既没判决行也没错误行（rc=0）⇒ FAIL-NO-VERDICT（不许把『跑完了』当过）")
rc, out = harness(P('run_leg "A" 5 "^Y-a" "^Y-a-PROD-FAIL" "^Y-a-PROD-PASS" "-" \\',
                    '  bash -c \'echo "some noise"; exit 0\''))
check("NO-VERDICT", out, "VERDICT-FAIL-NO-VERDICT")
check("exit 1", rc, 1)

# ------------------------------------------------------------------ S5 PASS 行 + 非 0 退码
print("S5 打了 PASS 行但退码非 0 ⇒ FAIL-EXIT1")
rc, out = harness(P('run_leg "A" 5 "^Y-a" "^Y-a-PROD-FAIL" "^Y-a-PROD-PASS" "-" \\',
                    '  bash -c \'echo "Y-a-PROD-PASS rel=1e-03"; exit 1\''))
check("FAIL-EXIT1", out, "VERDICT-FAIL-EXIT1 rc=1")
check("PASS 计数为 0", out, "pass=0")

# ------------------------------------------------------------------ S6 超时
print("S6 rc=124（timeout 到点）⇒ FAIL-TIMEOUT")
rc, out = harness(P('run_leg "A" 5 "^Y-a" "^Y-a-PROD-FAIL" "^Y-a-PROD-PASS" "-" \\',
                    '  bash -c \'echo "Y-a-PROD-PASS"; exit 124\''))
check("TIMEOUT", out, "VERDICT-FAIL-TIMEOUT rc=124")

# ------------------------------------------------------------------ S7 对照轨单列
print("S7 LEG_TRACK=advisory 的对照腿失败 ⇒ 单列 advisory_fail，不顶退码（ASC-PERF 同构口径）")
rc, out = harness(P('run_leg "A" 5 "^Y-a" "^Y-a-PROD-FAIL" "^Y-a-PROD-PASS" "-" \\',
                    '  bash -c \'echo "Y-a-PROD-PASS"; exit 0\'',
                    'LEG_TRACK=advisory',
                    'run_leg "P" 5 "^NUM-p" "^NUM-p-FAIL" "^NUM-p[ -]PASS" - \\',
                    '  bash -c \'echo "NUM-p-FAIL RuntimeError: not supported"; exit 1\'',
                    'LEG_TRACK=gating'))
check("gating 腿 PASS", out, "CASE-A VERDICT-PASS")
check("对照腿 track=advisory 记账", out, "CASE-P VERDICT-FAIL rc=1 track=advisory")
check("fail=0 但 advisory_fail=1（两笔都摊开，不含糊）", out,
      "SUMMARY pass=1 fail=0 skipped_legs=0 degraded=0 advisory_fail=1 exit=0")

# ------------------------------------------------------------------ S8 增采 C（真 tar）
print("S8 增采 C：缓存目录里有 device_kernel.asc ⇒ 打 KERNELS-TGZ 大小/件数并 PASS")
cache = os.path.join(SANDBOX, "cache")
for k in ("ascend|gemm", "ascend|dw"):
    d = os.path.join(cache, k)
    os.makedirs(d, exist_ok=True)
    for f in ("device_kernel.asc", "host_kernel.c"):
        with open(os.path.join(d, f), "w", encoding="utf-8") as fh:
            fh.write("// fake kernel body\n")
os.environ["TILELANG_CACHE_DIR"] = cache
rc, out = harness(P(f'TGZ="{os.path.join(SANDBOX, "k.tgz")}"',
                    'run_leg "KERNELS-TGZ" 180 \'^KERNELS-TGZ\' \'^KERNELS-TGZ-FAIL\' '
                    '\'^KERNELS-TGZ .*size=\' - \\', "  " + TGZ_CMD))
check("增采 C 腿 PASS", out, "CASE-KERNELS-TGZ VERDICT-PASS")
check("行里有 size 与 entries/kernels", out, "entries=4 kernels=2")
m = re.search(r"KERNELS-TGZ \S+ size=(\d+)B", out)
check("size 字段可解析且 >0", bool(m) and int(m.group(1)) > 0, True)
print(f"  判决行原文: {[ln for ln in out.splitlines() if 'KERNELS-TGZ' in ln]}")

print("S8b 产物真的能解开来（事后归因的前提）")
ex = os.path.join(SANDBOX, "untar")
os.makedirs(ex, exist_ok=True)
r = subprocess.run(["tar", "xzf", os.path.join(SANDBOX, "k.tgz"), "-C", ex],
                   capture_output=True, text=True)
check("tar xzf rc", r.returncode, 0)
check("device_kernel.asc 在内", sorted(os.listdir(os.path.join(ex, "ascend|gemm"))),
      ["device_kernel.asc", "host_kernel.c"])

# ------------------------------------------------------------------ S9 空缓存
print("S9 缓存目录空（本窗一次都没编）⇒ KERNELS-TGZ-FAIL ⇒ 计一条 fail")
empty = os.path.join(SANDBOX, "empty")
os.makedirs(empty, exist_ok=True)
rc, out = harness(P(f'TGZ="{os.path.join(SANDBOX, "k2.tgz")}"',
                    f'TILELANG_CACHE_DIR="{empty}"',
                    'run_leg "KERNELS-TGZ" 180 \'^KERNELS-TGZ\' \'^KERNELS-TGZ-FAIL\' '
                    '\'^KERNELS-TGZ .*size=\' - \\', "  " + TGZ_CMD))
check("FAIL", out, "VERDICT-FAIL rc=1")
check("原因写得出来", out, "KERNELS-TGZ-FAIL")

# ------------------------------------------------------------------ S10/S11 BENCH 门
print("S10 SYS1_BENCH 未设 ⇒ BENCH 腿记 SKIP、不进验收、不影响退码")
rc, out = harness([BENCH_BLOCK])
check("SKIP 行", out, "CASE-BENCH SKIP SYS1_BENCH 未设")
check("skipped_legs=1 且 fail=0", out, "SUMMARY pass=0 fail=0 skipped_legs=1")
check("退 0", rc, 0)

print("S11 SYS1_BENCH=1 但 E2 交付件不在 ⇒ 仍只 SKIP（契约：静默跳过，不红）")
rc, out = harness([BENCH_BLOCK], envs={"SYS1_BENCH": "1"})
check("SKIP 原因换成文件不在", out, "CASE-BENCH SKIP 交付件不在")
check("退 0", rc, 0)

# ------------------------------------------------------------------ S12 腿级抓 trainstep 假绿
print("S12 [4] 腿：P2-STEP 判过了但 KERNEL-USED-FAIL ⇒ 腿判 FAIL（本元静默假绿在窗级也被抓住）")
rc, out = harness(P('run_leg "trainstep" 900 \'^P2-STEP|^KERNEL-USED|^NPU-REG|已回退 torch eager\' \\',
                    "  \'^P2-STEP-.*FAIL|^KERNEL-USED-FAIL|NPU-REG fail\' \'^KERNEL-USED-PASS\' - \\",
                    '  bash -c \'echo "P2-STEP-ascend loss0=0.9 loss1=0.8 reduced=True '
                    'params_grad_ok=True"; echo "KERNEL-USED-FAIL kernels=0 blockers=0 '
                    'fallbacks=0 target=ascend why=kernels_absent"; exit 5\''))
check("腿判 FAIL（不是 PASS）", out, "CASE-trainstep VERDICT-FAIL rc=5")
check("exit 1", rc, 1)

print("S12b 反向对照：KERNEL-USED-PASS + P2-STEP PASS ⇒ 腿判 PASS")
rc, out = harness(P('run_leg "trainstep" 900 \'^P2-STEP|^KERNEL-USED|^NPU-REG|已回退 torch eager\' \\',
                    "  \'^P2-STEP-.*FAIL|^KERNEL-USED-FAIL|NPU-REG fail\' \'^KERNEL-USED-PASS\' - \\",
                    '  bash -c \'echo "NPU-REG ok 6.0.0 target=ascend npu_avail=True"; '
                    'echo "P2-STEP-ascend reduced=True"; echo "KERNEL-USED-PASS kernels=13 '
                    'blockers=0 fallbacks=0 target=ascend"; exit 0\''))
check("腿判 PASS", out, "CASE-trainstep VERDICT-PASS rc=0")
check("NPU-REG 行在窗内可见（H3 的 [0] 探针与 [4] 腿呼应）", out, "NPU-REG ok 6.0.0")

print(f"B-CR-DONE {'PASS' if not FAILS else 'FAIL ' + ','.join(FAILS)}  (沙盒 {SANDBOX})")
sys.exit(1 if FAILS else 0)
