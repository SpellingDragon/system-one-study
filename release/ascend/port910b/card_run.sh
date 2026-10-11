#!/bin/bash
# card_run.sh — 910B 单窗统一验收（P1-1i 修订：判决件必须量到产品件）
# 幂等、自包含、setsid 后台跑；verdict 行可直接 grep。
#
# ── P1-1i 三处修法（对应 wave2 误判根因，逐条可复核）────────────────────────
# ① [3] 段 target 列表**先跑产品件**（linear_prod / dw_prod / readout_prod：经
#    `ascend/kernels/*_kernel.py` 入口件 + ascend_env 路由，即 train_step 真正用的那条路），
#    原 attempts 内联 case 改名 `*_intrinsic` 作**性能对照轨**保留（判的是 §12 intrinsic 路，
#    从来不是产品件——这正是本元缺陷的来源）。判决行前缀：产品件 `Y-<case>-PROD-*`，
#    对照件 `NUM-<case>-*`。
# ② [2] 段 overlay 落点：编译 `-I` 根是 `<site-packages>/tilelang/src`，而桩头里的引用形如
#    `#include "c_api/asc_simd.h"`（见 patches/common.h:11、debug.h:12、gemm.h:3）⇒ c_api/simt_api
#    必须落 `$PIP_SRC = tilelang/src` 下。旧写法 `PIP_SRC=$(dirname "$TPL")` = `tilelang/src/tl_templates`
#    ⇒ wave2 首轮六件全灭于 `'c_api/asc_simd.h' file not found`。
#    （同一形态在 bundle_ondemand.sh:37 用 `$(dirname "$TPL")/../$d` 是对的，此处按字面写清。）
# ③ [2] 段起跑前**断言**桩头已就位：缺 `$PIP_SRC/c_api/asc_simd.h` 或任一同源头 ⇒
#    `FATAL overlay-missing: <路径>` 并 `exit 1`（旧版 `cp ... || true` 会把缺件吞成静默失败，
#    再到 [3] 炸成一堆看不懂的 file-not-found）。
#
# ── p2-13 代理 B 四处修法（卡窗可信化；病灶全部来自 reconcile/A3_PREFLIGHT.md §5/§6）───
# ① H1（真阻塞）：[0] 段最前加 **torch_npu 设备注册探针**，打 `NPU-REG ok|fail <version>`。
#    torch 内建清单里没有 npu，注册是 torch_npu 的 import 副作用；缺它则 [3] 产品腿只出
#    `Y-*-ASC-COMPILE-PASS note=无设备⇒只判编译`（长得像 PASS）、[4] 整腿塌成一行裸 RuntimeError。
#    产品侧配套修法在 `ascend/train_step.py`（ascend 轨条件 import，cpu 轨不 import）。
# ② H2：腿内过滤由 `grep -E "Y-$t|FAIL|Error:|error:|aicore" | head -6` 改**两趟**——先判决
#    前缀行（额度 30），再 `error:|FATAL|Traceback|RuntimeError|AssertionError`（额度 8）。
#    `aicore` 已去掉：它命中 bisheng 命令行里的 `-cce-aicore-*`，每件失败件白吃 3 行，
#    两件同灭时（readout_prod 实测）判决行被完全挤掉。
# ③ H3：窗末 [5] 自扫一遍完整判决前缀（OVERLAY-OK|INJECT-OK|^FATAL|CASE-|npu_avail|NPU-REG|
#    RuntimeError|Traceback|KERNEL-USED|ASC-|BENCH-|KERNELS-TGZ|note=无设备）——旧正则漏掉
#    的正是"整窗唯一的设备在场判据"和 [2] 的自证三行；`ASC-*` 这一族以前不在任何腿的过滤器里。
# ④ H4：`RUNNER-DONE` 改为**带计数**（pass/fail/skipped_legs/degraded）且 fail>0 ⇒ 非零退出；
#    每条腿按 `PASS / DEGRADED / FAIL / FAIL-NO-VERDICT / FAIL-TIMEOUT / FAIL-EXIT<rc>` 归档，
#    **没有判决行一律 FAIL**（不许空转，与 compile_all_asc 的门禁口径同构）；
#    PASS 还额外要求 rc=0（"打了 PASS 行但退码非 0"也是失败，单列 FAIL-EXIT<rc> 便于归因）。
#    DEGRADED = `note=无设备⇒只判编译` 那种"PASS 形状的降级"，同样计进 fail（A3 增采 B）。
#    对照轨（`NUM-*` / performance-track）的腿记 `track=advisory`：失败单列 `advisory_fail=`、
#    不顶退码——与 compile_all_asc 里 `ASC-PERF-*` 默认 non-blocking 的既有口径同构
#    （Y 口径：NUM-* 是 §12/intrinsic 对照，不可当正确性判决引用）。产品面
#    （`Y-*-PROD-*` / `P2-STEP-*` / `ASC-products` 门禁 / KERNEL-USED）全部计门禁。
# ⑤ 增采 C：收尾把 $TILELANG_CACHE_DIR 打成 /tmp/sys1_kernels.tgz 并报大小与件数——零额外
#    编译即可事后验 §12 位序 / T.Cube 是否在场。增采 D：[3] 末加一条 compile_all_asc 腿。
#
# ── 本机自证钩子（默认全部关闭 ⇒ 真机行为与旧版逐字节一致）──────────────────
#   SYS1_DIR=…           [1] 段 clone 的目标目录（默认 /tmp/sys1）
#   CARDRUN_SKIP_CLONE=1 跳过 git clone，用已就位的 SYS1_DIR（host 无网/禁 git 时自证用）
#   CARDRUN_STOP_AFTER=N 跑完第 N 步即出 SUMMARY（host 自证 [2] 段用 N=2）
#   TLROOT_OVERRIDE=…    覆写 tilelang 根（host 自证时指到一个空沙盒，避免污染 .venv）
#   Y_SELFTEST_DROP_CAPI=1 overlay 完成后**故意删掉** c_api/asc_simd.h ⇒ ③ 的断言必须响
#   CARDRUN_SKIP_BISHENG=1 跳过 bisheng 选项注入（host 自证用，免得改写本机 .venv 里的 bisheng.py）
#   CARDRUN_LOGDIR=…       每腿原文日志目录（默认 /tmp/sys1_legs；计数与事后归因都看它）
#   CARDRUN_TGZ=…          增采 C 的产物包落点（默认 /tmp/sys1_kernels.tgz）
#   SYS1_BENCH=1           [4b] 可选 bench 腿：跑 release/ascend/bench_card.py（E2 交付件）；
#                          未设或文件不在 ⇒ 只记 SKIP，不进验收、不影响退码（验收优先）
#   SYS1_SKIP_FP16_LEG=1   [4] 显式砍掉 fp16 档第二趟（**默认跑两趟**）。预算兜底用：
#                          砍了必须打得见（CASE-trainstep-fp16 SKIP reason=…）且计入
#                          skipped_legs——绝不许静默消失（A3 §4 的 20 分钟窗据此复核）
#
set -uo pipefail
export TILELANG_CACHE_DIR=$(mktemp -d)   # 缓存纪律：key 不含模板内容，必须全新；[3][4] 共享此目录=编译复用
for c in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann-8.5.2 /usr/local/Ascend/cann-*; do
  [ -f "$c/set_env.sh" ] && { set +u; source "$c/set_env.sh"; set -u; break; }
done
CCEC=$(find /usr/local/Ascend -path "*ccec_compiler/bin/bisheng" 2>/dev/null | head -1)
[ -n "$CCEC" ] && export BISHENG_HOME=$(dirname "$(dirname "$CCEC")")
export ASCEND_NPU_ARCH=dav-2201

SYS1=${SYS1_DIR:-/tmp/sys1}
STOP_AFTER=${CARDRUN_STOP_AFTER:-}

# --------------------------------------------------------------------------- 腿执行器（H2/H4）
PASS_N=0; FAIL_N=0; SKIP_N=0; DEG_N=0; ADV_N=0; LEG_TOTAL=0
LEG_TRACK=gating                          # gating=计门禁；advisory=对照轨（失败单列、不顶退码）
RUNLOG=${CARDRUN_LOGDIR:-/tmp/sys1_legs}
mkdir -p "$RUNLOG" 2>/dev/null || true
rm -f "$RUNLOG"/*.log "$RUNLOG"/tally.txt     # 幂等：不清上次的账会把这次的计数顶歪

bench_planned() { [ -n "${SYS1_BENCH:-}" ] && echo 1 || echo 0; }
BENCH_PLANNED=$(bench_planned)

_bump_fail() {
  if [ "$LEG_TRACK" = "advisory" ]; then ADV_N=$((ADV_N+1)); else FAIL_N=$((FAIL_N+1)); fi
}

# run_leg <标签> <超时秒> <判决前缀正则> <FAIL 正则> <PASS 正则> <降级正则|-> <命令...>
#   · 判决行与错误行分两趟打（H2）；原文整份落 $RUNLOG/<标签>.log，屏上截断不丢证据；
#   · 计数落点：PASS / DEGRADED(计进 fail) / FAIL / FAIL-NO-VERDICT / FAIL-TIMEOUT / FAIL-EXIT<rc>；
#   · LEG_TRACK=advisory 时失败进 ADV_N（对照轨，不顶退码），否则进 FAIL_N（顶退码）。
run_leg() {
  local tag=$1 tmo=$2 disp_re=$3 fail_re=$4 pass_re=$5 deg_re=$6 log rc verdict
  shift 6
  LEG_TOTAL=$((LEG_TOTAL+1))
  log="$RUNLOG/$tag.log"
  echo "CASE-$tag track=$LEG_TRACK tmo=${tmo}s"
  timeout "$tmo" "$@" >"$log" 2>&1
  rc=$?
  grep -E "$disp_re" "$log" 2>/dev/null | head -30
  grep -E "error:|FATAL|Traceback|RuntimeError|AssertionError" "$log" 2>/dev/null | head -8
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then verdict=FAIL-TIMEOUT
  elif grep -qE "$fail_re" "$log" 2>/dev/null; then verdict=FAIL
  elif [ "$deg_re" != "-" ] && grep -qE "$deg_re" "$log" 2>/dev/null; then verdict=DEGRADED
  elif grep -qE "$pass_re" "$log" 2>/dev/null; then
    if [ $rc -eq 0 ]; then verdict=PASS; else verdict="FAIL-EXIT$rc"; fi
  else verdict=FAIL-NO-VERDICT; fi
  case "$verdict" in
    PASS) PASS_N=$((PASS_N+1)) ;;
    DEGRADED) DEG_N=$((DEG_N+1)); _bump_fail ;;
    *) _bump_fail ;;
  esac
  echo "CASE-$tag VERDICT-$verdict rc=$rc track=$LEG_TRACK tally(pass=$PASS_N fail=$FAIL_N adv=$ADV_N) log=$log"
  echo "CASE-$tag VERDICT-$verdict rc=$rc track=$LEG_TRACK" >>"$RUNLOG/tally.txt"
}

# 唯一出口：把计数摊开再决定退码（"跑到底"绝不等于"过了"）
runner_done() {
  local code=0
  [ $FAIL_N -gt 0 ] && code=1
  echo "=== [5] verdict summary ==="
  [ -f "$RUNLOG/tally.txt" ] && cat "$RUNLOG/tally.txt"
  # H3：窗末自扫（腿内被 head 截断的判决行，这里按完整前缀族补回来）
  echo "VERDICT-SCAN-BEGIN"
  grep -hE 'OVERLAY-OK|INJECT-OK|^FATAL|CASE-|npu_avail|NPU-REG|RuntimeError|Traceback|KERNEL-USED|ASC-|BENCH-|KERNELS-TGZ|note=无设备' \
    "$RUNLOG"/*.log 2>/dev/null | head -80
  echo "VERDICT-SCAN-END"
  echo "SUMMARY-DEGRADED note=无设备 命中=$(cat "$RUNLOG"/*.log 2>/dev/null | grep -c 'note=无设备') 行"
  echo "LOGS $RUNLOG (每条腿原文一份，补看：grep -E '^Y-|^NUM-|^ASC-|^P2-STEP|^KERNEL-USED|^DTYPE-TRACE' $RUNLOG/<腿>.log)"
  echo "RUNNER-DONE SUMMARY pass=$PASS_N fail=$FAIL_N skipped_legs=$SKIP_N degraded=$DEG_N advisory_fail=$ADV_N exit=$code"
  exit $code
}

# 计划内的腿数（STOP_AFTER 分支要如实报"没跑几腿"，不能报 0）
LEGS_PROD=(linear_prod dw_prod readout_prod)
LEGS_PERF=(gemm_l1_intrinsic dw_intrinsic readout_intrinsic addln_intrinsic gdn)
N_LEGS_3=$(( ${#LEGS_PROD[@]} + ${#LEGS_PERF[@]} + 2 ))     # + delta 腿 + 增采 D 的 ASC-products 腿
# G（p2-13）：[4] 的 fp16 档第二趟是**计划内腿**，必须进 N_LEGS_4，否则 STOP_AFTER=2/3
# 分支会把"少跑了一条腿"报成"一条没少"。显式砍腿时它由 [4] 自己记 SKIP ⇒ 此处不再计（不重复计账）。
FP16_PLANNED=1
if [ "${SYS1_SKIP_FP16_LEG:-0}" = "1" ]; then FP16_PLANNED=0; fi
N_LEGS_4=$(( 2 + BENCH_PLANNED + FP16_PLANNED ))   # [4] train_step(fp32) + train_step(fp16) + [4c] kernels-tgz +（可选）[4b] bench

echo "=== [0] env ==="
# H1 探针（**必须早于任何腿**）：整窗唯一能把"Y-*-ASC-COMPILE-PASS note=无设备"与
# "Y-*-PROD-PASS rel=…"区分开的先验。判读口径：
#   NPU-REG ok <ver> avail=True   ⇒ [3] 的 PROD-PASS 是数值判决、[4] 能真发射
#   NPU-REG fail <原因>            ⇒ 本窗数值面作废（别烧窗），先补 torch_npu/驱动再开
python3 - <<'PYEOF_PROBE' 2>&1 | tail -6
import sys
try:
    import torch
except Exception as exc:
    print("NPU-REG fail no-torch:%s:%s" % (type(exc).__name__, str(exc)[:120])); sys.exit(0)
try:
    import torch_npu
    ver = getattr(torch_npu, "__version__", "unknown")
except Exception as exc:
    print("NPU-REG fail torch_npu:%s:%s" % (type(exc).__name__, str(exc)[:120]))
    print("NPU-REG-CONSEQUENCE 未注册 npu 设备 ⇒ train_step 走不到 npu、产品腿只出编译判")
    sys.exit(0)
try:
    torch.device("npu"); ok = True
except Exception as exc:
    ok = False; print("NPU-REG device-string:%s:%s" % (type(exc).__name__, str(exc)[:120]))
try:
    avail, cnt = bool(torch.npu.is_available()), int(torch.npu.device_count())
except Exception:
    avail, cnt = False, -1
print("NPU-REG %s %s avail=%s count=%s torch=%s" % ("ok" if ok else "fail", ver, avail, cnt, torch.__version__))
PYEOF_PROBE
python3 -c "import torch; print('torch', torch.__version__, 'npu_avail', hasattr(torch,'npu') and torch.npu.is_available())"
python3 -c "import tilelang" 2>/dev/null || pip3 install -q -i https://pypi.tuna.tsinghua.edu.cn/simple tilelang 2>&1 | tail -1
python3 -c "import tilelang; print('tilelang', tilelang.__version__)"

echo "=== [1] repo @latest ==="
if [ "${CARDRUN_SKIP_CLONE:-0}" = "1" ]; then
  echo "SKIP-CLONE（自证钩子）：用已就位的 $SYS1"
else
  rm -rf "$SYS1"
  git clone -q --depth 1 https://github.com/SpellingDragon/system-one-study.git "$SYS1" || { echo "FATAL clone"; exit 1; }
fi
cd "$SYS1" && git log --oneline -1 || echo "[note] SKIP-CLONE 且无 .git ⇒ 无 log 行"

echo "=== [2] overlay 910B templates + bisheng inject ==="
cd "$SYS1/release/ascend/port910b"
# tilelang 包根：`-I` 根是 $TLROOT/src，桩头按 $TLROOT/src/<c_api|simt_api>/<name>.h 解析
TLROOT=${TLROOT_OVERRIDE:-$(python3 -c "import tilelang,os;print(os.path.dirname(tilelang.__file__))")}
TPL=$TLROOT/src/tl_templates/ascend
PIP_SRC=$TLROOT/src                       # ← P1-1i 修正点（旧：$(dirname "$TPL") = tl_templates，错一层）
echo "TLROOT=$TLROOT"
echo "PIP_SRC(桩头落点根)=$PIP_SRC   TPL(模板落点)=$TPL"
mkdir -p "$TPL" "$PIP_SRC/c_api" "$PIP_SRC/simt_api"
cp -f patches/port910b_compat.h patches/gemm.h patches/common.h patches/debug.h \
      patches/dcache_bypass.h patches/numeric_limits.h "$TPL/" || { echo "FATAL overlay-cp(templates)"; exit 1; }
cp -f patches/c_api/*.h "$PIP_SRC/c_api/"     || { echo "FATAL overlay-cp(c_api)"; exit 1; }
cp -f patches/simt_api/*.h "$PIP_SRC/simt_api/" || { echo "FATAL overlay-cp(simt_api)"; exit 1; }
if [ "${Y_SELFTEST_DROP_CAPI:-0}" = "1" ]; then
  rm -f "$PIP_SRC/c_api/asc_simd.h"
  echo "[selftest] 已按 Y_SELFTEST_DROP_CAPI=1 删掉 c_api/asc_simd.h ⇒ 下面的断言应当响"
fi
# ③ 存在性断言（防再犯）：cp 的返回码不够，必须按“编译期真正会找的那个路径”逐个验文件
for f in "$TPL/port910b_compat.h" "$TPL/common.h" \
         "$PIP_SRC/c_api/asc_simd.h" \
         "$PIP_SRC/simt_api/asc_simt.h" "$PIP_SRC/simt_api/asc_fp16.h" \
         "$PIP_SRC/simt_api/asc_bf16.h" "$PIP_SRC/simt_api/asc_fp8.h"; do
  [ -f "$f" ] || { echo "FATAL overlay-missing: $f  ← 桩头没落到 -I 根（-I 根=$TLROOT/src，引用形如 \"c_api/asc_simd.h\"）"; exit 1; }
done
ls -l "$PIP_SRC/c_api/asc_simd.h" || { echo "FATAL overlay-missing: $PIP_SRC/c_api/asc_simd.h"; exit 1; }
echo "OVERLAY-OK headers=$(ls "$PIP_SRC/c_api" "$PIP_SRC/simt_api" | grep -c '\.h$') tpl=$(ls "$TPL"/*.h | wc -l | tr -d ' ')"
echo "compat §12+fill: $(grep -c 'asc_fill_l1\|(uint8_t)sid, false' "$TPL/port910b_compat.h")"
if [ "${CARDRUN_SKIP_BISHENG:-0}" = "1" ]; then
  echo "[selftest] SKIP-BISHENG：不碰本机 tilelang/contrib/bisheng.py（host 自证专用，真机必须注入）"
else
  python3 patches/patch_bisheng.py && echo INJECT-OK || { echo "FATAL bisheng-inject"; exit 1; }
fi
if [ -n "$STOP_AFTER" ] && [ "$STOP_AFTER" -ge 2 ] 2>/dev/null; then
  echo "STOP-AFTER-2 OVERLAY-OK"
  SKIP_N=$((N_LEGS_3 + N_LEGS_4)); runner_done
fi

echo "=== [3] numerics（产品件判决 = 下窗要的东西；intrinsic = 性能对照轨）==="
cd "$SYS1/release/ascend/port910b"
# 产品件（入口件 → ascend_env 路由 → *_asc.py 生产正文）。三态判决：
#   ^Y-<case>-PROD-PASS rel=…            数值判（真机才可能出）⇒ PASS
#   ^Y-<case>-ASC-COMPILE-PASS note=无设备⇒只判编译 ⇒ DEGRADED（计进 fail，A3 增采 B）
#   ^Y-<case>-PROD-FAIL                  ⇒ FAIL
LEG_TRACK=gating
for t in "${LEGS_PROD[@]}"; do
  # 预算 900s：真机首次编译+发射的余量（容器冷编译实测 4–7s，卡上留足）
  run_leg "P11i-$t" 900 "^Y-$t" "^Y-$t-PROD-FAIL" "^Y-$t-PROD-PASS" "^Y-$t-ASC-COMPILE-PASS" \
    env DMLAYA_ASCEND_TARGET=ascend python3 run_numerics.py --only "$t" --track card
done
# 对照轨（attempts/§12 intrinsic 正文，非产品件）：advisory——失败单列、不顶退码，
# 但一定出现在 tally/SUMMARY 里（旧版把它和判决混在一个 head -4 里，两边都看不清）
LEG_TRACK=advisory
for t in "${LEGS_PERF[@]}"; do
  run_leg "PERF-$t" 500 "^NUM-$t" "^NUM-$t-FAIL" "^NUM-$t[ -]PASS" - \
    python3 run_numerics.py --only "$t"
done
LEG_TRACK=gating
run_leg "delta" 500 "^NUM-delta_fwd" "^NUM-delta_fwd FAIL" "^NUM-delta_fwd[ -]PASS" - \
  python3 run_delta_card.py
# 增采 D：A2 的单一入口（九件接口 + 三件产品 + 对照腿，20 行判决；门禁=门禁失败数=0）
cd "$SYS1/release"
run_leg "ASC-products" 1200 "^ASC-|^—— |^P2-STEP" \
  "^ASC-products FAIL|^ASC-PROD-skipped FAIL|门禁失败数=[1-9]" "门禁失败数=0$" - \
  env DMLAYA_ASCEND_TARGET=ascend python3 ascend/compile_all_asc.py --target ascend --products
cd "$SYS1/release/ascend/port910b"
if [ -n "$STOP_AFTER" ] && [ "$STOP_AFTER" -ge 3 ] 2>/dev/null; then
  echo "STOP-AFTER-3"; SKIP_N=$N_LEGS_4; runner_done
fi

echo "=== [4] mixed-stack train step on npu (P2 首步 + P1-1l 内核自证 + F2 位宽档) ==="
cd "$SYS1/release"
# PASS 需要**两样**：接线判（P2-STEP-*）+ 内核启用判（KERNEL-USED-PASS blockers=0 fallbacks=0）。
# 只看 P2-STEP 就是本元静默假绿：内核一次没走、整步落 eager，loss 照样降、旧版照样 PASS 退 0。
# ⚠ 这条 fp32 腿的**标签/超时/三个正则/命令**与 B 写面逐字同，一个新 token 都没加
#   （F2 §7：卡窗加档是"一行参数"的事，fp32 档行为不许被顺手改掉）。改前改后的窗级差异
#   应当**只多出下面 trainstep-fp16 一条腿**，其余都算越权。
LEG_TRACK=gating
run_leg "trainstep" 900 '^P2-STEP|^KERNEL-USED|^NPU-REG|已回退 torch eager' \
  '^P2-STEP-.*FAIL|^KERNEL-USED-FAIL|NPU-REG fail' '^KERNEL-USED-PASS' - \
  env DMLAYA_ASCEND_TARGET=ascend python3 ascend/train_step.py --target ascend

# G（p2-13）：**同窗第二趟 = F2 的 fp16 档**（F2_RESULT §7 明示"card_run.sh 加不加 --dtype fp16
# 属 B 写面、判决行格式已兼容"）。一次开窗同时验 fp32/fp16 **两种模具形状**，不多开一窗。
# 三条老规矩一条不破：
#   ① LEG_TRACK=gating ⇒ 退码优先级 2>3>6>5 里的 **6（EXIT_DTYPE_DISCIPLINE，位宽纪律钉违规）
#      计 gating FAIL 并顶窗**。advisory 只归 NUM-*/PERF-* 对照轨（Y 口径：不可当正确性判决引用），
#      fp16 属产品面 ⇒ 严禁挂 advisory；
#   ② 判决行仍走 B 的**两趟 grep**，不新造通道：第一趟 disp_re 加 `^DTYPE-TRACE`（F2 新增的位宽
#      判决，fp16 腿专属——fp32 档它也会打，但 fp32 腿"逐字不动"的优先级更高，故只在 fp16 腿补）；
#      fail_re 加 `^DTYPE-TRACE-FAIL` 与 `fp32_accum=false`，让 rc=6 **在判决级就红**，
#      不只靠 rc 校验（真件 rc=6 两样同红：有 DTYPE-TRACE-FAIL 行 + rc≠0，双保险）；
#   ③ 第二趟错误 grep 的匹配词一个都没动（`error:|FATAL|Traceback|RuntimeError|AssertionError`），
#      **没把 `aicore` 加回来**——它命中 bisheng 命令行的 `-cce-aicore-*` 会挤掉判决行（B §3.1/H2）。
# 无卡时它不可能静默 PASS：train_step 在 NPU-REG 门上就 `P2-STEP-ascend FAIL: NPU-REG-FAIL` + 退 4
#   （B §5 同形）⇒ 既凑不出 `KERNEL-USED-PASS`，也凑不出 rc=0 ⇒ verdict=FAIL。
#   （不是 DEGRADED：train_step 没有"note=无设备⇒只判编译"那种 PASS 形状的降级，那是 run_numerics 面。）
# 预算复核（A3 §4 的 20 分钟窗；账见 reconcile/G_RESULT.md §4）：fp16 模具键含位宽段
#   （`…dfloat16` / `…qfloat16afloat32`）⇒ **与 fp32 不共享缓存**，本腿另付一次 13 键冷编
#   （容器实测 ≈20s；F2 host 13 枚模具 19.1s）+ 一次 npu 上下文初始化（3–15s）+ 真发射对拍（<30s）
#   ⇒ 净链 2.5–6 分钟 → **3.5–8 分钟**；腿级 900s 超时富余仍 ≈15×，20 分钟窗含一次 fixup 补跑仍够
#   ⇒ **默认跑两趟，不静默砍腿**；确需省时间用 `SYS1_SKIP_FP16_LEG=1` 显式砍（打 SKIP reason= 并计账）。
if [ "${SYS1_SKIP_FP16_LEG:-0}" = "1" ]; then
  echo "CASE-trainstep-fp16 SKIP reason=SYS1_SKIP_FP16_LEG=1 显式砍腿 ⇒ fp16 档本窗未验"
  SKIP_N=$((SKIP_N+1))
else
  run_leg "trainstep-fp16" 900 '^P2-STEP|^KERNEL-USED|^NPU-REG|^DTYPE-TRACE|已回退 torch eager' \
    '^P2-STEP-.*FAIL|^KERNEL-USED-FAIL|^DTYPE-TRACE-FAIL|^DTYPE-TRACE fp16 .*fp32_accum=false|NPU-REG fail' \
    '^KERNEL-USED-PASS' - \
    env DMLAYA_ASCEND_TARGET=ascend python3 ascend/train_step.py --target ascend --dtype fp16
fi

echo "=== [4b] bench（可选腿：SYS1_BENCH=1 且 E2 交付件在场才跑）==="
if [ -z "${SYS1_BENCH:-}" ]; then
  echo "CASE-BENCH SKIP SYS1_BENCH 未设（默认不进窗，保验收优先）"
  SKIP_N=$((SKIP_N+1))
elif [ ! -f "$SYS1/release/ascend/bench_card.py" ]; then
  echo "CASE-BENCH SKIP 交付件不在：$SYS1/release/ascend/bench_card.py（E2 未入库）"
  SKIP_N=$((SKIP_N+1))
else
  cd "$SYS1/release/ascend/port910b"
  run_leg "BENCH" 900 '^BENCH-' '^BENCH-.*FAIL' '^BENCH-.*PASS' - \
    python3 ../bench_card.py
fi

echo "=== [4c] 增采 C：内核缓存打包（零额外编译，事后验 §12 位序 / T.Cube）==="
# 每键目录自带 device_kernel.asc + host_kernel.c + manifest.json + params.json（A3 §6-C）。
# kernels=0 ⇒ 本窗压根没编过任何东西（缓存目录是 mktemp 全新的，不可能"本来就空"）⇒ 计一条 fail。
TGZ=${CARDRUN_TGZ:-/tmp/sys1_kernels.tgz}
run_leg "KERNELS-TGZ" 180 '^KERNELS-TGZ' '^KERNELS-TGZ-FAIL' '^KERNELS-TGZ .*size=' - \
  bash -c 'd="$1"; t="$2"; n=$(find "$d" -type f 2>/dev/null | wc -l | tr -d " "); a=$(find "$d" -name device_kernel.asc 2>/dev/null | wc -l | tr -d " "); if [ "$n" = "0" ]; then echo "KERNELS-TGZ-FAIL dir=$d entries=0 ⇒ 本窗一次都没编译"; exit 1; fi; if ! tar czf "$t" -C "$d" . 2>/dev/null; then echo "KERNELS-TGZ-FAIL tar-failed dir=$d"; exit 1; fi; echo "KERNELS-TGZ $t size=$(wc -c <"$t" | tr -d " ")B entries=$n kernels=$a"' _ "$TILELANG_CACHE_DIR" "$TGZ"

runner_done
