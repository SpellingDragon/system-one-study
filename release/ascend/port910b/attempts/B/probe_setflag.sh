#!/bin/bash
# attempts/B/probe_setflag.sh — 910B 上 set_flag/wait_flag 的合法实参形态取证
# 背景：GAP-B 的 G2 写成 `__aicore__ inline void asc_sync_notify(pipe_t from,...)` 转调
#   __cce_scalar::set_flag(from,to,evt)，dav-2201 报
#   "error: the 1st parameter maybe need a type 'pipe_t'"。而 toolchain 自己的
#   __cce_set_flag(pipe_t p, pipe_t tp, event_t n) 把 p/tp (void) 丢弃、硬编码 PIPE_M->PIPE_V
#   （__clang_cce_aicore_functions.h:2796-2810）—— 强烈暗示该 builtin 要求 pipe 实参为**字面常量**。
# 本探针逐形态试编（prelude 放全局作用域，body 放核内），取完整诊断，定出可用的 G2 写法。
set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1
export BISHENG_HOME=/usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler
B=$BISHENG_HOME/bin/bisheng
TPL=$(python3 -c "import tilelang,os;print(os.path.dirname(os.path.join(os.path.dirname(tilelang.__file__),'src','tl_templates')))")
INC=/usr/local/Ascend/cann/aarch64-linux
D=/tmp/agentB/sf; mkdir -p $D

run() { # $1=tag  $2=global-prelude  $3=kernel-body
  cat > $D/v.asc <<EOF
#include <tl_templates/ascend/common.h>
$2
extern "C" __global__ __vector__ void k(__gm__ float* G) {
  (void)G;
$3
}
EOF
  echo "===== $1 ====="
  $B -O2 -fPIC -std=c++20 -DTL_PORT910B_NATIVE_TYPES -I$INC/asc/impl -I$INC/asc/include \
     -I"$TPL" --npu-arch=dav-2201 --cce-aicore-only -c $D/v.asc -o $D/v.o 2>&1 \
    | grep -E ": error:|maybe need|not allowed" | head -6
  if [ -f $D/v.o ]; then echo "RESULT: COMPILE-OK"; else echo "RESULT: COMPILE-FAIL"; fi
  rm -f $D/v.o
}

run "V1_literal_all" "" \
  '  __cce_scalar::set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
  __cce_scalar::wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);'
run "V2_wrapper_func" \
  '__aicore__ inline void sf(pipe_t a, pipe_t b, event_t c) { __cce_scalar::set_flag(a, b, c); }' \
  '  sf(PIPE_MTE2, PIPE_S, EVENT_ID0);'
run "V3_macro_literals" \
  '#define SN(f,t,e) __cce_scalar::set_flag(f, t, e)' \
  '  SN(PIPE_MTE2, PIPE_S, static_cast<event_t>(0));'
run "V4_macro_parenthesized" \
  '#define SNP(f,t,e) __cce_scalar::set_flag((f), (t), (e))' \
  '  SNP(PIPE_MTE2, PIPE_S, static_cast<event_t>(0));'
run "V5_nonzero_evt_static_cast" \
  '#define SN(f,t,e) __cce_scalar::set_flag(f, t, e)
#define WF(f,t,e) __cce_scalar::wait_flag(f, t, e)' \
  '  SN(PIPE_S, PIPE_MTE3, static_cast<event_t>(3));
  WF(PIPE_S, PIPE_MTE3, static_cast<event_t>(3));'
echo "PROBE-SETFLAG-DONE"
