#!/bin/bash
# attempts/B/probe_math_symbols.sh — dav-2201 面标量数学件可见性（一符号一 TU，避免诊断截断）
set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1
export BISHENG_HOME=/usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler
B=$BISHENG_HOME/bin/bisheng
TPL=$(python3 -c "import tilelang,os;print(os.path.dirname(os.path.join(os.path.dirname(tilelang.__file__),'src','tl_templates')))")
INC=/usr/local/Ascend/cann/aarch64-linux
D=/tmp/agentB/mp; mkdir -p $D

try() { # $1=name $2=expr
  cat > $D/t.asc <<EOF
#include <tl_templates/ascend/common.h>
extern "C" __global__ __vector__ void k(__gm__ float* G) {
  volatile float x = G[0];
  volatile float y = ($2);
  (void)y;
}
EOF
  if $B -O2 -fPIC -std=c++20 -DTL_PORT910B_NATIVE_TYPES -I$INC/asc/impl -I$INC/asc/include \
     -I"$TPL" --npu-arch=dav-2201 --cce-aicore-only -c $D/t.asc -o $D/t.o 2>$D/t.err; then
    echo "OK    :: $1"
  else
    echo "MISSING:: $1  -> $(grep -m1 ': error:' $D/t.err | sed 's/.*error: //' | cut -c1-110)"
  fi
}

for pair in "rsqrtf=rsqrtf(x)" "sqrtf=sqrtf(x)" "sqrt=sqrt(x)" "expf=expf(x)" "exp=exp(x)" \
            "logf=logf(x)" "log=log(x)" "powf=powf(x,2.0f)" "fabs=fabs(x)" "floorf=floorf(x)" \
            "ceilf=ceilf(x)" "tanhf=tanhf(x)" "__nv_rsqrtf=__nv_rsqrtf(x)" "__nv_expf=__nv_expf(x)" \
            "builtin_rsqrt=__builtin_sqrt(x)" "rsqrt_alt=1.0f/sqrt(x)"; do
  try "${pair%%=*}" "${pair#*=}"
done
echo "PROBE-MATH-DONE"
