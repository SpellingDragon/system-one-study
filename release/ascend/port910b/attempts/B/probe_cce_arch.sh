#!/bin/bash
# attempts/B/probe_cce_arch.sh — 取 dav-2201 面 bisheng 的预定义宏 + 验证候选 CCE 原生件是否可见
set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1
export BISHENG_HOME=/usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler
B=$(command -v bisheng || echo $BISHENG_HOME/bin/bisheng)
echo "BISHENG=$B"

echo "=== [1] predefined macros (arch) ==="
cat > /tmp/agentB/empty.asc <<'EOF'
int dummy;
EOF
$B -O2 -std=c++20 --npu-arch=dav-2201 -E -dM /tmp/agentB/empty.asc 2>/dev/null | grep -i "DAV_\|NPU_ARCH\|CCE_AICORE" | sort

echo "=== [2] 候选搬运件可见性（逐个 TU 试编）==="
for sym in "copy_gm_to_ubuf((__ubuf__ uint8_t*)0, (__gm__ uint8_t*)0, (uint64_t)0)" \
           "copy_ubuf_to_gm((__gm__ uint8_t*)0, (__ubuf__ uint8_t*)0, (uint64_t)0)" \
           "copy_ubuf_to_gm((__gm__ uint8_t*)0, (__ubuf__ uint8_t*)0, (uint8_t)0, (uint16_t)0, (uint16_t)0, (uint16_t)0, (uint16_t)0)" \
           "copy_ubuf_to_gm_align((__gm__ uint8_t*)0, (__ubuf__ uint8_t*)0, (uint64_t)0, (uint64_t)0)" \
           "copy_gm_to_ubuf((__ubuf__ uint8_t*)0, (__gm__ uint8_t*)0, (uint8_t)0, (uint16_t)0, (uint16_t)0, (uint16_t)0, (uint16_t)0)" \
           "__cce_scalar::set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0)" \
           "__cce_scalar::wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0)" \
           "__cce_scalar::pipe_barrier(PIPE_V)" \
           "__cce_pipe_barrier(PIPE_V)"; do
  cat > /tmp/agentB/symtest.asc <<EOF
__global__ __vector__ void k() { $sym; }
EOF
  if $B -O2 -std=c++20 --npu-arch=dav-2201 --cce-aicore-only -c /tmp/agentB/symtest.asc -o /tmp/agentB/symtest.o 2>/tmp/agentB/symtest.err; then
    echo "OK   :: $sym"
  else
    echo "FAIL :: $sym"
    echo "       -> $(grep -m1 'error' /tmp/agentB/symtest.err | cut -c1-160)"
  fi
done
