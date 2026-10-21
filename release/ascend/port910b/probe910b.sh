#!/usr/bin/env bash
# probe910b.sh — TileLang 昇腾后端 910B 移植的**上卡探针**（INVENTORY.md §6 的 P1..P10 落地）
#
# 【做什么】在 910B/CANN 机器上依次裁决 44 个 D 类符号 + 墙0/墙3/墙4 的存在性真值，
#           输出一份可直接贴回 run notes 的 probe.log。
# 【怎么做】每个探针只做"找一个头 / 编一个最小 TU / 打一个宏值"这类零副作用动作；
#           全部命令都 `set +e` 捕获退出码与首个 error 行，不做任何 tilelang 编译。
# 【为什么】本机无 NPU 也无 CANN 头（已核实 /usr/local/Ascend 不存在），P0-1 的所有
#           "910B 等价物"都是推断级；只有这 10 个探针能把 D 类转成 A/B/C 判决。
#           顺序 P1→P2→P3 约 15 分钟即可拿到 90% 裁决，务必**先探针后改代码**。
#
# 用法（在 910B 机器上）：
#   ./probe910b.sh            # 跑 P1..P10（P10 需 tilelang 可 import）
#   ./probe910b.sh P1 P2 P3   # 只跑指定探针
#   REPO=/path/to/tilelang ASCEND_NPU_ARCH=dav-2201 ./probe910b.sh
# 本机（无 bisheng）执行会直接打印 ABORT 并退出 2 —— 这是设计行为，不是失败。

set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../../../.." && pwd)/tilelang}"  # codes/tilelang
TPL="${REPO}/src"                              # TL_DEVICE 侧 -I 根（env.py:648 同源）
ARCH="${ASCEND_NPU_ARCH:-dav-2201}"            # bisheng.py:93 读的就是这个环境变量
OUT="${PROBE_OUT:-$(dirname "$0")/probe.log}"
: > "$OUT"

say() { echo "$*" | tee -a "$OUT"; }
run() { # 白话: 执行一条命令，把退出码与首行 error 记进日志，绝不中断脚本
  local desc="$1"; shift
  set +e
  local tmp; tmp="$(mktemp)"
  timeout 90 "$@" >"$tmp" 2>&1  # P0-2a: 全探针统一超时守卫（round1 P3 挂起之鉴）
  local rc=$?
  set -e 2>/dev/null || true
  say "  [rc=$rc] $desc"
  if [ $rc -ne 0 ]; then
    grep -m3 -E "error:|fatal error:|No such file|undefined|unknown" "$tmp" | sed 's/^/      > /' | tee -a "$OUT"
  fi
  rm -f "$tmp"
  return $rc
}

# ── 前置：本机无编译器就拒绝执行（离线纪律） ─────────────────────────────────
if ! command -v bisheng >/dev/null 2>&1; then
  echo "ABORT: 未找到 bisheng —— 本脚本必须在 910B/CANN 环境执行（P0-1 纪律：本机零执行）"
  exit 2
fi
ASCEND_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
CANN_INC="${CANN_INC:-$(ls -d /usr/local/Ascend/ascend-toolkit/latest/include 2>/dev/null || echo /usr/local/Ascend/ascend-toolkit/latest/include)}"
[ -f /usr/local/Ascend/ascend-toolkit/latest/bin/set_env.bash ] && . /usr/local/Ascend/ascend-toolkit/latest/bin/set_env.bash 2>/dev/null || true
say "# probe910b  ARCH=$ARCH  ASCEND_HOME=$ASCEND_HOME  REPO=$REPO"

# ── P1 头存在性（裁决 墙0 / 墙4） ────────────────────────────────────────────
P1() {
  say "## P1 头存在性：950 方言头在 910B 镜像里到底有没有"
  for h in c_api/asc_simd.h simt_api/asc_simt.h simt_api/asc_bf16.h \
           simt_api/asc_fp16.h simt_api/asc_fp8.h simt_api/cooperative_groups.h; do
    local found; found="$(find "$ASCEND_HOME" -path "*$h" 2>/dev/null | head -1)"
    if [ -n "$found" ]; then say "  HIT  $h -> $found"; else say "  MISS $h"; fi
  done
  say "  CANN 版本线："
  cat "$ASCEND_HOME/ascend_toolkit_install.info" 2>/dev/null | sed 's/^/    /' | tee -a "$OUT"
  say "  pipe_*/simd intrinsic 家族目录数：$(find "$ASCEND_HOME" -name 'pipe_*.h' 2>/dev/null | wc -l)"
}

# ── P2 最小 TU 空编（裁决 A5 罩 + reduce.h 漏罩缺口） ────────────────────────
P2() {
  say "## P2 最小 TU：只 include common.h（910B 面，不定义 TL_ASCEND_SIMT）"
  local tu; tu="$(mktemp -d)/p2.cc"
  cat > "$tu" <<'EOF'
#include <tl_templates/ascend/common.h>
extern "C" __global__ __attribute__((aicore)) void probe_empty() {}
EOF
  say "  --- 默认（910B 面）---"
  run "common.h 910B 面" bisheng -I "$CANN_INC" --npu-arch="$ARCH" -I"$TPL" -std=c++17 -fsyntax-only "$tu"
  say "  --- 对照：950 面（应失败在缺头，若成功说明桩头已够）---"
  run "common.h + -DTL_ASCEND_SIMT" bisheng -I "$CANN_INC" --npu-arch="$ARCH" -DTL_ASCEND_SIMT=1 \
      -I"$TPL" -std=c++17 -fsyntax-only "$tu"
  say "  预期首个 error 落在 reduce.h（§3-A 残留缺口：它未被 TL_ASCEND_SIMT 罩）"
}

# ── P3 intrinsic 点名编（裁决 D1/D2/D3 = 24 个符号） ─────────────────────────
P3() {
  say "## P3 逐个点名：AIC 通路 + SIMD 寄存器类型 + 管道枚举"
  local syms="asc_init asc_copy_gm2ub_align asc_copy_ub2gm_align asc_copy_l12l0a \
asc_copy_l12l0b asc_copy_l0c2ub asc_mmad asc_lock asc_unlock asc_sync asc_sync_pipe \
vector_f32 vector_bool vector_u16 vector_uint16_t vector_uint32_t \
asc_loadalign asc_add asc_select PIPE_ALL PIPE_MTE2"
  local tu; tu="$(mktemp -d)/p3.cc"
  {
    echo '#include <tl_templates/ascend/common.h>'
    echo 'void probe() {'
    for s in $syms; do echo "  (void)&${s};"; done
    echo '}'
  } > "$tu"
  run "D1/D2/D3 符号点名（单次 TU，看未声明个数）" bisheng -I "$CANN_INC" --npu-arch="$ARCH" \
      -I"$TPL" -std=c++17 -fsyntax-only "$tu" || true
  say "  逐条判定：日志里 'undeclared identifier' 的即 D→C（910B 无此件）"
  for s in $syms; do
    echo "  CHECK $s"
  done | sed 's/^/  /' >> "$OUT"
}

# ── P4 legacy CCE 内建签名（裁决 C3 的 ::vcvt/::vpack/::vsstb） ──────────────
P4() {
  say "## P4 legacy CCE：nd2nz_copy.h:108-111 的四参调用形在 2201 上是否成立"
  local tu; tu="$(mktemp -d)/p4.cc"
  cat > "$tu" <<'EOF'
#include <tl_templates/ascend/common.h>
void probe(::vector_f32 src, ::vector_bool<8> mask) {
  auto c = ::vcvt<::bfloat16_t>(src, mask, ROUND_R, RS_DISABLE, PART_EVEN, LOWER);
  (void)c;
}
EOF
  run "::vcvt 四控制参数签名" bisheng -I "$CANN_INC" --npu-arch="$ARCH" -I"$TPL" -std=c++17 -fsyntax-only "$tu"
}

# ── P5 核拓扑（裁决 C7 的 __mix__(1,2) 与 sub_block） ────────────────────────
P5() {
  say "## P5 核拓扑：AIC:AIV 分组与 sub-block 身份"
  local tu; tu="$(mktemp -d)/p5.cc"
  cat > "$tu" <<'EOF'
extern "C" __global__ __mix__(1, 2) void probe_mix() {}
extern "C" __global__ __cube__ void probe_cube() {}
__device__ void probe_sub() { (void)asc_get_sub_block_id(); }
EOF
  run "__mix__(1,2)/__cube__/asc_get_sub_block_id" bisheng -I "$CANN_INC" --npu-arch="$ARCH" -I"$TPL" \
      -std=c++17 -fsyntax-only "$tu"
  say "  另需 host 侧打印实际核数：aclrtGetDeviceInfo 的 AIC/AIV 配比（人工执行）"
}

# ── P6 arch 宏与编译选项（裁决 墙3：C++ 侧无 arch gate 的补法） ──────────────
P6() {
  say "## P6 arch 宏：__NPU_ARCH__ 在 2201 下取值 + bisheng 支持的 arch 集"
  local tu; tu="$(mktemp -d)/p6.cc"
  printf '#include <cstdio>\nint main(){\n#ifdef __NPU_ARCH__\n  printf("__NPU_ARCH__=%%d\\n", (int)__NPU_ARCH__);\n#else\n  printf("__NPU_ARCH__ UNDEF\\n");\n#endif\n  return 0;\n}\n' > "$tu"
  run "取值" bisheng -I "$CANN_INC" --npu-arch="$ARCH" -std=c++17 -E "$tu"
  bisheng --help 2>&1 | grep -i -A2 "npu-arch" | sed 's/^/    /' | tee -a "$OUT"
  say "  结论用途：若 __NPU_ARCH__ 可靠，则 TL_ASCEND_SIMT 可由它自动推导，免掉 -D 注入链"
}

# ── P7 分形数值对拍（裁决 D4：C0 / TRANS_B 的静默错值风险） ──────────────────
P7() {
  say "## P7 分形数值：16x16 bf16 MMA 与 CPU 对拍（TRANS_B true/false 各一遍）"
  say "  本探针需要 tilelang 实跑，属 P0-2 里程碑件；给出最小 kernel 骨架供填写："
  cat <<'EOF' | sed 's/^/    /' | tee -a "$OUT"
# probe7_gemm16.py —— 910B 上执行；期望 max_abs_err == 0（bf16 输入 fp32 累加）
import tilelang, tilelang.language as T
@tilelang.jit
def kern():
    M = N = K = 16
    A = T.alloc_shared([M, K], "bfloat16"); B = T.alloc_shared([K, N], "bfloat16")
    C = T.alloc_fragment([M, N], "float32")
    ...  # 用 gemm_asc.py 的 AIC 主体，TRANS_B 两种取值各编一次
EOF
}

# ── P8 事件枚举（裁决 C8：运行期整数 -> event_t 是否被 2201 接受） ───────────
P8() {
  say "## P8 事件同步：static_cast<event_t>(3) 与 flag 上限"
  local tu; tu="$(mktemp -d)/p8.cc"
  cat > "$tu" <<'EOF'
#include <tl_templates/ascend/common.h>
void probe() {
  asc_sync_notify(PIPE_MTE2, PIPE_M, static_cast<event_t>(3));
  asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
  asc_sync_pipe(PIPE_ALL);
}
EOF
  run "event_t 运行期转换 + asc_sync_pipe" bisheng -I "$CANN_INC" --npu-arch="$ARCH" -I"$TPL" \
      -std=c++17 -fsyntax-only "$tu"
  say "  若失败：codegen_ascend.cc:1349 EmitHardEventSync_ 须改为编译期 HardEvent<X_Y> 模板参数"
}

# ── P9 兼容层自检（把 B 类判决在 bisheng 前端复现一遍） ──────────────────────
P9() {
  say "## P9 port910b_compat.h 在 bisheng 上的形状检查（主机侧已 PASS，见 host_selfcheck.cc）"
  local dir; dir="$(cd "$(dirname "$0")" && pwd)"
  run "compat 头 + 自检件" bisheng -I "$CANN_INC" --npu-arch="$ARCH" -std=c++17 -I"$dir" -I"$TPL" \
      -o /tmp/p01_sc_npu "$dir/host_selfcheck.cc"
  say "  冲突处理：报 redefinition 时逐个加 -DTL_PORT910B_SKIP_<name>（见 README §3）"
}

# ── P10 vecadd 端到端（P0-2 里程碑） ────────────────────────────────────────
P10() {
  say "## P10 最小 vecadd：bf16 T.copy + T.Kernel 实编实跑数值对"
  python3 - <<'EOF' 2>&1 | tee -a "$OUT"
import tilelang, tilelang.language as T
N = 1024
@tilelang.jit(out_idx=[2])
def make():
    @T.prim_func
    def main(A: T.Tensor((N,), "bfloat16"), B: T.Tensor((N,), "bfloat16"),
             C: T.Tensor((N,), "bfloat16")):
        with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
            tx = T.get_thread_binding()
            for i in T.Parallel(128):
                C[bx * 128 + i] = A[bx * 128 + i] + B[bx * 128 + i]
    return main
k = make()
import torch
a = torch.randn(N, dtype=torch.float32).to("npu").bfloat16()
b = torch.randn(N, dtype=torch.float32).to("npu").bfloat16()
c = k(a, b)
ref = (a.float() + b.float()).bfloat16()
print("P10 max_abs_err =", (c.float() - ref.float()).abs().max().item())
EOF
}

# ── 调度 ─────────────────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
  for p in P1 P2 P3 P4 P5 P6 P7 P8 P9 P10; do "$p"; done
else
  for p in "$@"; do "$p"; done
fi
say "# probe910b done -> $OUT"
say "# 回填要求：把 $OUT 原文贴进 run notes，并按结果更新 INVENTORY.md §3-D 的分类计数"
