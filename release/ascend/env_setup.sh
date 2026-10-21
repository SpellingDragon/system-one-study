#!/usr/bin/env bash
# =============================================================================
# ascend/env_setup.sh —— 昇腾（Ascend NPU）运行环境安装脚本（P2 / p2-13 域 R1 交付）
#
# 【做什么】
#   在一台没有昇腾软件栈的机器上，按固定顺序把 CANN / torch_npu / TileLang /
#   TileKernels 四件套装到 System-One P2 要求的版本，并落一个可 source 的环境文件。
#
# 【怎么做】
#   一张版本 pin 表 + 一条九步流水线（体检 -> 复用 set_env -> CANN -> 环境变量落盘
#   -> torch -> torch_npu -> TileLang 源码构建 -> 参照件 -> 自检），凡是会产生副
#   作用的命令一律过 run()：--dry-run 只打印"将执行什么、装成哪一版"，不碰系统；
#   真跑必须显式带 --yes，脚本本身从不阻塞等输入。
#
# 【为什么】
#   被否方案一：把安装步骤写成 README 让人手敲 —— 手敲不可复现，云上短租按小时
#   计费，装错一次白烧一台机，故必须脚本化且支持先空跑自查。
#   被否方案二：直接 pip install tilelang 了事 —— 本机（macOS）实测已装的
#   tilelang 0.1.15 轮子缺 AscendC 算子注册，编译方言件报
#   "Operator tl.tileop.ascend_copy is not registered"；查 tilelang 的
#   CMakeLists.txt 与 docs/get_started/Installation.md 得知 USE_ASCEND 只在 Linux
#   默认 ON、macOS 默认 OFF，故此处强制源码构建并把开关写死。
#   被否方案三：随手 pip 最新版 torch_npu —— torch_npu 需与 torch 主版本、CANN
#   版本三元组对齐，装错会 import 期崩，故显式 pin，装不上就直接退出而非回退。
#
# 用法：
#   bash ascend/env_setup.sh --dry-run                 # 本地空跑，只出清单（应 exit 0）
#   bash ascend/env_setup.sh --yes [--device 910b|950] # 云端真装
#   bash ascend/env_setup.sh --plan                    # 只出 pin 表
# 退出码：0 = 成功（含 dry-run / plan）；1 = 参数错或未加 --yes；2 = 真装途中失败。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_ARCH="$(uname -m)"

# ---------------------------- 版本 pin 表（唯一真源） ----------------------------
# 口径来源：TileKernels README/pyproject（CANN>=9.2.0、TileLang>=0.1.15）、
# tilelang docs/get_started/Installation.md（USE_ASCEND 默认值）、
# tilelang/contrib/bisheng.py（ASCEND_NPU_ARCH 默认 dav-3510）。
PYTHON_MIN="3.10"
CANN_VERSION="${CANN_VERSION:-9.2.0}"                 # 昇腾异构计算架构（toolkit + kernels）
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"               # 与 torch_npu 配套的主版本
TORCH_NPU_VERSION="${TORCH_NPU_VERSION:-2.5.1.post0}"
# ↑ torch_npu 可用版本随 CANN 走，真装前须用昇腾《软件兼容关系表》复核；这里给的是
#   "与 torch 主版本同号 + postN" 的命名形态，云端可用环境变量覆盖，不是拍死值。
TILELANG_VERSION="${TILELANG_VERSION:-v0.1.15}"       # TileLang 仓 tag（P2 契约下限 0.1.15）
TILEKERNELS_REF="${TILEKERNELS_REF:-main}"            # 只作参照件，绝不进 import 链
CANN_BASE_URL="${CANN_BASE_URL:-https://ascend-repo.obs.cn-east-2.myhuaweicloud.com}"
DEVICE="${DEVICE:-910b}"                               # 910b | 950（决定 arch 与 kernels 包）
NPU_ARCH_910B="${NPU_ARCH_910B:-dav-2201}"            # bisheng --npu-arch 取值，按 CANN 支持表填
NPU_ARCH_950="${NPU_ARCH_950:-dav-3510}"              # tilelang 侧默认值即此
INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/Ascend}"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/ascend.env}"
WITH_TILEKERNELS="${WITH_TILEKERNELS:-0}"

DRY_RUN=0
ASSUME_YES=0
PLAN_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    --plan) PLAN_ONLY=1; shift ;;
    --device) DEVICE="${2:?--device 需要参数 (910b|950)}"; shift 2 ;;
    --prefix) INSTALL_PREFIX="${2:?--prefix 需要参数}"; shift 2 ;;
    --env-file) ENV_FILE="${2:?--env-file 需要参数}"; shift 2 ;;
    --with-tilekernels) WITH_TILEKERNELS=1; shift ;;
    -h|--help) sed -n '1,33p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数: $1（支持 --dry-run/--yes/--plan/--device/--prefix/--env-file/--with-tilekernels）" >&2; exit 1 ;;
  esac
done

case "$DEVICE" in
  910b) ASCEND_NPU_ARCH="$NPU_ARCH_910B" ;;
  950)  ASCEND_NPU_ARCH="$NPU_ARCH_950" ;;
  *) echo "不支持的 device: ${DEVICE}（请给 910b 或 950）" >&2; exit 1 ;;
esac

log()  { printf '[setup] %s\n' "$*"; }
warn() { printf '[setup][WARN] %s\n' "$*" >&2; }

# print_plan：把 pin 表原样打到 stdout，供 --plan 与 --dry-run 共用。
# 白话:这一步不装任何东西，只是把"准备装哪些包、每个包锁在哪个版本号"摊开给人
# 看一眼，好在真金白银开机器之前先确认清单对不对，看错了随时改环境变量重跑。
print_plan() {
  cat <<EOF
—— 版本 pin 清单（device=$DEVICE, npu_arch=$ASCEND_NPU_ARCH, prefix=${INSTALL_PREFIX}）——
  Python            >= $PYTHON_MIN
  CANN toolkit      = $CANN_VERSION
  PyTorch           = $TORCH_VERSION
  torch_npu         = $TORCH_NPU_VERSION   （须与 torch/CANN 三元组对齐）
  TileLang          = $TILELANG_VERSION   （源码构建，强制 -DUSE_ASCEND=ON）
  TileKernels       = $TILEKERNELS_REF    （仅参照；with_tilekernels=${WITH_TILEKERNELS}）
  环境变量落盘      -> $ENV_FILE
  安装序: 体检 -> set_env -> CANN -> env 落盘 -> torch -> torch_npu -> TileLang -> 参照件 -> 自检
EOF
}

# run：所有真实副作用命令的唯一出口。dry-run 下只打印不执行，保证本机可空跑。
# 白话:这是个开关，拨到空跑档时它只把"我本来要执行这条命令"念出来，绝不真去动
# 系统盘和网络，这样在没有昇腾卡的本机上也能完整走一遍流程验证脚本没写错。
run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  DRY-RUN 将执行: %s\n' "$*"
  else
    log "执行: $*"
    eval "$@" || { warn "步骤失败: $*"; exit 2; }
  fi
}

preflight() {
  log "步骤 1/9：平台体检"
  local os
  os="$(uname -s)"
  printf '  读取 uname -s/-m = %s/%s\n' "$os" "$HOST_ARCH"
  if [[ "$DRY_RUN" != "1" ]]; then
    [[ "$os" == "Linux" ]] || { warn "当前系统 $os 非 Linux，CANN/torch_npu 装不上；本机请只用 --dry-run"; exit 2; }
    case "$HOST_ARCH" in
      x86_64|aarch64) : ;;
      *) warn "不支持的架构 $HOST_ARCH"; exit 2 ;;
    esac
    run "python3 -c 'import sys; assert sys.version_info >= (3,10), sys.version'"
  else
    printf '  （真装时此处会拦截非 Linux / 非 x86_64|aarch64 / Python < %s）\n' "$PYTHON_MIN"
    printf '  需自查磁盘可用 >= 30GiB（CANN toolkit + kernels + 源码构建产物）\n'
  fi
}

reuse_setenv() {
  log "步骤 2/9：复用已有 CANN set_env.sh（若存在）"
  local f="$INSTALL_PREFIX/ascend-toolkit/set_env.sh"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  DRY-RUN 将检查并 source %s（存在则复用，避免重复装 CANN）\n' "$f"
    return
  fi
  if [[ -f "$f" ]]; then log "source $f"; . "$f"; else log "无 set_env.sh，继续全新安装"; fi
}

install_cann() {
  log "步骤 3/9：CANN toolkit + kernels ($CANN_VERSION, arch=$HOST_ARCH)"
  local pkg="$INSTALL_PREFIX/pkg" suffix="x86_64"
  [[ "$HOST_ARCH" == "aarch64" ]] && suffix="aarch64"
  run "mkdir -p $pkg"
  # .run 包名随 CANN 小版本/架构变化，这里按官方命名模板拼 URL，真装前需人工核对一次。
  run "curl -fL -o $pkg/toolkit.run '$CANN_BASE_URL/$CANN_VERSION/Ascend-cann-toolkit_${CANN_VERSION}_linux-$suffix.run'"
  run "chmod +x $pkg/toolkit.run"
  run "$pkg/toolkit.run --install --install-path=$INSTALL_PREFIX"
  run "curl -fL -o $pkg/kernels.run '$CANN_BASE_URL/$CANN_VERSION/Ascend-cann-kernels-$DEVICE.run'"
  run "chmod +x $pkg/kernels.run && $pkg/kernels.run --install --install-path=$INSTALL_PREFIX"
}

write_env_file() {
  log "步骤 4/9：环境变量落盘 $ENV_FILE"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  DRY-RUN 将写入 %s，内容五行：\n' "$ENV_FILE"
    printf '    export ASCEND_HOME_PATH=%s/ascend-toolkit/latest\n' "$INSTALL_PREFIX"
    printf '    export ASCEND_OPP_PATH=$ASCEND_HOME_PATH/opp\n'
    printf '    export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH\n'
    printf '    export PATH=$ASCEND_HOME_PATH/bin:$PATH\n'
    printf '    export ASCEND_NPU_ARCH=%s   # bisheng --npu-arch（tilelang 默认 dav-3510）\n' "$ASCEND_NPU_ARCH"
    return
  fi
  cat > "$ENV_FILE" <<EOF
# 由 ascend/env_setup.sh 生成；source 之即可用。改版本请改脚本 pin 表重跑，勿手改。
export ASCEND_HOME_PATH="$INSTALL_PREFIX/ascend-toolkit/latest"
export ASCEND_OPP_PATH="\$ASCEND_HOME_PATH/opp"
export LD_LIBRARY_PATH="\$ASCEND_HOME_PATH/lib64:\${LD_LIBRARY_PATH:-}"
export PATH="\$ASCEND_HOME_PATH/bin:\${PATH:-}"
export ASCEND_NPU_ARCH="$ASCEND_NPU_ARCH"
EOF
  log "已写入 $ENV_FILE"
}

install_torch() {
  log "步骤 5/9：PyTorch ${TORCH_VERSION}（先于 torch_npu）"
  run "pip install --no-input 'torch==$TORCH_VERSION'"
}

install_torch_npu() {
  log "步骤 6/9：torch_npu ${TORCH_NPU_VERSION}（必须在 torch 之后）"
  # 若当日 CANN 版本没有对应 wheel，这里会失败退出 2 —— 宁停勿猜，改源码构建需人工确认。
  run "pip install --no-input 'torch_npu==$TORCH_NPU_VERSION'"
  run "python -c 'import torch, torch_npu; print(\"torch\", torch.__version__, \"npu_available\", torch.npu.is_available())'"
}

install_tilelang() {
  log "步骤 7/9：TileLang $TILELANG_VERSION —— 源码构建，开关写死 USE_ASCEND=ON"
  local src="$INSTALL_PREFIX/src/tilelang"
  run "mkdir -p $INSTALL_PREFIX/src"
  run "git clone --depth 1 --branch $TILELANG_VERSION https://github.com/tile-ai/tilelang.git $src"
  run "cmake -S $src -B $src/build -DUSE_ASCEND=ON -DCMAKE_BUILD_TYPE=Release"
  run "cmake --build $src/build -j\$(nproc)"
  run "pip install --no-input -e $src --no-build-isolation"
  # 装完立刻自证方言后端已注册；没注册就停，别把问题推到写内核那天才发现。
  run "python -c 'import tilelang, tilelang.ascend.language as T; print(\"tilelang\", tilelang.__version__)'"
  printf '  提示：若后续 selfcheck 仍报 ascend_copy is not registered，说明装的是轮子而非本步源码构建\n'
}

fetch_tilekernels_ref() {
  log "步骤 8/9：TileKernels 参照件（WITH_TILEKERNELS=${WITH_TILEKERNELS}）"
  if [[ "$WITH_TILEKERNELS" != "1" ]]; then
    printf '  跳过：本域内核不 import 外部件；参照件仅供人工对读（需要时加 --with-tilekernels）\n'
    return
  fi
  run "git clone --depth 1 --branch $TILEKERNELS_REF https://github.com/tile-ai/TileKernels.git $INSTALL_PREFIX/src/tile_kernels"
}

run_selfcheck() {
  log "步骤 9/9：环境自检（C1 探针）"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  DRY-RUN 将执行: python %s/selfcheck.py\n' "$SCRIPT_DIR"
  else
    run "python '$SCRIPT_DIR/selfcheck.py'"
  fi
}

if [[ "$PLAN_ONLY" == "1" ]]; then
  print_plan
  exit 0
fi

print_plan
if [[ "$DRY_RUN" == "1" ]]; then
  log "空跑模式（--dry-run）：以下所有步骤只打印，不产生任何副作用"
elif [[ "$ASSUME_YES" != "1" ]]; then
  warn "未加 --yes，拒绝真装（云端真装请：bash ascend/env_setup.sh --yes --device 910b）"
  exit 1
fi

preflight
reuse_setenv
install_cann
write_env_file
install_torch
install_torch_npu
install_tilelang
fetch_tilekernels_ref
run_selfcheck
log "流程结束（dry-run 表示清单已核对；真装请去掉 --dry-run 并加 --yes）"
