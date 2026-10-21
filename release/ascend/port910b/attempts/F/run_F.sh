#!/bin/bash
# attempts/F/run_F.sh — 宿主侧一键：同步 F 波件 → cann910b-f → 跑 compile/golden 判决
#
# 用法:
#   bash attempts/F/run_F.sh --matrix            # 全部判决（RESULT.md 的凭据源）
#   bash attempts/F/run_F.sh --cell fwd --variant ub         # 单件判决（参数透传）
#   bash attempts/F/run_F.sh --cell bwd --variant pure
#   bash attempts/F/run_F.sh --golden [--npz /tmp/f_gdn_pairs.npz]   # CPU 分步对拍（纯 CPU）
#   bash attempts/F/run_F.sh --probe             # GAP-F logf 数值定标 + 往返（纯 CPU）
#   bash attempts/F/run_F.sh --diff              # GAP-F 编译差分（负形必 FAIL / 正形必 PASS）
#   bash attempts/F/run_F.sh --restore           # 容器 pip 副本还原为未打 GAP-F 的快照
#
# 判决纪律（C 波踩过）：tilelang 按 **kernel 源码哈希** 命中编译缓存，改 compat **头**
#   不会让缓存失效 → 不换 cache 就是陈旧 PASS。故每条判决都 TILELANG_CACHE_DIR=$(mktemp -d)。
# 边界纪律：**不做** `cp /tilelang/...`（编排者正在改 trunk §12，本容器快照=隔离基线）；
#   GAP-F 只**追加**到容器 pip 副本文件尾（patch_compat_F.py 幂等，可 --restore）。
set -e
F=${TL_F_CONTAINER:-cann910b-f}
D=$(cd "$(dirname "$0")" && pwd)
ENV='source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1;
export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201;
export TILELANG_CACHE_DIR=$(mktemp -d /tmp/tlc_cache.XXXXXX);'

FILES=(compat_patch_F.h patch_compat_F.py f_probe_logf.py
       f_gdn_delta_fwd_910b.py f_gdn_delta_bwd_910b.py f_gdn_golden.py)
# 推送用 **docker cp**（不是 `cat >`）：本容器 exec 侧 root 对 uid=501 属主的 644 文件
# 没有覆写权（DAC_OVERRIDE 被剥），`cat >` 会 "Permission denied" 而静默留下**旧版本**
# —— 本波真的踩过：矩阵跑完才发现容器里是修复前的 bwd。故 cp 之后逐个 md5 强校验。
for f in "${FILES[@]}"; do
  [ -f "$D/$f" ] || continue
  docker cp "$D/$f" "$F:/tmp/$f"
  h=$(md5 -q "$D/$f" 2>/dev/null || md5sum "$D/$f" | cut -d" " -f1)
  c=$(docker exec "$F" md5sum "/tmp/$f" | cut -d" " -f1)
  if [ "$h" = "$c" ]; then echo "PUSH-VERIFY SAME $f"
  else echo "PUSH-VERIFY MISMATCH $f host=$h cont=$c"; exit 1; fi
done

run_cell() {  # $1=fwd|bwd  其余透传给件脚本
  local k="$1"; shift
  docker exec "$F" bash -c "$ENV cd /tmp && timeout 300 python3 /tmp/f_gdn_delta_${k}_910b.py $*"
}

do_diff() {
  echo "---------- GAP-F 差分·负形（还原快照后，未注入补丁，预期 FAIL）----------"
  docker exec "$F" bash -c "cd /tmp && python3 /tmp/patch_compat_F.py --restore" || true
  docker exec "$F" bash -c "$ENV cd /tmp && timeout 300 python3 /tmp/f_probe_logf.py --mode compile" \
    && echo "DIFF-ANOMALY: 未打补丁也编过了（缺口不成立，需复查）" \
    || echo "DIFF-NEGCONFIRMED: 未注入 GAP-F 时 logf 面确实不可编 ⇒ 缺口成立"
  echo "---------- GAP-F 差分·正形（注入 compat_patch_F.h 后，预期 PASS）----------"
  docker exec "$F" bash -c "cd /tmp && python3 /tmp/patch_compat_F.py"
  docker exec "$F" bash -c "$ENV cd /tmp && timeout 300 python3 /tmp/f_probe_logf.py --mode compile"
}

case "${1:---matrix}" in
  --cell) shift; run_cell "$@" ;;
  --golden) shift; docker exec "$F" bash -c "cd /tmp && timeout 900 python3 /tmp/f_gdn_golden.py $*" ;;
  --probe) shift; docker exec "$F" bash -c "cd /tmp && python3 /tmp/f_probe_logf.py --mode accuracy && python3 /tmp/f_probe_logf.py --mode gdn-path" ;;
  --restore) shift; docker exec "$F" bash -c "cd /tmp && python3 /tmp/patch_compat_F.py --restore" ;;
  --diff) shift; do_diff ;;
  --matrix) shift
    echo "########## [1/8] FWD ub（验证形状 H8 T32 DK16 DV16 cores8）##########"
    run_cell fwd "--variant ub --dump /tmp/f_fwd_ub.asc"
    echo "########## [2/8] FWD pure（零 UB，状态就地 GM）##########"
    run_cell fwd "--variant pure --dump /tmp/f_fwd_pure.asc"
    echo "########## [3/8] BWD ub（伴随态 (DV,DK) 常驻 UB）##########"
    run_cell bwd "--variant ub --dump /tmp/f_bwd_ub.asc"
    echo "########## [4/8] BWD pure（零 UB，GM 读改写累加）##########"
    run_cell bwd "--variant pure --dump /tmp/f_bwd_pure.asc"
    echo "########## [5/8] BWD ub --dstate gm（末状态梯度非零入口）##########"
    run_cell bwd "--variant ub --dstate gm --dump /tmp/f_bwd_ub_gmstate.asc"
    echo "########## [6/8] 生产形状外推 H16 DK64 DV64 T128 cores16 ##########"
    run_cell fwd "--variant ub --H 16 --T 128 --DK 64 --DV 64 --cores 16 --dump /tmp/f_fwd_prod.asc"
    run_cell bwd "--variant ub --H 16 --T 128 --DK 64 --DV 64 --cores 16 --dump /tmp/f_bwd_prod.asc"
    echo "########## [7/8] GAP-F logf 编译差分（正/负形）##########"
    do_diff
    echo "########## GAP-F 数值定标（numpy 逐行复刻 vs np.log）##########"
    docker exec "$F" bash -c "cd /tmp && python3 /tmp/f_probe_logf.py --mode accuracy"
    docker exec "$F" bash -c "cd /tmp && python3 /tmp/f_probe_logf.py --mode gdn-path"
    echo "########## [8/8] 回归：注入 GAP-F 后 FWD/BWD 仍须 COMPILE-PASS ##########"
    run_cell fwd "--variant ub"
    run_cell bwd "--variant ub" ;;
esac
