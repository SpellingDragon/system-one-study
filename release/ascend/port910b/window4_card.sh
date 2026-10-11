#!/bin/bash
# window4_card.sh — 第四窗：GDN 反向三缺陷钉死 + 双因子提案判决（五段，各独立进程；预计 15–20 分钟）
# V1 卡上 gradcheck 全 8 件（验 F4 的 conv 反向修复）
# V2 E3 逐路分诊（把 gdn_delta 的 nan 钉到"前向内核 vs 反向内核 × 哪个输入"）
# V3 delta_bwd 逐项判（是否只坏 dbeta）  V4 D3 ub/pure 两档差分 + 发射顺序
# V5 S3 双因子提案 v2 判决（打补丁前后各一趟 nextshot，STALE-DIFF 是否归零）
set -u
T=/tmp/sys1
SRC=https://github.com/SpellingDragon/system-one-study.git

[ -d $T ] || git clone -q --depth 1 $SRC $T || { echo "W4-ABORT clone-fail"; exit 1; }
cd $T/release && git -C $T pull -q 2>/dev/null
source /usr/local/Ascend/cann-8.5.2/set_env.sh 2>/dev/null
export ASCEND_NPU_ARCH=dav-2201
export BISHENG_HOME=$(dirname $(dirname $(command -v bisheng)))
TLPKG=$(python3 -c "import tilelang,os;print(os.path.dirname(tilelang.__file__))")
TLHDR=$TLPKG/src/tl_templates/ascend/port910b_compat.h
echo "W4-ENV TLPKG=$TLPKG hdr_md5=$(md5sum $TLHDR | cut -c1-8)"

echo "=== V1 卡上 gradcheck（F4 conv 反向修复的判决）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 900 python3 ascend/gradcheck_card.py --target ascend > /tmp/w4_gradcheck.log 2>&1; echo "W4-V1-EXIT=$?"
grep -E "^GRADCHECK-" /tmp/w4_gradcheck.log

echo "=== V2 E3 逐路分诊（gdn_delta 坏路 + gdn_conv 现状）==="
timeout 900 python3 ascend/kernels/reconcile/E3_probe_cardtriage.py --target ascend > /tmp/w4_triage.log 2>&1; echo "W4-V2-EXIT=$?"
grep -E "VERDICT|raise-frame|^E3-" /tmp/w4_triage.log | head -30

echo "=== V3 delta_bwd 逐项判（是否只坏 dbeta）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 900 python3 ascend/port910b/run_delta_card.py --target ascend --bwd > /tmp/w4_deltabwd.log 2>&1; echo "W4-V3-EXIT=$?"
grep -E "^ITEM|^NUM-" /tmp/w4_deltabwd.log

echo "=== V4 D3 ub/pure 两档差分（dbeta 责任面）==="
for VAR in ub pure; do
  export TILELANG_CACHE_DIR=$(mktemp -d)
  timeout 600 python3 ascend/kernels/reconcile/D3_probe_dbeta_order.py --dev npu --variant $VAR --target ascend > /tmp/w4_d3_$VAR.log 2>&1
  echo "W4-V4-$VAR-EXIT=$?"; grep -E "^D3-ORDER|^gdnb keys|^DBETA-FP|^ +[a-z-]+ +dbeta-fp=|^ITEM |^NUM-" /tmp/w4_d3_$VAR.log | head -16
done

echo "=== V5 双因子提案 v2 判决（补丁前后各一趟）==="
export TILELANG_CACHE_DIR=$(mktemp -d)
timeout 900 python3 ascend/kernels/reconcile/S3_probe_nextshot.py --dev npu --prefills none,zero,perm > /tmp/w4_v2_before.log 2>&1
echo "W4-V5-BASE-EXIT=$?"; grep -E "STALE-DIFF|U2-READOUT|unexpected" /tmp/w4_v2_before.log | head -8
cp $TLHDR $TLHDR.w4bak
DIFF=ascend/port910b/attempts/S3_compat_proposal_v2.diff
if (cd $TLPKG && patch -p2 --dry-run < ../release/$DIFF >/dev/null 2>&1 && cd $TLPKG && patch -p2 < ../release/$DIFF >/dev/null 2>&1); then
  echo "W4-V5-PATCHED mode=p2 md5=$(md5sum $TLHDR | cut -c1-8)"
elif (cd $TLPKG && patch -p1 --dry-run < $T/release/$DIFF >/dev/null 2>&1 && cd $TLPKG && patch -p1 < $T/release/$DIFF >/dev/null 2>&1); then
  echo "W4-V5-PATCHED mode=p1 md5=$(md5sum $TLHDR | cut -c1-8)"
else
  echo "W4-V5-SKIP reason=patch-unapplicable（如实不判 v2）"
fi
if [ -f $TLHDR ] && [ "$(md5sum $TLHDR | cut -c1-8)" != "$(md5sum $TLHDR.w4bak | cut -c1-8)" ]; then
  export TILELANG_CACHE_DIR=$(mktemp -d)
  timeout 900 python3 ascend/kernels/reconcile/S3_probe_nextshot.py --dev npu --prefills none,zero,perm > /tmp/w4_v2_after.log 2>&1
  echo "W4-V5-AFTER-EXIT=$?"; grep -E "STALE-DIFF|U2-READOUT|unexpected" /tmp/w4_v2_after.log | head -8
  cp $TLHDR.w4bak $TLHDR; echo "W4-V5-RESTORED md5=$(md5sum $TLHDR | cut -c1-8)"
fi

echo "=== 现场保全 ==="
tar czf /tmp/w4_artifacts.tgz -C /tmp w4_gradcheck.log w4_triage.log w4_deltabwd.log w4_d3_ub.log w4_d3_pure.log w4_v2_before.log w4_v2_after.log 2>/dev/null
echo "W4-TGZ size=$(wc -c </tmp/w4_artifacts.tgz)B"; echo "W4-DONE"
