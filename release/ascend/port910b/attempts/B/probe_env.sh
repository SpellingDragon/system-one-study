#!/bin/bash
# probe pip tree state in cann910b-b
set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null 2>&1
PIP=$(python3 -c "import tilelang,os;print(os.path.join(os.path.dirname(tilelang.__file__),'src','tl_templates','ascend'))")
echo "PIP_TPL=$PIP"
ls "$PIP" | tr '\n' ' '
echo
echo "-- compat present:"; ls -l "$PIP/port910b_compat.h" 2>&1 | tail -1
echo "-- NATIVE_TYPES in bisheng.py:"; grep -c "TL_PORT910B_NATIVE_TYPES" $(python3 -c "import tilelang,os;print(os.path.join(os.path.dirname(tilelang.__file__),'contrib','bisheng.py'))")
echo "-- compat section10 present:"; grep -c "asc_copy_gm2ub_align" "$PIP/port910b_compat.h" 2>/dev/null
echo "-- arch:"; echo "${ASCEND_NPU_ARCH:-unset}"
echo "-- ccec:"; ls /usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler/bin/bisheng 2>&1 | tail -1
