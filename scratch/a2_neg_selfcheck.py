"""A2 负例自证（非交付件，供 verify_dw.py 的假绿守卫复跑）：
  NEG-1  ascend_env.backend_available 恒 False  ⇒ 编译根本没发生 ⇒ 判决必须 FAIL
  NEG-2  tilelang.compile 被换成"零生成码"假件  ⇒ 桩的自证账目为空 ⇒ 判决必须 FAIL
跑法（容器内，release 挂 /work）：
  source /usr/local/Ascend/cann-8.5.0/set_env.sh
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) python3 /tmp/a2_neg_selfcheck.py
"""
import os
import sys

sys.path.insert(0, "/work")
sys.path.insert(0, os.path.join("/work", "ascend", "kernels", "reconcile"))
import tilelang                      # noqa: E402
import verify_dw as V                # noqa: E402
from ascend.kernels import ascend_env  # noqa: E402

NEG = sys.argv[1] if len(sys.argv) > 1 else "1"

if NEG == "1":
    orig_gate = V._gate

    def sabotage_gate():
        orig_gate()
        ascend_env.backend_available = lambda *a, **k: False   # 后端不可用 ⇒ get_compiled 直返 None
    V._gate = sabotage_gate
    print("[NEG-1] 打桩：ascend_env.backend_available ≡ False（编译不会发生）")
else:
    class _ZeroSrc:
        def get_kernel_source(self):
            return ""                                          # "编了但什么都没产出"的假件
        def __call__(self, *a, **k):
            return None

    tilelang.compile = lambda *a, **k: _ZeroSrc()               # 先于 _stub_compile capture ⇒ 桩拿到假 real
    print("[NEG-2] 打桩：tilelang.compile ≡ 零生成码假件（若有键就报 PASS 的桩即假绿）")

rc = V._interface_dynamic_m()
print(f"[NEG-{NEG}] interface rc={rc}（0=PASS/1=FAIL）")
sys.exit(rc)
