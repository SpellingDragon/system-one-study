import sys, traceback
sys.path.insert(0, "$REPO_ROOT/release")
import tilelang
from ascend.kernels import gemm_asc

try:
    fn = gemm_asc.gemm_cpu_impl(n=16, k=32, bm=16, bn=16, bk=32, act_mode=1)
    print("traced ok:", type(fn))
except Exception:
    traceback.print_exc()
    sys.exit(1)

try:
    ker = tilelang.compile(fn, out_idx=[], **{"target": "c", "target_host": "c", "execution_backend": "cython"})
    print("compiled ok:", type(ker))
except Exception:
    traceback.print_exc()
