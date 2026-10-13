import sys
sys.path.insert(0, "/tmp")
import torch as _torch

_orig_device = _torch.device


def _device(s=None, *a, **k):
    if isinstance(s, str) and s.startswith("npu"):
        return _orig_device("cpu")
    return _orig_device(s, *a, **k)


_torch.device = _device
if not hasattr(_torch, "npu"):

    class _NpuShim:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def set_device(*a, **k):
            pass

        @staticmethod
        def get_device_name(*a, **k):
            return "dump-shim"

        @staticmethod
        def current_stream():
            class _S:
                npu_stream = 0

            return _S()

        @staticmethod
        def synchronize():
            pass

    _torch.npu = _NpuShim

sys.argv = ["w9_probe_v7_loop.py"]
import importlib.util as _iu
import types

sys.modules["torch_npu"] = types.ModuleType("torch_npu")  # container stub

_spec = _iu.spec_from_file_location("probe", "/tmp/w9_probe_v7_loop.py")
_m = _iu.module_from_spec(_spec)
# prevent the driver main() from running: only import the kernel factory
_m.__name__ = "probe"
_spec.loader.exec_module(_m)

for lvl in (12, 18, 23):
    k = _m.gb5(1, 1, 1, 8, 8, lvl)
    src = k.get_kernel_source()
    open(f"/tmp/w9_gen_L{lvl}.cpp", "w").write(src)
    print(f"DUMP L{lvl} src={len(src)}")
print("DUMPS-DONE")
