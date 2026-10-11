"""I3_probe_gdn_bwd_shape.py — 只读探针：`gdn_asc.backward`（昇腾伴随件路）对两种 `g` 记账形
式的**回报形状**是否与输入同形（这是 P2 真形档跑一步时暴露的形状缺陷的隔离件）。

【背景（为什么要这枚探针）】
`ascend/train_step_real.py`（真形档判决件）在容器 `--stub-run` 下跑到 `loss0.backward()` 时炸：
```
RuntimeError: Function _GdnFnBackward returned an invalid gradient at index 3 -
got [16, 512, 128] but expected shape compatible with [16, 512]
```
index 3 = `g`（`_GdnFn.forward(ctx, q, k, v, g, beta, …)` 的第 4 个输入）。真形档 `g` 是
**每 token 一个标量门**、形状 `(heads, seq)=(16,512)`；而回报的 `dg` 是 `(16,512,128)`。
d=32 最小冒烟 / dry 档**看不到**这枚缺陷（见下方"为什么 dry 不触发"），所以它是一枚
**只有真形档才现形**的形状缺陷。

【为什么 dry 不触发（探针要顺手证的口径）】
`_GdnFn` 是 `torch.autograd.Function`：只有**当至少一个输入 requires_grad** 时引擎才会调它的
`backward`，也才会对回报做形状校验。dry/冒烟的隐藏态在**第一个全注意力层之前**的全是线性
注意力层，那些层 `q/k/v/g/beta` 都来自**冻结权重** ⇒ 全不 requires_grad ⇒ 这些层的
`_GdnFn.backward` 根本不被调用，形状缺陷被"没有可训参数"这层壳挡住。真形档里 LoRA 让
`h` 在 idx3 之后带梯度，于是 idx4 起的线性注意力层**输入带梯度** ⇒ `_GdnFn.backward` 首次
被调用 ⇒ 缺陷现形。

【本探针做什么】不走模型，直接构造 `q/k/v/g/beta`（都 requires_grad），调
`gdn_asc.backward(q, k, v, g, beta, dout, target="ascend")`，把五个回报梯度的形状与"与输入同形"
逐项对照，并对 `g` 的两种记账形式各判一次：① `(heads, seq, dk)` 每通道门（`_gate3` 的原生形）；
② `(heads, seq)` 每 token 标量门（`modeling_qwen3_5.py:616-618` 的口径，也是真形档的交法）。
只 import、只打印、不改任何件。

用法（容器，需 device 门打开让 ascend 路可选中）：
    docker exec <ctr> bash -c 'cd <release>; source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
      export BISHENG_HOME=…/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \
      TILELANG_CACHE_DIR=$(mktemp -d) python3 <本文件>'
host cpu 轨（`--target cpu` 落 torch 尺子，用作对照）：
    python3 <本文件> cpu
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))       # .../release
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from ascend.kernels import ascend_env, gdn_asc  # noqa: E402

#: 两档形状：(名字, heads, seq, dk, dv)。dk=key 头维、dv=value 头维。
#: - dry 档取自 `train_step_real.Topo.dry_scaled()`：gdn 头数 16 不动、头维缩到 16；
#: - real 档取自 `config.json:46-49`（linear_key_head_dim=128 / linear_value_head_dim=128）
#:   与 `:47-48`（linear_num_key_heads=linear_num_value_heads=16）。
SHAPES = (
    ("dry",   16, 16,  16,  16),
    ("real",  16, 512, 128, 128),
)


def _gate_env(target: str) -> None:
    """非 cpu 轨：装 A3 同款三量桩 + 发射桩，让 ascend 路可选中且不真发射（容器无 npu）。"""
    if target == "cpu":
        return
    import ascend.train_step_real as TSR
    TSR.install_device_gate()
    TSR.install_compile_only(TSR.Ledger(), "compile")


def probe_facade(target: str, name: str, heads: int, seq: int, dk: int, dv: int, g_form: str) -> bool:
    """走**门面 + autograd**（模型真实路径）：`ops.gdn_delta(...)` 后 `.backward()`，看引擎是否
    对回报的 `dg` 做形状校验。这是"真形档 vs dry 档差在哪"的决定性对照。"""
    from ascend import autograd_asc as ops
    q, k, v, g, beta, dout = _make(name, heads, seq, dk, dv, g_form)
    try:
        out = ops.gdn_delta(q, k, v, g, beta, out_dtype=torch.float32, target=target)
        out.backward(dout)
    except Exception as exc:  # noqa: BLE001  这正是要取证的那条腿
        msg = f"{type(exc).__name__}: {exc}"
        print(f"I3-GDNPROBE-FACADE {name} gform={g_form} target="
              f"{ascend_env.normalize_target(target)} AUTOGRAD-RAISE {msg[:160]}")
        return False
    gs = tuple(None if t.grad is None else tuple(t.grad.shape) for t in (q, k, v, g, beta))
    print(f"I3-GDNPROBE-FACADE {name} gform={g_form} target="
          f"{ascend_env.normalize_target(target)} OK g_in={tuple(g.shape)} grad_shapes={gs}")
    return True


def _make(name: str, heads: int, seq: int, dk: int, dv: int, g_form: str, device="cpu"):
    gen = torch.Generator(device="cpu").manual_seed(20261010)
    q = torch.randn(heads, seq, dk, generator=gen).to(device).requires_grad_(True)
    k = torch.randn(heads, seq, dk, generator=gen).to(device).requires_grad_(True)
    v = torch.randn(heads, seq, dv, generator=gen).to(device).requires_grad_(True)
    if g_form == "3d":
        g = torch.randn(heads, seq, dk, generator=gen).to(device).requires_grad_(True)
    else:
        g = torch.randn(heads, seq, generator=gen).to(device).requires_grad_(True)
    beta = torch.rand(heads, seq, generator=gen).to(device).requires_grad_(True)
    dout = torch.randn(heads, seq, dv, generator=gen).to(device)
    return q, k, v, g, beta, dout


def probe_one(target: str, name: str, heads: int, seq: int, dk: int, dv: int, g_form: str) -> bool:
    """跑一次 `gdn_asc.forward` + `gdn_asc.backward`，对五梯逐项验"与输入同形"。"""
    q, k, v, g, beta, dout = _make(name, heads, seq, dk, dv, g_form)
    # 前向只为把 kernelized 路的 work buffer/规格走通；数值在桩档下无意义（不判）
    try:
        _ = gdn_asc.forward(q, k, v, g, beta, out_dtype=torch.float32, target=target)
    except Exception as exc:  # noqa: BLE001
        print(f"I3-GDNPROBE {name} gform={g_form} FORWARD-RAISE {type(exc).__name__}: {exc}")
        return False
    try:
        grads = gdn_asc.backward(q, k, v, g, beta, dout, target=target)
    except Exception as exc:  # noqa: BLE001
        print(f"I3-GDNPROBE {name} gform={g_form} BACKWARD-RAISE {type(exc).__name__}: {exc}")
        return False
    ins = (q, k, v, g, beta)
    labels = ("dq", "dk", "dv", "dg", "dbeta")
    bad = []
    got = []
    for lab, gi, gshape in zip(labels, grads, (tuple(t.shape) for t in ins)):
        got.append(f"{lab}{tuple(gi.shape)}")
        if tuple(gi.shape) != gshape:
            bad.append(f"{lab}: got{tuple(gi.shape)} want{gshape}")
    key = ascend_env.normalize_target(target)
    blk = [b for b in ascend_env.blockers() if b.startswith("gdn_backward")]
    verdict = "SAME-SHAPE" if not bad else "SHAPE-MISMATCH"
    print(f"I3-GDNPROBE {name} gform={g_form} target={key} {verdict} "
          f"g_in={tuple(g.shape)} {' '.join(got)}"
          + (f" why={';'.join(bad)}" if bad else "")
          + (f" blocker={blk}" if blk else ""))
    return not bad


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    _gate_env(target)
    n_ok = 0
    n_all = 0
    for name, heads, seq, dk, dv in SHAPES:
        for g_form in ("3d", "2d"):
            n_all += 1
            if probe_one(target, name, heads, seq, dk, dv, g_form):
                n_ok += 1
    print(f"I3-GDNPROBE-SUMMARY target={ascend_env.normalize_target(target)} "
          f"same_shape={n_ok}/{n_all} "
          f"(gform=3d 每通道门；gform=2d 每 token 标量门 —— 真形档与 modeling:616-618 的交法)")
    print("--- 门面 + autograd 路径（模型真实路径；看引擎何时对 dg 做形状校验）---")
    f_ok = f_all = 0
    for name, heads, seq, dk, dv in SHAPES:
        f_all += 1
        if probe_facade(target, name, heads, seq, dk, dv, "2d"):
            f_ok += 1
    print(f"I3-GDNPROBE-FACADE-SUMMARY target={ascend_env.normalize_target(target)} "
          f"autograd_ok={f_ok}/{f_all} (均为 gform=2d)")
    return 0 if (n_ok == n_all and f_ok == f_all) else 1


if __name__ == "__main__":
    sys.exit(main())
