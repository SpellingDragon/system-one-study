"""ascend **真形档**：Qwen3.5-0.8B 生产拓扑在**混合栈**上跑一步 fwd+bwd+opt（p2-13 代理 I3）。

【为什么需要本件】`ascend/train_step.py` 是 **d=32 最小冒烟**（自陈"只验接线、不验 rel"，
`train_step.py:9,402` 把 `seq,d,heads,dim` 钉死成 `16,32,2,16`），它证明的是"八个门面的接线与
反向公式对"，**不证明 0.8B 形状能不能落到内核上**。而 `R6/R7` 熔断判定要的是"生产拓扑每步各算子被
调多少次、每次多贵、峰值多少显存"——这是本件唯一要交的东西。C3 复核（`reconcile/C3_closure_audit.md`
§4c/§4d）已判：全库无 0.8B 训练步产物，且 `train_step.py:42-44` 现不含 `gdn_conv`（GDN 件替换
Conv2D 那条路未接）。故本件与 `train_step.py` **并行存在、互不改动**：既有件继续保持"接线档"职责，
本件新增"真形档"。

【拓扑真源（数字全部引自配置原文，行号即凭据）】
  `bench/ms_models/models/Qwen--Qwen3.5-0.8B/snapshots/master/config.json` 的 `text_config`：
    hidden_size=1024 (:16) · num_hidden_layers=24 (:56) · num_attention_heads=8 (:55)
    num_key_value_heads=2 (:57) · head_dim=256 (:14) · intermediate_size=3584 (:18)
    vocab_size=248320 (:61) · rms_norm_eps=1e-6 (:58) · tie_word_embeddings=true (:59)
    attn_output_gate=true (:10) · full_attention_interval=4 (:13) · layer_types 24 项 (:19-44)
    linear_conv_kernel_dim=4 (:45) · linear_num_key_heads=16 (:47) · linear_key_head_dim=128 (:46)
    linear_num_value_heads=16 (:48) · linear_value_head_dim=128 (:49)
    rope: partial_rotary_factor=0.25 / rope_theta=1e7 (:63-73)
  逐层 op 顺序真源：`transformers/models/qwen3_5/modeling_qwen3_5.py`
    `Qwen3_5GatedDeltaNet` (:503-662) · `Qwen3_5Attention` (:748-820) · `Qwen3_5MLP` (:823-836)
    `Qwen3_5RMSNorm` 零中心 `(1+w)` (:840-854) · `Qwen3_5RMSNormGated` 逐头 + silu 门 (:217-233)
    `Qwen3_5DecoderLayer` 残差/前后置 norm 次序 (:860-913)
  训练面口径真源：`production/configs/gate08b.yaml`（`:10` backbone=qwen3.5-0.8b · `:13` dtype
    bfloat16 · `:21` max_length=8192 · `:32` r=16 · `:33` alpha=32 · `:43` max_tokens=16384 ·
    `:35` gradient_checkpointing=true）与 `production/sft.py`（`:146` LoRA 只挂 q/k/v/o 四投影、
    `:533` 损失=读点位软交叉熵 `option_scores`）。LoRA 目标面与 r/alpha 由 `sft.py:146,140-142` 钉死。

【怎么做】① 按上表把一层拆成 op 序列，可微点**一律走 `ascend.autograd_asc` 门面**（不改门面、
  不改方言件、不改 `train_step.py`）；② **混合栈分流**：门面内部 `plan()` 拒形状时会静默落 eager
  （`gemm_kernel.py:46-51`，C2 已判的盲点），本件在**不改既有件**的前提下，给 9 个 plan 入口套一层
  只读观察器——`plan` 回 None 即打一行 `SWAP <族> <形状> -> native`，绝不静默；③ fp16 主干 / fp32
  归约纪律**沿用 F2**（直接 import `train_step` 的 `install_reduction_guard` / `dtype_trace_line`，
  不改它们）；④ `KERNEL-USED` 三证与 `DTYPE-TRACE` 也沿用既有实现（import 即用）。
  被否方案一：直接把 `train_step.run()` 的尺寸改大——那会动既有产品件的语义与既有判据的基线；
  被否方案二：实例化 HF `Qwen3_5ForConditionalGeneration` 再逐层换后端——0.75B 权重装载要真检查点
  （本机未缓存），且 HF 的 attention 后端不是我们要记的自研件；被否方案三：用 `sys1/model.py` 的
  自训 GPT——那不是生产学生（`gate08b.yaml:10` 学生=Qwen3.5-0.8B）。

【本件刻意不做的（如实登记，不假绿）】
  · **不做 HF→自研栈的坐标置换**（`production/backbone/layout.py::rope_pair_permutation`）：那影响
    rope 的**数值口径**，不影响 op 结构/形状/调用数与显存记账（本件唯一交付物）。数值等价由 p2-01
    的布局对拍（`tests/test_backbone_layout.py`）与 C5 卡窗负责。
  · **不实现 gradient_checkpointing**：`gate08b.yaml:35` 开着，本件记的是"不重算"的峰值 ⇒ 是**上界**；
    重算档的账在 `I3_REALSHAPE_SPEC.md` 里按 A3 方法另算。
  · **不做 `gdn_delta` 的 qk-l2norm**（模型 `:646` `use_qk_l2norm_in_kernel=True`）：门面契约里没有
    这一步，`gdn_delta` 门面的就是"纯递推"，本件忠实于**门面契约**而非 HF 内部实现。
  · 输入数据是随机张量（同 `train_step._make_inputs` 口径）：本件不产评测分，只产记账。

【运行】
  host 自证（秒级，小形状降级但层数/调用次数结构不变）：
      <venv> release/ascend/train_step_real.py --target cpu --dry
  卡窗一枪（真形状 + 真设备，见 reconcile/I3_REALSHAPE_SPEC.md §窗六）：
      <py> release/ascend/train_step_real.py --target ascend --dtype fp16
  容器证可编（无卡、走 A3 同款三量桩 + 真编译 + 发射桩）：
      <py> release/ascend/train_step_real.py --target ascend --no-run
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import warnings
from dataclasses import dataclass, replace

_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ascend import autograd_asc as ops  # noqa: E402
from ascend.kernels import (  # noqa: E402
    ascend_env,
    attn_sw_asc,
    gdn_asc,
    gdn_conv_asc,
    gemm_asc,
    gemm_bwd_dw_asc,
    letter_readout_asc,
    rope_asc,
)

# ---- 复用既有实现（import 即用，一字不改；F2 归约纪律 / H1 设备注册 / 判决行格式都走它们）----
import ascend.train_step as TS  # noqa: E402

npu_requested = TS.npu_requested
ensure_torch_npu = TS.ensure_torch_npu
npu_reg_line = TS.npu_reg_line
install_reduction_guard = TS.install_reduction_guard
uninstall_reduction_guard = TS.uninstall_reduction_guard
reduction_violations = TS.reduction_violations
dtype_trace_line = TS.dtype_trace_line
DTYPE_BY_NAME = TS.DTYPE_BY_NAME
EXIT_DEVICE_UNAVAILABLE = TS.EXIT_DEVICE_UNAVAILABLE
EXIT_KERNEL_UNUSED = TS.EXIT_KERNEL_UNUSED
EXIT_DTYPE_DISCIPLINE = TS.EXIT_DTYPE_DISCIPLINE
ENV_KERNEL_USAGE_STRICT = TS.ENV_KERNEL_USAGE_STRICT

#: 拓扑真源（相对 release/）；不存在时退回下方字面量（每个数带 config.json 行号注释）
CONFIG_REL = "bench/ms_models/models/Qwen--Qwen3.5-0.8B/snapshots/master/config.json"
#: 读点位字母表大小（`sys1/decision/render.py:51` `LETTERS = string.ascii_uppercase` ⇒ 26 候选）
LETTER_COUNT = 26
#: LoRA 挂载的四枚投影（`production/sft.py:146` `LORA_TARGET_LEAVES`）
LORA_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj")


# =============================================================== 拓扑（Topo）
def _layer_types(n: int) -> tuple[str, ...]:
    """按 `full_attention_interval=4`（config.json:13）生成层型序列：每 4 层 1 个全注意力。

    与 config.json:19-44 的显式清单逐项一致（全注意力落在 idx 3/7/11/15/19/23）。
    """
    return tuple("full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n))


@dataclass(frozen=True)
class Topo:
    """一层不多、一个形状不改的 0.8B 学生拓扑（数值全来自配置原文，见模块 docstring 行号表）。"""

    hidden: int          # config.json:16
    layers: int          # config.json:56
    heads: int           # config.json:55
    kv_heads: int        # config.json:57
    head_dim: int        # config.json:14
    inter: int           # config.json:18
    vocab: int           # config.json:61
    rms_eps: float       # config.json:58
    conv_kernel: int     # config.json:45
    gdn_k_heads: int     # config.json:47
    gdn_k_dim: int       # config.json:46
    gdn_v_heads: int     # config.json:48
    gdn_v_dim: int       # config.json:49
    layer_types: tuple[str, ...]   # config.json:19-44
    lora_r: int          # gate08b.yaml:32
    lora_alpha: int      # gate08b.yaml:33
    rope_theta: float = 1e7        # config.json:71
    dry: bool = False
    src: str = ""

    # ---- 派生量（全部是上表的算术，不含新数字）----
    @property
    def rotary_dim(self) -> int:
        """每头真正被旋转的宽度 = head_dim × partial_rotary_factor(config.json:72)=0.25 ⇒ 64。"""
        return int(self.head_dim * 0.25)

    @property
    def q_gate_n(self) -> int:
        """q_proj 出口宽 = heads×head_dim×2（`modeling:760-762`，每头前半查询后半门）。"""
        return self.heads * self.head_dim * 2

    @property
    def kv_n(self) -> int:
        """k_proj/v_proj 出口宽 = kv_heads×head_dim（`modeling:763-768`）⇒ 512。"""
        return self.kv_heads * self.head_dim

    @property
    def o_in(self) -> int:
        """o_proj 入口宽 = heads×head_dim（`modeling:769-771`）⇒ 2048。"""
        return self.heads * self.head_dim

    @property
    def gdn_key_dim(self) -> int:
        """GDN 键宽 = linear_num_key_heads×linear_key_head_dim（`modeling:511`）⇒ 2048。"""
        return self.gdn_k_heads * self.gdn_k_dim

    @property
    def gdn_val_dim(self) -> int:
        """GDN 值宽 = linear_num_value_heads×linear_value_head_dim（`modeling:512`）⇒ 2048。"""
        return self.gdn_v_heads * self.gdn_v_dim

    @property
    def conv_dim(self) -> int:
        """短卷积通道 = key_dim×2 + value_dim（`modeling:520`）⇒ 6144。"""
        return 2 * self.gdn_key_dim + self.gdn_val_dim

    @property
    def n_full(self) -> int:
        return sum(1 for t in self.layer_types if t == "full_attention")

    @property
    def n_gdn(self) -> int:
        return sum(1 for t in self.layer_types if t == "linear_attention")

    @property
    def group(self) -> int:
        """分组注意力复用倍数（`modeling:756` num_attention_heads//num_key_value_heads）⇒ 4。"""
        if self.heads % self.kv_heads:
            raise ValueError(f"heads={self.heads} 不是 kv_heads={self.kv_heads} 的整数倍")
        return self.heads // self.kv_heads

    def summary(self) -> str:
        return (f"P2-REAL-TOPO d={self.hidden} L={self.layers} gdn={self.n_gdn} full={self.n_full} "
                f"heads={self.heads} kv={self.kv_heads} hd={self.head_dim} inter={self.inter} "
                f"vocab={self.vocab} conv_k={self.conv_kernel} gdn_h={self.gdn_k_heads}x{self.gdn_k_dim}"
                f"/{self.gdn_v_heads}x{self.gdn_v_dim} qkv_n={self.q_gate_n}/{self.kv_n}/{self.kv_n} "
                f"conv_dim={self.conv_dim} rotary={self.rotary_dim} lora_r={self.lora_r} "
                f"lora_alpha={self.lora_alpha} dry={'true' if self.dry else 'false'} src={self.src}")

    def dry_scaled(self) -> "Topo":
        """小形状降级：**层数与层型逐项不动**（L=24、18 GDN + 6 全注意力），只缩各维宽度。

        为什么只缩宽度：真形档的判据是"层数/调用次数结构不变"，而 op 调用次数只由 层数 × 层型
        决定；宽度只影响每 op 的形状与耗时。缩宽度让 host 上秒级跑完，缩层数则会改调用次数结构。
        每个降级值都挑"仍落在 gemm 块阶梯(64,32,16)可整除档 + 与真档同为 16 的整除关系"上，
        使 **plan 接受/拒绝面与真档逐项同构**（唯一拒绝点：letter 头 n=26，两档同在）。
        """
        return replace(
            self, hidden=64, heads=2, kv_heads=1, head_dim=32, inter=128, vocab=256,
            gdn_k_dim=16, gdn_v_dim=16, dry=True,
        )


def load_topo(root: str, *, layers: int | None = None, dry: bool = False) -> Topo:
    """从配置原文读数建拓扑；配置缺失时退回带行号注释的字面量（并如实标注 src）。"""
    path = os.path.join(root, CONFIG_REL)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)["text_config"]
        n = int(cfg["num_hidden_layers"])                       # config.json:56
        lt = tuple(cfg["layer_types"])                          # config.json:19-44
        src = CONFIG_REL
        topo = Topo(
            hidden=int(cfg["hidden_size"]), layers=n,           # :16 / :56
            heads=int(cfg["num_attention_heads"]),              # :55
            kv_heads=int(cfg["num_key_value_heads"]),           # :57
            head_dim=int(cfg["head_dim"]),                      # :14
            inter=int(cfg["intermediate_size"]),                # :18
            vocab=int(cfg["vocab_size"]),                       # :61
            rms_eps=float(cfg["rms_norm_eps"]),                 # :58
            conv_kernel=int(cfg["linear_conv_kernel_dim"]),      # :45
            gdn_k_heads=int(cfg["linear_num_key_heads"]),        # :47
            gdn_k_dim=int(cfg["linear_key_head_dim"]),           # :46
            gdn_v_heads=int(cfg["linear_num_value_heads"]),      # :48
            gdn_v_dim=int(cfg["linear_value_head_dim"]),         # :49
            layer_types=lt,
            lora_r=16, lora_alpha=32,                            # gate08b.yaml:32-33
            rope_theta=float(cfg["rope_parameters"]["rope_theta"]),  # config.json:71
            src=src,
        )
    else:                                                        # 配置缺失：字面量（行号同上）
        n = 24
        topo = Topo(hidden=1024, layers=n, heads=8, kv_heads=2, head_dim=256, inter=3584,
                    vocab=248320, rms_eps=1e-6, conv_kernel=4, gdn_k_heads=16, gdn_k_dim=128,
                    gdn_v_heads=16, gdn_v_dim=128, layer_types=_layer_types(n),
                    lora_r=16, lora_alpha=32, src="builtin-literals（config.json 缺）")
    if layers is not None and layers != topo.layers:
        topo = replace(topo, layers=layers, layer_types=_layer_types(layers))
    if dry:
        topo = topo.dry_scaled()
    if len(topo.layer_types) != topo.layers:
        raise ValueError(f"layer_types={len(topo.layer_types)} 与 layers={topo.layers} 不符")
    return topo


# =============================================================== 账本 / 观察器
class Ledger:
    """一步之内所有可机器判的账：门面调用数/耗时、plan 请求与拒绝、无门面位点、编译键请求。"""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.times: dict[str, list[float]] = {}
        self.nofacade: dict[str, int] = {}
        self.plan_req: dict[str, int] = {}
        self.plan_rej: dict[str, int] = {}
        self.swaps: list[str] = []
        self.key_req: dict[str, int] = {}
        self.mem_calls: int = 0

    def bump(self, table: dict[str, int], key: str, n: int = 1) -> None:
        table[key] = table.get(key, 0) + n

    def nof(self, name: str, n: int = 1) -> None:
        """登记一处"没有自研内核、只能落底座"的位点（RMSNorm / repeat_kv / silu / 残差 …）。"""
        self.bump(self.nofacade, name, n)

    def native_total(self) -> int:
        """native = plan 拒绝点（swaps）+ 无门面位点总数（口径写死，见 SPEC §记账口径）。"""
        return len(self.swaps) + sum(self.nofacade.values())

    def p50(self, name: str) -> float | None:
        xs = sorted(self.times.get(name, ()))
        if not xs:
            return None
        return xs[len(xs) // 2]


#: plan 观察器要罩住的入口（族名 → 模块 → 属性）。**只读**：调原函数、看返回值，不改行为。
_PLAN_SPECS = (
    ("linear", gemm_asc, "plan"),
    ("linear_dW", gemm_bwd_dw_asc, "plan"),
    ("gdn_delta", gdn_asc, "plan"),
    ("gdn_delta_bwd", gdn_asc, "plan_backward"),
    ("gdn_conv", gdn_conv_asc, "plan"),
    ("gdn_conv_bwd", gdn_conv_asc, "plan_backward"),
    ("attn_sw", attn_sw_asc, "plan"),
    ("rope", rope_asc, "_plan"),
    ("readout", letter_readout_asc, "_plan"),
)


def _first_tensor_shape(args, kwargs) -> tuple[str, str]:
    """取本次调用的"代表形状/位宽"：第一个张量参数的 (shape, dtype)；取不到就交 '?'。"""
    for x in list(args):
        if torch.is_tensor(x):
            return str(tuple(x.shape)), str(x.dtype).replace("torch.", "")
    return "?", "?"


def install_plan_watch(ledger: Ledger) -> list[tuple]:
    """给 9 个 plan 入口套只读观察器：**plan 回 None 就是一次"拒形状落底座"**。

    为什么用观察器而不是自己在调度层复判一遍：复判要把每件的 dtype 归一（cpu 升 fp32、ascend 保位宽）、
    连续化、参数表都抄一遍，任何一处走样就会记错账；直接看真判据的返回值则**天然同源**。
    观察器不改返回值、不改参数、不包 get_compiled（那是"编译失败"那条腿，由既有
    `ascend_env.record_blocker` 的 warning 覆盖），所以既有件的语义零改动。
    """
    saved: list[tuple] = []
    for label, mod, attr in _PLAN_SPECS:
        orig = getattr(mod, attr)

        def make(label, orig):
            def wrapper(*args, **kwargs):
                spec = orig(*args, **kwargs)
                ledger.bump(ledger.plan_req, label)
                if spec is None:
                    shape, dt = _first_tensor_shape(args, kwargs)
                    ledger.bump(ledger.plan_rej, label)
                    tag = label
                    if label == "readout" and len(args) >= 6 and isinstance(args[5], str):
                        tag = f"readout/{args[5]}"
                    line = f"SWAP {tag} {shape} dtype={dt} -> native"
                    ledger.swaps.append(line)
                    print(line, flush=True)
                return spec

            wrapper.__name__ = getattr(orig, "__name__", attr)
            wrapper.__doc__ = getattr(orig, "__doc__", None)
            return wrapper

        setattr(mod, attr, make(label, orig))
        saved.append((mod, attr, orig))
    return saved


def remove_plan_watch(saved: list[tuple]) -> None:
    for mod, attr, orig in saved:
        setattr(mod, attr, orig)


def install_key_watch(ledger: Ledger) -> tuple:
    """包 `ascend_env.get_compiled`：**只记请求键与次数**，再原样转给真实现（零行为改动）。"""
    real_get = ascend_env.get_compiled

    def wrapped(key, builder, target=None):
        try:
            full = ascend_env.make_key(target, key)
            ledger.bump(ledger.key_req, full)
        except Exception:  # noqa: BLE001  记账失败绝不带崩主线
            pass
        return real_get(key, builder, target)

    ascend_env.get_compiled = wrapped
    return real_get


# =============================================================== 设备门桩（--no-run / --keys-only）
def install_device_gate() -> dict:
    """A3 同款三量桩：把"设备在场 + 探针可编 + 后端选中"写成常量，让编译请求真发生。

    取证出处：`reconcile/A3_probe_trainstep_route.py:60-62`（同一手法，run
    `1011-p2-13-p02-...` 的 `A3-STEP` 行）。仅在 `--no-run` / `--keys-only` 下装，装完只影响
    `ascend_env` 的**三个函数对象**（内存里），既不碰磁盘也不改实现。
    """
    before = {"npu_present": ascend_env._npu_present}
    ascend_env._npu_present = lambda: True
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG
    return before


def install_compile_only(ledger: Ledger, mode: str) -> tuple:
    """`compile` 档：真编译每个键、然后把发射壳换成空转桩；`keys` 档：只记账、返回 None 落 eager。

    返回 (恢复函数, 真编译清单)。空转桩形态照抄 `A3_probe_trainstep_route.py:77-103`。
    """
    import tilelang

    real_compile = tilelang.compile
    real_get = ascend_env.get_compiled
    compiles: list[tuple[str, float]] = []
    fails: list[str] = []
    bad: set[str] = set()                     # 编不过的键：记一次就够，别对着同一键反复撞

    class _Stub:
        """编译产物外壳：`__call__` 空转（不发射），其余属性转发真产物。"""

        def __init__(self, kern):
            self._kern = kern

        def __call__(self, *a, **kw):
            return None

        def __getattr__(self, name):
            return getattr(self._kern, name)

    def _prog_name(prog) -> str:
        for attr in ("name", "func_name", "kernel_name"):
            v = getattr(prog, attr, None)
            if isinstance(v, str) and v:
                return v
        return "?"

    def compile_wrap(prog, *a, **kw):
        t0 = time.perf_counter()
        nm = _prog_name(prog)
        try:
            kern = real_compile(prog, *a, **kw)
        except Exception as exc:  # noqa: BLE001  这正是要取证的那条腿
            dt = time.perf_counter() - t0
            fails.append(f"{nm}: {type(exc).__name__}: {str(exc)[:160]}")
            print(f"P2-REAL-COMPILE-FAIL {nm} sec={dt:.2f} {str(exc)[:160]}", flush=True)
            raise
        dt = time.perf_counter() - t0
        compiles.append((nm, dt))
        print(f"P2-REAL-COMPILE {nm} sec={dt:.2f}", flush=True)
        return _Stub(kern) if mode == "compile" else kern

    def get_wrap(key, builder, target=None):
        full = ascend_env.make_key(target, key)
        ledger.bump(ledger.key_req, full)
        if mode == "keys":
            return None                       # ⇒ 上层落回 eager，整步跑得完（只收键清单）
        if full in bad:
            return None                       # 已判编不过：直接兜底，不重复尝试
        if full in ascend_env._compiled:
            return ascend_env._compiled[full]
        try:
            kern = builder()
        except Exception as exc:  # noqa: BLE001
            ascend_env._blockers[full] = f"{type(exc).__name__}: {str(exc)[:120]}"
            bad.add(full)
            return None
        ascend_env._compiled[full] = kern
        ascend_env._counts["compile"] += 1
        return kern

    tilelang.compile = compile_wrap
    ascend_env.get_compiled = get_wrap

    def restore():
        tilelang.compile = real_compile
        ascend_env.get_compiled = real_get

    return restore, compiles, fails


# =============================================================== 调度台（混合栈分流）
class Dispatcher:
    """每个可微点一个方法：走 `autograd_asc` 门面、计时、计数；无门面位点单列 `nofacade`。

    计时口径照 `ascend/bench_card.py`：发前发后各 sync 一次（npu 上不同步会把排队时间算进别人账上）；
    cpu 上 sync 是空操作。
    """

    def __init__(self, target: str, device: torch.device, ledger: Ledger, out_dtype: torch.dtype):
        self.target = target
        self.device = device
        self.lg = ledger
        self.dt = out_dtype

    def _sync(self) -> None:
        if self.device.type == "npu":
            try:
                torch.npu.synchronize()
            except Exception:  # noqa: BLE001  —— 无 torch_npu / 未注册时不必带崩
                pass

    def _call(self, name: str, fn, *args, **kwargs):
        self.lg.bump(self.lg.counts, name)
        self.lg.times.setdefault(name, [])
        self._sync()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        self._sync()
        self.lg.times[name].append(time.perf_counter() - t0)
        return out

    # ---- 八个门面（逐个：走 autograd_asc，不动它）----
    def linear(self, A, W, bias=None):
        return self._call("linear", ops.linear, A, W, bias, out_dtype=self.dt, target=self.target)

    def rope(self, qkv, cos, sin):
        return self._call("rope", ops.rope, qkv, cos, sin,
                          rotate_slots=TS.rope_slots_default(), target=self.target)

    def attn_sw(self, q, k, v, window):
        return self._call("attn_sw", ops.attn_sw, q, k, v, window,
                          out_dtype=self.dt, target=self.target)

    def gdn_delta(self, q, k, v, g, beta):
        return self._call("gdn_delta", ops.gdn_delta, q, k, v, g, beta,
                          out_dtype=self.dt, target=self.target)

    def gdn_conv(self, x, w, bias=None):
        return self._call("gdn_conv", ops.gdn_conv, x, w, bias,
                          out_dtype=self.dt, target=self.target)

    def readout(self, rows, ids):
        return self._call("readout", ops.readout, rows, ids,
                          out_dtype=self.dt, target=self.target)

    def lora(self, x, a, b, scaling):
        return self._call("lora_apply", ops.lora_apply, x, a, b, scaling, None,
                          out_dtype=self.dt, target=self.target)


# =============================================================== 底座位点（无门面件）
def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3_5RMSNorm（`modeling:840-854`）：**零中心**——权重按 `(1 + w)` 用，w 初值 0。"""
    dt = x.dtype
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(-1, keepdim=True)
    return (x32 * torch.rsqrt(var + eps) * (1.0 + w.float())).to(dt)


def rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, w: torch.Tensor,
                   head_dim: int, eps: float) -> torch.Tensor:
    """Qwen3_5RMSNormGated（`modeling:217-233`）：逐头归一 + silu 门；权重是 ones（非零中心）。"""
    dt = x.dtype
    shape = x.shape
    xh = x.reshape(-1, head_dim)
    gh = gate.reshape(-1, head_dim)
    x32 = xh.to(torch.float32)
    var = x32.pow(2).mean(-1, keepdim=True)
    y = (x32 * torch.rsqrt(var + eps)).to(dt)
    y = y * w.to(dt)
    y = y * F.silu(gh.to(torch.float32)).to(dt)
    return y.reshape(shape)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """`modeling:711-720` 同口径：把 kv 小头按相邻分组补齐到 heads（GQA）。"""
    if n_rep == 1:
        return x
    t, kv, hd = x.shape
    return x[:, :, None, :].expand(t, kv, n_rep, hd).reshape(t, kv * n_rep, hd)


#: `spread_gate` 的"只喊一次"开关（18 层 × 2 前向都喊会淹掉判决行）
_GDN_SPREAD_ANNOUNCED = False


def spread_gate(gate: torch.Tensor, dk: int) -> torch.Tensor:
    """把"每 token 标量门" `(heads, seq)` 显式摊成"每键通道门" `(heads, seq, dk)`。

    【为什么是具名函数而不是就地一行】**这不是本事，是既有件缺陷的等价桥**，必须显式公告：
    `ascend/kernels/gdn_asc.py` 的 `backward` 在 docstring `:694` 自陈
    "`(dq, dk, dv, dg, dbeta)` 五个与输入同形的 fp32 梯度"，但实现里 `dg` 恒按 **`_gate3` 摊开后的
    3D 形**回报（昇腾伴随件路 `:716,720-722` 与 torch 尺子路 `:734` 都是 `gg/g3` 的 3D 形），
    即"`g` 收 2D 却回 3D"。`torch.autograd` 只在回报形状**不能**广播对齐到输入形状时才报
    `returned an invalid gradient at index 3`：`(heads,seq)`→`(1,heads,seq)` 与 `(heads,seq,dk)`
    对齐要求 `heads==seq 或 seq==dk`。于是：
      * **d=32 冒烟 / 本件 dry 档**（`heads` 与 `seq` 都是 16）**恰好被这层巧合盖住** —— 错的 3D
        回报被当成"可广播"，`g.grad` 被静默对齐成 2D；
      * **真形档**（`(16,512)` vs `(16,512,128)`，`512≠16` 且 `512≠128`）对齐失败 ⇒ 炸。

    本件不改禁改件，改为**在交接口自己摊开**：前向数学逐位不变（既有件的 `_gate3` 对 2D 输入
    本来就这么摊），反向则由 autograd 的 `ExpandBackward` 把 3D 回报**沿末维求和**折回
    `(heads, seq)` —— 与"修好 `gdn_asc.backward` 后应有的 2D 回报"**逐位等价**。等价性已在
    `reconcile/I3_probe_gdn_bwd_shape.py` 里取证（3D 交法 `SAME-SHAPE`，2D 交法在真形档
    `AUTOGRAD-RAISE`）。这处摊开计入 `COUNTS gdn_g_spread`，不静默。
    """
    global _GDN_SPREAD_ANNOUNCED
    if not _GDN_SPREAD_ANNOUNCED:
        print(f"GDN-G-SPREAD gform=2d->3d dk={dk} why=既有件 gdn_asc.backward 对 2D `g` 回报 3D "
              f"`dg`（违约其 :694 自陈『与输入同形』）；按等价广播交 3D，反向由 ExpandBackward 折回 2D",
              flush=True)
        _GDN_SPREAD_ANNOUNCED = True
    return gate.unsqueeze(-1).expand(-1, -1, dk).contiguous()


# =============================================================== 权重
class Weights:
    """0.8B 主干权重（**整体冻结**）+ LoRA 垫片（**唯一可训面**，`sft.py:146,756`）。"""

    def __init__(self, topo: Topo, device: torch.device, dtype: torch.dtype, seed: int):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.topo = topo
        self.device, self.dtype = device, dtype
        self.lora_tensors: list[torch.Tensor] = []

        def R(*shape, scale=0.02, grad=False):
            t = (torch.randn(*shape, generator=gen) * scale).to(device=device, dtype=dtype)
            if grad:
                t.requires_grad_(True)
            return t

        h, kh, kd, vh, vd = topo.hidden, topo.gdn_k_heads, topo.gdn_k_dim, topo.gdn_v_heads, topo.gdn_v_dim
        self.embed = R(topo.vocab, h)                                  # 与 lm_head 共享（:59 tie）
        self.final_norm = torch.zeros(h, device=device, dtype=dtype)    # 零中心 RMSNorm 的 w 初值 0
        self.layers: list[dict] = []
        self.lora: list[dict | None] = []
        r, alpha = topo.lora_r, topo.lora_alpha
        self.lora_scaling = alpha / r
        for t in topo.layer_types:
            lw = {"in_norm": torch.zeros(h, device=device, dtype=dtype),
                  "post_norm": torch.zeros(h, device=device, dtype=dtype),
                  "mlp_gate": R(topo.inter, h), "mlp_up": R(topo.inter, h), "mlp_down": R(h, topo.inter)}
            if t == "full_attention":
                lw.update({
                    "q_proj": R(topo.q_gate_n, h), "k_proj": R(topo.kv_n, h), "v_proj": R(topo.kv_n, h),
                    "o_proj": R(h, topo.o_in),
                    "q_norm": torch.zeros(topo.head_dim, device=device, dtype=dtype),
                    "k_norm": torch.zeros(topo.head_dim, device=device, dtype=dtype),
                })
                lora = {}
                for leaf, out_n, in_n in (("q_proj", topo.q_gate_n, h), ("k_proj", topo.kv_n, h),
                                          ("v_proj", topo.kv_n, h), ("o_proj", h, topo.o_in)):
                    a = R(r, in_n, scale=0.05, grad=True)
                    b = R(out_n, r, scale=0.0, grad=True)     # B 初值 0 ⇒ 起步时 delta=0（LoRA 常规）
                    lora[leaf] = (a, b)
                    self.lora_tensors += [a, b]
                self.lora.append(lora)
            else:
                lw.update({
                    "in_qkv": R(topo.conv_dim, h), "in_z": R(topo.gdn_val_dim, h),
                    "in_b": R(vh, h), "in_a": R(vh, h),
                    "conv_w": R(topo.conv_dim, topo.conv_kernel),
                    "A_log": R(vh, scale=0.0), "dt_bias": torch.ones(vh, device=device, dtype=dtype),
                    "gnorm_w": torch.ones(vd, device=device, dtype=dtype),
                    "out_proj": R(h, topo.gdn_val_dim),
                })
                self.lora.append(None)
            self.layers.append(lw)

    def total_params(self) -> int:
        n = self.embed.numel() + self.final_norm.numel()
        for lw in self.layers:
            n += sum(v.numel() for v in lw.values())
        return n + sum(t.numel() for t in self.lora_tensors)


# =============================================================== 前向
class Step:
    """一步的前向：24 层 × (GDN|全注意力) + 读点位头。每层 op 序列照 `modeling:860-913`。"""

    def __init__(self, topo: Topo, w: Weights, d: Dispatcher, lead: Ledger):
        self.t, self.w, self.d, self.lg = topo, w, d, lead

    # ---- 一层的前馈块（两型共用，`modeling:907-911` + `:823-836`）----
    def _mlp(self, h, lw):
        y = rms_norm(h, lw["post_norm"], self.t.rms_eps)
        self.lg.nof("rmsnorm")
        g = self.d.linear(y, lw["mlp_gate"])
        u = self.d.linear(y, lw["mlp_up"])
        m = F.silu(g) * u
        self.lg.nof("silu_mul")
        out = self.d.linear(m, lw["mlp_down"])
        self.lg.nof("residual_add")
        return h + out

    # ---- 线性注意力层（`modeling:503-662`）----
    def _gdn_layer(self, h, lw):
        y = rms_norm(h, lw["in_norm"], self.t.rms_eps)
        self.lg.nof("rmsnorm")
        qkv = self.d.linear(y, lw["in_qkv"])                     # (T, conv_dim=6144)
        z = self.d.linear(y, lw["in_z"])                         # (T, value_dim=2048)
        b = self.d.linear(y, lw["in_b"])                         # (T, num_v_heads=16)
        a = self.d.linear(y, lw["in_a"])                         # (T, num_v_heads=16)
        t, cd = qkv.shape[0], self.t.conv_dim
        # 短卷积走"通道优先"契约 (..., C, L)（`gdn_conv_asc.plan:308-346`：C=size(-2), L=size(-1)）
        xin = qkv.transpose(0, 1).unsqueeze(0).contiguous()      # (1, conv_dim, T)
        xc = self.d.gdn_conv(xin, lw["conv_w"], None)            # (1, conv_dim, T)，内含 +silu
        xc = xc.squeeze(0).transpose(0, 1).contiguous()          # (T, conv_dim)
        kd, kv = self.t.gdn_k_dim, self.t.gdn_k_heads
        vd, vh = self.t.gdn_v_dim, self.t.gdn_v_heads
        q = xc[:, :kw_len(self.t)].reshape(t, kv, kd).transpose(0, 1).contiguous()
        k = xc[:, kw_len(self.t):2 * kw_len(self.t)].reshape(t, kv, kd).transpose(0, 1).contiguous()
        v = xc[:, 2 * kw_len(self.t):].reshape(t, vh, vd).transpose(0, 1).contiguous()
        self.lg.nof("layout_split", 2)
        beta = b.sigmoid().transpose(0, 1).contiguous()           # (heads, T)（`modeling:616`）
        gate = (-lw["A_log"].float().exp()
                * F.softplus(a.float() + lw["dt_bias"].float())).transpose(0, 1).contiguous()
        self.lg.nof("gdn_gate")                                   # g/beta 的逐元素（`modeling:616-618`）
        gate3 = spread_gate(gate, self.t.gdn_k_dim)               # 见 `spread_gate` docstring（等价桥）
        self.lg.nof("gdn_g_spread")
        o = self.d.gdn_delta(q, k, v, gate3, beta)                # (heads, T, dv)，fp32 面唯一
        o = o.transpose(0, 1).reshape(t, self.t.gdn_val_dim)
        zz = z.reshape(t, self.t.gdn_val_dim)
        o = rms_norm_gated(o, zz, lw["gnorm_w"], vd, self.t.rms_eps)
        self.lg.nof("rmsnorm_gated")
        out = self.d.linear(o, lw["out_proj"])
        self.lg.nof("residual_add")
        h = h + out
        return self._mlp(h, lw)

    # ---- 全注意力层（`modeling:748-820`）----
    def _full_layer(self, h, lw, lora, cos, sin):
        y = rms_norm(h, lw["in_norm"], self.t.rms_eps)
        self.lg.nof("rmsnorm")
        # 三条投影 + LoRA（LoRA 加在投影出口，`sft.py:146` 只挂这四枚）
        q_gate = self.d.linear(y, lw["q_proj"]) + self.d.lora(y, *lora["q_proj"], self.w.lora_scaling)
        k = self.d.linear(y, lw["k_proj"]) + self.d.lora(y, *lora["k_proj"], self.w.lora_scaling)
        v = self.d.linear(y, lw["v_proj"]) + self.d.lora(y, *lora["v_proj"], self.w.lora_scaling)
        t, hh, hd = q_gate.shape[0], self.t.heads, self.t.head_dim
        qg = q_gate.reshape(t, hh, 2, hd)
        q = qg[:, :, 0, :].contiguous()                          # 前半=查询（`modeling:786-788`）
        g = qg[:, :, 1, :].reshape(t, hh * hd)                   # 后半=出口门
        self.lg.nof("layout_split")
        q = rms_norm(q, lw["q_norm"], self.t.rms_eps)             # 逐头 q_norm（`modeling:772,791`）
        k = rms_norm(k.reshape(t, self.t.kv_heads, hd), lw["k_norm"], self.t.rms_eps)
        v = v.reshape(t, self.t.kv_heads, hd)
        self.lg.nof("rmsnorm", 2)
        k = repeat_kv(k, self.t.group)                            # 小头补齐（`modeling:733-734`）
        v = repeat_kv(v, self.t.group)
        self.lg.nof("repeat_kv", 2)
        qkv = torch.stack([q, k, v], dim=1)                       # (T, 3, heads, hd)：packed 契约
        self.lg.nof("layout_pack")
        qkv = self.d.rope(qkv, cos, sin)                          # 只转 q/k 两槽
        qq = qkv[:, 0].transpose(0, 1).contiguous()               # (heads, T, hd)
        kk = qkv[:, 1].transpose(0, 1).contiguous()
        vv = qkv[:, 2].transpose(0, 1).contiguous()
        o = self.d.attn_sw(qq, kk, vv, t)                         # 全因果（window=seq）
        o = (o.transpose(0, 1).reshape(t, hh * hd)) * torch.sigmoid(g)
        self.lg.nof("attn_out_gate")
        out = self.d.linear(o, lw["o_proj"]) + self.d.lora(o, *lora["o_proj"], self.w.lora_scaling)
        self.lg.nof("residual_add")
        h = h + out
        return self._mlp(h, lw)

    def forward(self, ids, pos_ids, letter_ids, cos, sin, *, head: str, batch: int):
        h = self.d.readout(self.w.embed, ids)                     # token 嵌入（gather 门面）
        self.lg.nof("layout_embed")
        n_full = 0
        for i, t in enumerate(self.t.layer_types):
            if t == "full_attention":
                h = self._full_layer(h, self.w.layers[i], self.w.lora[i], cos, sin)
                n_full += 1
            else:
                h = self._gdn_layer(h, self.w.layers[i])
        h = rms_norm(h, self.w.final_norm, self.t.rms_eps)
        self.lg.nof("rmsnorm")

        # ---- 读点位头：生产口径是"末位 × 字母行"（`sft.py:533` → `decision/readout.py:90-114`）----
        rows = self.d.readout(h, pos_ids)                          # (B, d)：每行取末位
        letter_rows = self.w.embed.index_select(0, letter_ids.to(torch.long))   # (k, d)，冻结权重
        self.lg.nof("letter_rows_gather")                          # 冻结表上的取样（`readout.py:112`）
        if head == "vocab":
            logits = self.d.linear(h, self.w.embed)                # 全词表投影（tie，n=vocab）
            rows = logits.index_select(0, pos_ids.to(torch.long))  # (B, vocab)
            self.lg.nof("letter_select_cols")
        scores = self.d.linear(rows, letter_rows)                  # (B, k)：n=26
        return scores


def kw_len(t: Topo) -> int:
    """GDN 卷积输出里"键"那一段的长度（key_dim）；q/k/v 三段按 [key, key, value] 切（`modeling:602-610`）。"""
    return t.gdn_key_dim


# =============================================================== 一步
def _mem_gb(device: torch.device) -> tuple[float, float] | None:
    if device.type != "npu":
        return None
    try:
        return (torch.npu.memory_allocated() / 2 ** 30, torch.npu.max_memory_allocated() / 2 ** 30)
    except Exception:  # noqa: BLE001
        return None


def _analytic_mem_gb(t: Topo, seq: int, weights: int, batch: int,
                     dtype: torch.dtype = torch.float16) -> dict[str, float]:
    """A3 方法（`reconcile/A3_PREFLIGHT.md` §预算）的**解析上界**：权重 + LoRA 状态 + 最大激活。

    只算三项可闭式复算的大头：① 主干权重（按请求位宽：fp16 2B/元素、fp32 4B/元素）；
    ② LoRA master+Adam 两状态（fp32）；③ 最大瞬时激活 = 全注意力反向里重算的 scores
    `(heads, seq, seq)` fp32（`autograd_asc._AttnSwFn.backward:317` 确实物化这一张）。卷积那张
    `(1, conv_dim, seq)` 与 readout 的 (vocab,d) 零表（`_ReadoutFn.backward:511`，仅当 rows 需
    梯度时才建，本件冻结表不触发）单列，便于对账。
    """
    elem = 2 if dtype == torch.float16 else 4
    w_gb = weights * elem / 2 ** 30
    lora_numel = 0
    for out_n, in_n in ((t.q_gate_n, t.hidden), (t.kv_n, t.hidden), (t.kv_n, t.hidden), (t.hidden, t.o_in)):
        lora_numel += t.lora_r * in_n + out_n * t.lora_r
    lora_numel *= t.n_full
    opt_gb = lora_numel * 4 * 3 / 2 ** 30                        # master + m + v 三份 fp32
    attn_gb = t.n_full * t.heads * seq * seq * 4 / 2 ** 30       # 反向 scores 重算（fp32）
    conv_gb = t.conv_dim * seq * 4 / 2 ** 30
    return {"weights_gb": w_gb, "opt_gb": opt_gb, "attn_scores_gb": attn_gb, "conv_gb": conv_gb,
            "total_gb": w_gb + opt_gb + max(attn_gb, conv_gb), "lora_numel": float(lora_numel),
            "batch": float(batch)}


def run(target: str, device: torch.device, *, topo: Topo, seq: int, batch: int, head: str,
        dtype: torch.dtype, seed: int, lr: float, verbose: bool = True, compile_only: bool = False):
    """跑一步 fwd+bwd+opt（+再前向一次取 loss1），返回成果字典（供 main 打判决行）。"""
    torch.manual_seed(seed)
    lead = Ledger()
    watch = install_plan_watch(lead)
    key_real = install_key_watch(lead)

    if device.type == "npu":
        try:
            torch.npu.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001
            pass
    t0 = time.perf_counter()
    guarded = dtype == torch.float16
    try:
        w = Weights(topo, device, dtype, seed)
        step = Step(topo, w, Dispatcher(target, device, lead, dtype), lead)
        gen = torch.Generator(device="cpu").manual_seed(seed + 1)
        ids = torch.randint(0, topo.vocab, (seq,), generator=gen).to(device)
        pos_ids = torch.full((batch,), seq - 1, dtype=torch.long, device=device)
        letter_ids = torch.arange(LETTER_COUNT, dtype=torch.long, device=device) % topo.vocab
        # 角表：按 `production/backbone/layout.expand_rope_tables` 口径扩到整头宽（恒等角补尾）
        half = topo.head_dim // 2
        r_half = topo.rotary_dim // 2
        pos = torch.arange(seq, device=device).float().unsqueeze(1)
        j = torch.arange(r_half, device=device).float().unsqueeze(0)
        inv = topo.rope_theta / (10000.0 ** (2.0 * j / topo.head_dim))
        ang = pos * inv
        cos = torch.cat([torch.cos(ang),
                         torch.ones(seq, half - r_half, device=device)], dim=-1)
        sin = torch.cat([torch.sin(ang),
                         torch.zeros(seq, half - r_half, device=device)], dim=-1)
        tgt = torch.rand(batch, LETTER_COUNT, generator=gen).to(device)
        tgt = (tgt / tgt.sum(-1, keepdim=True)).float()

        if guarded:
            install_reduction_guard()
        scores0 = step.forward(ids, pos_ids, letter_ids, cos, sin, head=head, batch=batch)
        loss0 = -(tgt * torch.log_softmax(scores0.float(), dim=-1)).sum(-1).mean()
        lead.nof("loss_ce")

        params = list(w.lora_tensors)
        for p in params:
            p.grad = None
        if not compile_only:
            loss0.backward()
        missing = [i for i, p in enumerate(params) if p.grad is None]
        master = [p.detach().float().requires_grad_(True) for p in params]
        for m, p in zip(master, params):
            m.grad = None if p.grad is None else p.grad.detach().float()
        torch.optim.Adam(master, lr=lr).step()
        with torch.no_grad():
            for m, p in zip(master, params):
                p.copy_(m.detach().to(p.dtype))
        scores1 = step.forward(ids, pos_ids, letter_ids, cos, sin, head=head, batch=batch)
        loss1 = -(tgt * torch.log_softmax(scores1.float(), dim=-1)).sum(-1).mean()
        l0, l1 = float(loss0.detach().item()), float(loss1.detach().item())
        audit = {
            "dtype": "fp16" if guarded else "fp32",
            "params": len(params),
            "fp16": sum(1 for p in params if p.dtype == torch.float16),
            "violations": reduction_violations() if guarded else [],
        }
    finally:
        if guarded:
            uninstall_reduction_guard()
        wall = time.perf_counter() - t0
        remove_plan_watch(watch)
        ascend_env.get_compiled = key_real

    mem = _mem_gb(device)
    return {
        "ledger": lead, "loss0": l0, "loss1": l1, "missing": missing, "wall": wall,
        "audit": audit, "params_total": w.total_params(), "mem": mem,
        "analytic": _analytic_mem_gb(topo, seq, w.total_params(), batch, dtype),
        "compile_count": ascend_env.compile_count(), "keys": ascend_env.compiled_keys(),
    }


# =============================================================== 判决行
def _print_ledger(lead: Ledger, out) -> None:
    for op in sorted(lead.counts):
        p50 = lead.p50(op)
        print(f"COUNTS {op} calls={lead.counts[op]} p50_ms="
              f"{'n/a' if p50 is None else f'{p50 * 1e3:.3f}'}", file=out)
    for op in sorted(lead.nofacade):
        if lead.nofacade[op]:
            print(f"NOFACADE {op} calls={lead.nofacade[op]}", file=out)
    for fam in sorted(lead.plan_req):
        print(f"PLAN {fam} requested={lead.plan_req[fam]} rejected={lead.plan_rej.get(fam, 0)}", file=out)
    for key, n in sorted(lead.key_req.items()):
        print(f"COMPILE-KEY {key} requests={n}", file=out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", default=None, choices=list(ascend_env.TARGETS))
    ap.add_argument("--device", default=None, help="cpu 轨默认 cpu；ascend 轨默认 npu")
    ap.add_argument("--dry", action="store_true", help="小形状降级（层数/层型/调用次数结构不变）")
    ap.add_argument("--dtype", default=None, choices=sorted(DTYPE_BY_NAME),
                    help="位宽档：cpu 轨默认 fp32，ascend 轨默认 fp16（生产位宽）")
    ap.add_argument("--seq", type=int, default=None, help="一桌 token 数（缺省：dry→16，真档→8192）")
    ap.add_argument("--layers", type=int, default=None, help="覆盖层数（缺省取 config.json:56）")
    ap.add_argument("--batch", type=int, default=1, help="读点位行数 B（组批细节，config 不定；缺省 1）")
    ap.add_argument("--head", default="letter", choices=("letter", "vocab"),
                    help="读出头：letter=生产读点位(n=26) / vocab=全词表投影(tie, n=248320)")
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--lr", type=float, default=1e-4, help="LoRA 学习率（gate08b.yaml:38）")
    ap.add_argument("--no-run", action="store_true", help="只编译（A3 三量桩 + 发射桩），不真跑")
    ap.add_argument("--keys-only", action="store_true", help="只枚举编译键（不编译、落 eager 跑完）")
    ap.add_argument("--stub-run", action="store_true",
                    help="桩档下跑完整 fwd+bwd（编译正/反向全部键；发射被桩掉 ⇒ 数值无意义，仅供预算）")
    args = ap.parse_args(argv)

    name = ascend_env.normalize_target(args.target)
    dtype_name = args.dtype or ("fp16" if name == ascend_env.TARGET_ASCEND else "fp32")
    dt = DTYPE_BY_NAME[dtype_name]
    seq = args.seq if args.seq is not None else (16 if args.dry else 8192)
    dev_default = "npu" if name == ascend_env.TARGET_ASCEND else "cpu"

    stub_mode = args.no_run or args.keys_only or args.stub_run
    if name == ascend_env.TARGET_ASCEND and not stub_mode:
        reg = ensure_torch_npu()
    else:
        reg = {"status": "skip", "version": "not-imported",
               "detail": "cpu 轨 / 桩档不 import torch_npu"}
    print(npu_reg_line(reg, name))

    topo = load_topo(_ROOT, layers=args.layers, dry=args.dry)
    topo = replace(topo, dry=args.dry)
    print(topo.summary())

    try:
        dev = torch.device(args.device or dev_default)
    except Exception as exc:  # noqa: BLE001
        print(f"P2-REAL-STEP FAIL: NPU-REG-FAIL 设备名不可用 {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_DEVICE_UNAVAILABLE

    seen: list[str] = []
    show = warnings.showwarning

    def _hook(message, category, filename, lineno, file=None, line=None):
        seen.append(str(message))
        show(message, category, filename, lineno, file, line)

    warnings.showwarning = _hook
    restore = None
    saved_gate = None
    comp_fails: list[str] = []
    try:
        if stub_mode:
            saved_gate = install_device_gate()
            mode = "keys" if args.keys_only else "compile"
            restore, _compiles, comp_fails = install_compile_only(Ledger(), mode)
        if name == ascend_env.TARGET_ASCEND and dev.type != "npu" and not stub_mode:
            print(f"WARN: target=ascend 但 device={dev}，非桩档下 active_backend 会判 torch_eager",
                  file=sys.stderr)
        res = run(name, dev, topo=topo, seq=seq, batch=args.batch, head=args.head, dtype=dt,
                  seed=args.seed, lr=args.lr, verbose=True,
                  compile_only=args.no_run)
    except Exception as exc:  # noqa: BLE001
        print(f"P2-REAL-STEP-{name} FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    finally:
        warnings.showwarning = show
        if restore is not None:
            restore()
        if saved_gate is not None:
            ascend_env._npu_present = saved_gate["npu_present"]

    lead: Ledger = res["ledger"]
    n_ok = len(res["missing"])
    n_lora = res["audit"]["params"]
    swaps, native = len(lead.swaps), lead.native_total()
    reduced = res["loss1"] < res["loss0"]
    mode_label = "compile-only" if args.no_run else ("keys-only" if args.keys_only
                                                     else ("stub-run" if args.stub_run else "run"))
    if args.no_run:
        # `--no-run` 跳反向 + 桩掉发射 ⇒ 数值位**按构造无意义**，如实置 void（A3 同口径：
        # 「这一趟数值来自 eager 兜底，不证明内核数值」）。不许把未初始化读数当判决。
        print(f"P2-REAL-STEP target={name} loss0=void loss1=void reduced=n/a params={n_lora} "
              f"grads_ok=n/a swaps={swaps} native={native} mode={mode_label}")
    else:
        print(f"P2-REAL-STEP target={name} loss0={res['loss0']:.6f} loss1={res['loss1']:.6f} "
              f"reduced={reduced} params={n_lora} grads_ok={n_lora - n_ok}/{n_lora} "
              f"swaps={swaps} native={native} mode={mode_label}")
    print(f"P2-REAL-PARAMS total={res['params_total']} trainable={n_lora} "
          f"frozen={res['params_total'] - n_lora} seq={seq} batch={args.batch} "
          f"head={args.head} dtype={dtype_name} dry={'true' if args.dry else 'false'}")

    _print_ledger(lead, sys.stdout)

    fb = ops.count_fallback_events(seen)
    report = ops.kernel_usage_report(name, fallback_events=fb)
    print(report["line"])
    print(dtype_trace_line(res["audit"], fb))
    print(f"P2-REAL-COMPILE count={res['compile_count']} keys={len(res['keys'])} "
          f"fails={len(comp_fails)}")
    mem = res["mem"]
    if mem is None:
        print("P2-REAL-MEM alloc_gb=n/a peak_gb=n/a (device=%s；非 npu 设备无此读数)" % dev.type)
    else:
        print(f"P2-REAL-MEM alloc_gb={mem[0]:.3f} peak_gb={mem[1]:.3f}")
    an = res["analytic"]
    print(f"P2-REAL-MEM-ANALYTIC weights_gb={an['weights_gb']:.3f} opt_gb={an['opt_gb']:.3f} "
          f"attn_scores_gb={an['attn_scores_gb']:.3f} conv_gb={an['conv_gb']:.3f} "
          f"total_gb={an['total_gb']:.3f} lora_numel={int(an['lora_numel'])}")
    print(f"P2-REAL-WALL sec={res['wall']:.2f}")

    gating = (name == ascend_env.TARGET_ASCEND or os.environ.get(ENV_KERNEL_USAGE_STRICT) == "1")
    if stub_mode:
        return 0
    if res["missing"]:
        print(f"P2-REAL-STEP FAIL: grad 缺失 {res['missing']}", file=sys.stderr)
        return 2
    if not reduced:
        print(f"P2-REAL-STEP FAIL: loss 未下降（{res['loss0']} → {res['loss1']}）", file=sys.stderr)
        return 3
    if res["audit"].get("violations"):
        return EXIT_DTYPE_DISCIPLINE
    if gating and not report["ok"]:
        return EXIT_KERNEL_UNUSED
    return 0


if __name__ == "__main__":
    sys.exit(main())
