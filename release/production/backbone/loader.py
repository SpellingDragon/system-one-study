"""loader — 从 HF safetensors 直读权重、换成自研栈布局，并与 HF 一层前向对拍。

【做什么】
    三件事：① 按 index 从分片文件里只读出需要的那几块张量（不把整个模型搬进内存）；
    ② 调 `layout` 把一层的注意力权重换成自研栈口径（打包 qkv / 拆门 / 补齐小头 / 换座次）；
    ③ 用换好的权重按自研栈的算子顺序复跑这一层的注意力，与 HF 同一个模块的输出对拍，
    同时把"逐元素相等"的抽样断言也交出来——两路都过才算布局接缝合上。

【怎么做】
    读文件走 `safetensors.safe_open`，文件名由 `model.safetensors.index.json` 的 weight_map 决定，
    同一分片只开一次（一次一层只要 8 块张量，几 MB 而已）。逐头缩放另有一处折算：Qwen3_5RMSNorm
    的口径是 `norm(x) * (1 + w)`（w 零初始化），而自研栈的融合件只认 `norm(x) * s`，故在换算时
    把 `1 + w` 折进系数里（`layout.fold_unit_shift`），并对拍时按同一口径断言。文本栈的键前缀不写死：从键名里
    探出来（Qwen3.5 系是 `model.language_model.layers.`，纯文本栈是 `model.layers.`），因为两类
    检查点都在这条路上跑。对拍用"把 HF 那一个子模块复制成 fp32"当参考侧：两侧都在 fp32 域，
    容差能压到 1e-4 级；整模型保持 fp16，不动载入口径。

【为什么】
    布局错一位，权重的数字还是原数字，只是去了别的坐标——单看每块张量全都"对"，合起来算就
    是错的，所以断言必须落在"逐元素相等 + 一层输出相等"两级上。
    被否方案一：只比 shape（形状对了就当布局对了）——本文件的全部换算都是同形状的行列重排，
    只比 shape 等于什么都没验。
    被否方案二：从已载入的 fp16 模型里取权重再转换——那把"读文件"这条路径跳过了，检查点名
    称/分片/前缀错了也发现不了；本文件的职责之一就是替 p2-13 把"从 safetensors 起算"这条路
    先趟平。
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from . import layout as L

#: 一层注意力要动到的 HF 权重名（后缀口径；Qwen3.5 系实证：q_proj 带门、kv 两路、无 bias）
ATTN_NAMES = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
              "q_norm.weight", "k_norm.weight")
#: 可选的偏置名（attention_bias=True 的检查点才有，缺了就如实交 None）
ATTN_BIAS_NAMES = ("q_proj.bias", "k_proj.bias", "v_proj.bias")


@dataclass
class LayerConversion:
    """一层注意力的自研栈产物 + 对拍用的索引凭据。"""

    layer_idx: int
    tensors: dict[str, torch.Tensor]          # 转换后的权重（qkv.weight / gate.weight / o_proj.weight / 两个 norm）
    layout: L.HeadLayout                      # 这层的头布局
    row_index: dict[str, torch.Tensor] = field(default_factory=dict)   # 打包行 → HF 行（逐元素断言用）
    gated: bool = True
    notes: tuple[str, ...] = ()

    def keys(self) -> list[str]:
        """产物里有哪些键（进 run notes，便于事后核对"转换到底交了几块"）。"""
        return sorted(self.tensors)


def read_index(snapshot_dir: str | Path) -> dict[str, Any]:
    """读 model.safetensors.index.json（weight_map + 总规模）；没有分片 index 时回空表。

    白话：先把这本"页码目录"翻开——哪个数记在哪一页、那一页又在哪一本书里。后面点名
    取数全照着它走；没有目录时不猜，交一张空表出去，让调用侧自己决定要不要另想办法。"""
    path = Path(snapshot_dir)
    for cand in sorted(path.glob("*.safetensors.index.json")):
        return json.loads(cand.read_text(encoding="utf-8"))
    return {"weight_map": {}, "metadata": {}}


def text_prefix(index: dict[str, Any]) -> str:
    """从 weight_map 的键名里探出文本栈的层前缀（不把 language_model 写死）。

    白话：两本账的页码编法不一样——多模态那本把正文记在"语言子模型"名下，纯文本那本直接记在
    "模型"名下。这里翻一遍目录，看见 `.layers.0.` 出现在哪个前缀后面就认哪个，省得写死后
    换一种检查点就抓瞎。

    :raises KeyError: index 里找不到任何 `.layers.0.` 形态的键。
    """
    keys = list(index.get("weight_map", {}))
    if not keys:
        raise KeyError("index 里没有 weight_map：检查点不完整或不是 safetensors 形态")
    probes = [k for k in keys if k.startswith("mtp.") is False and ".layers.0." in k]
    if not probes:
        raise KeyError(f"weight_map 里找不到文本层（样本键：{keys[:5]}）")
    probe = next((k for k in probes if "language_model" in k), probes[0])
    return probe.split(".layers.0.")[0] + "."


def read_tensors(snapshot_dir: str | Path, names: Sequence[str], index: dict[str, Any] | None = None,
                 *, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    """按名字从 safetensors 分片里读张量（同一分片只打开一次），可选统一转精度。

    白话：点名要哪几页就从哪几本里撕下来，一本翻一次就够；翻不到的名字当场说"这页没有"，
    不拿空白页糊弄。
    """
    root = Path(snapshot_dir)
    idx = index or read_index(root)
    wmap = idx["weight_map"]
    missing = [n for n in names if n not in wmap]
    if missing:
        raise KeyError(f"safetensors 里没有这些张量：{missing[:6]}")
    by_shard: dict[str, list[str]] = {}
    for n in names:
        by_shard.setdefault(wmap[n], []).append(n)
    out: dict[str, torch.Tensor] = {}
    for shard, wanted in by_shard.items():
        with safe_open(str(root / shard), framework="pt") as f:
            for n in wanted:
                t = f.get_tensor(n)
                out[n] = t.to(dtype) if dtype is not None else t
    return out


def convert_attention_layer(
    snapshot_dir: str | Path,
    layer_idx: int,
    text_config,
    *,
    prefix: str | None = None,
    index: dict[str, Any] | None = None,
    compute_dtype: torch.dtype = torch.float32,
    norm_plus_one: bool = True,
) -> LayerConversion:
    """把一层的注意力换成自研栈布局（产出打包 qkv + 拆出的门 + 换座次的两个逐头缩放）。

    白话：这一层的材料原本散成六页——查询（带阀门）、键、值、回投，外加查询和键各自的逐坐标
    缩放。这里把它们收拾成一摞内核认得的打包件：阀门单独摘出来、小头补齐、每头内部换成
    内核的坐次。

    :param snapshot_dir: 快照目录（HF 检查点本体）。
    :param layer_idx: 层号（Qwen3.5-0.8B 实证：3/7/11… 是整注意力，0/1/2… 是线性注意力）。
    :param text_config: HF 的文本 config（提供 heads / kv_heads / head_dim / partial 比例）。
    :param prefix: 文本栈键前缀；None 时由 `text_prefix()` 探。
    :param index: 预先读好的 index（多轮转换时省一次磁盘）。
    :param compute_dtype: 转换产出的精度（默认 fp32，对拍在 fp32 域做）。
    :returns: `LayerConversion`。
    :raises ValueError: 层的行数与头布局不符（说明这层不是同一族注意力）。
    """
    idx = index or read_index(snapshot_dir)
    pfx = prefix or text_prefix(idx)
    lay = L.HeadLayout.from_config(text_config)
    base = f"{pfx}layers.{layer_idx}.self_attn."
    have = set(idx["weight_map"])
    names = [base + n for n in ATTN_NAMES if base + n in have]
    raw = read_tensors(snapshot_dir, names, idx, dtype=compute_dtype)
    q_raw, k_raw, v_raw = raw[base + "q_proj.weight"], raw[base + "k_proj.weight"], raw[base + "v_proj.weight"]
    o_raw = raw[base + "o_proj.weight"]

    gated = q_raw.shape[0] == lay.heads * lay.head_dim * 2
    notes: list[str] = []
    if gated:
        q_w, gate_w = L.split_gated_q(q_raw, lay)
        notes.append("attn_output_gate=true：q_proj 每头 head_dim×2，已拆出 gate.weight")
    else:
        q_w, gate_w = q_raw, None
        notes.append("该层 q_proj 未开门（每头 head_dim），gate 缺席")

    packed = L.pack_qkv(q_w, k_raw, v_raw, lay)
    conv_tensors = {"qkv.weight": packed, "o_proj.weight": o_raw.contiguous()}
    if gate_w is not None:
        conv_tensors["gate.weight"] = gate_w
    for extra in ("q_norm.weight", "k_norm.weight"):
        if base + extra in raw:
            moved = L.permute_norm(raw[base + extra], lay)
            conv_tensors[extra] = L.fold_unit_shift(moved) if norm_plus_one else moved
    if norm_plus_one and any(k.endswith("_norm.weight") for k in conv_tensors):
        notes.append("q_norm/k_norm 系数已折算 (1+w)：Qwen3_5RMSNorm 的读数是『零起步加一格』，"
                     "自研栈融合件只认『读数直接乘』，故把垫底的 1 折进系数")

    bias_names = [base + n for n in ATTN_BIAS_NAMES if base + n in have]
    if len(bias_names) == 3:
        braw = read_tensors(snapshot_dir, bias_names, idx, dtype=compute_dtype)
        packed_b = L.attention_bias_pack(braw[base + "q_proj.bias"], braw[base + "k_proj.bias"],
                                         braw[base + "v_proj.bias"], lay)
        conv_tensors["qkv.bias"] = packed_b
    elif bias_names:
        raise ValueError(f"偏置形态不完整（只见到 {bias_names}），无法拼打包偏置")
    else:
        notes.append("attention_bias=false：三条投影无偏置（qkv.bias 键缺席，非全零）")

    row_index = {
        "q": L.rotated_row_index(lay.heads, lay, lay.heads),
        "k": L.rotated_row_index(lay.heads, lay, lay.kv_heads),
        "v": L.plain_row_index(lay.heads, lay, lay.kv_heads),
    }
    return LayerConversion(layer_idx=layer_idx, tensors=conv_tensors, layout=lay, row_index=row_index,
                           gated=gated, notes=tuple(notes))


# ---------------------------------------------------------------- 自研栈一层前向（对拍参考侧）
def rmsnorm_rows(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """逐行按自身均方根缩放、再逐坐标配一个系数（fp32 累加，与 HF 的 RMSNorm 同式）。

    白话：每一行先按自己的平均响度压到同一档，然后每个坐标再拧一下属于它的音量旋钮；
    旋钮跟着坐标一起搬过家，所以"先搬座次再拧旋钮"与"先拧旋钮再搬座次"算的是同一件事。
    """
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return weight.to(torch.float32) * (xf * torch.rsqrt(var + eps))


def norm_weight_view(weight: torch.Tensor | None, lay: L.HeadLayout) -> torch.Tensor | None:
    """把逐坐标缩放系数整形成能广播的形状：全头共用则 (hd,)，逐头各配则 (heads, hd)。

    白话：音量旋钮有两种配法——要么一层共用一副、要么每个头自己一副。这里先看副数够不够
    分给每个头，够就一头一副地摆开，不够就原样一副横着放，让后面按行乘的时候自动对上。
    """
    if weight is None:
        return None
    n = int(weight.numel())
    if n == lay.head_dim:
        return weight
    if n == lay.heads * lay.head_dim:
        return weight.view(lay.heads, lay.head_dim)
    raise ValueError(f"逐坐标缩放长度 {n} 既不是 head_dim={lay.head_dim} 也不是 heads*head_dim={lay.heads * lay.head_dim}")


def self_stack_attention(
    conv: LayerConversion, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, eps: float
) -> torch.Tensor:
    """用自研栈的三件入口（打包 gemm → packed rope → 因果滑窗）跑一层的注意力，`(T,d)` 进 `(T,d)` 出。

    白话：这一层原本散成六页材料，这里按内核认得的顺序重做一遍同一道工：先把每行数字
    一次折成三摞（查询/键/值），前两摞先调响度再按位置转角（转角用的是"前后两半互相搭一手"
    那种坐法），然后每个位置只朝后看把内容加权合拢，合完先过一道阀门、最后折回一行宽交出去。
    """
    from sys1.kernels import attn_sw_kernel, gemm_kernel, rope_kernel

    lay = conv.layout
    tokens = int(x.shape[0])
    packed = gemm_kernel.forward(x, conv.tensors["qkv.weight"], None, "none", torch.float32)
    qkv = packed.view(tokens, 3, lay.heads, lay.head_dim).contiguous()
    qn = norm_weight_view(conv.tensors.get("q_norm.weight"), lay)
    kn = norm_weight_view(conv.tensors.get("k_norm.weight"), lay)
    if qn is not None:
        qkv[:, 0] = rmsnorm_rows(qkv[:, 0], qn, eps)
    if kn is not None:
        qkv[:, 1] = rmsnorm_rows(qkv[:, 1], kn, eps)
    c_exp, s_exp = L.expand_rope_tables(cos, sin, lay)
    qkv = rope_kernel.forward(qkv, c_exp, s_exp)
    q, k, v = (qkv[:, s].transpose(0, 1) for s in (0, 1, 2))        # 打包 (T,3,H,hd) → 每路 (heads, T, hd)
    out = attn_sw_kernel.forward(
        q, k, v, window=tokens, scale=lay.head_dim ** -0.5, out_dtype=torch.float32
    )
    out = out.transpose(0, 1).reshape(tokens, lay.heads * lay.head_dim)
    if conv.gated:
        gate = gemm_kernel.forward(x, conv.tensors["gate.weight"], None, "none", torch.float32)
        out = out * torch.sigmoid(gate)
    return gemm_kernel.forward(out, conv.tensors["o_proj.weight"], None, "none", torch.float32)


def sample_element_match(conv: LayerConversion, hf_tensors: dict[str, torch.Tensor],
                         *, sample_rows: int = 2048, seed: int = 0) -> dict[str, Any]:
    """抽样逐元素对拍：自研栈打包权重的每一行，必须与来源那一行的每个数都一模一样。

    白话：本文件的换算只做"搬行"，一个数都不改。所以随机点一批行号，把自研栈这一行与来源
    那一行逐个数对着看；有一位不同就是行号列子排歪了，当场把槽位、行号与首处差异报出来，
    免得事后拿"形状对得上"当"布局对得上"。
    """
    lay = conv.layout
    block = lay.heads * lay.head_dim
    packed = conv.tensors["qkv.weight"]
    q_full = hf_tensors["q_proj.weight"]
    q_src = L.split_gated_q(q_full, lay)[0] if conv.gated else q_full
    sources = {"q": q_src, "k": hf_tensors["k_proj.weight"], "v": hf_tensors["v_proj.weight"]}
    g = torch.Generator().manual_seed(seed)
    rows = torch.randint(0, block, (min(sample_rows, block),), generator=g)
    report: dict[str, Any] = {"sampled_rows": int(rows.numel()), "checked": {}, "mismatch": []}
    for slot, name in enumerate(("q", "k", "v")):
        idx = conv.row_index[name].to(torch.long)[rows]
        ours = packed[slot * block + rows]
        want = sources[name].index_select(0, idx)
        eq = torch.eq(ours, want).all(dim=1)
        report["checked"][name] = int(rows.numel())
        for pos in (~eq).nonzero().flatten().tolist()[:5]:
            report["mismatch"].append(f"slot{slot}/{name}: 打包行 {int(rows[pos])} 与源行 "
                                       f"{int(idx[pos])} 首个差异在第 {int((ours[pos] != want[pos]).nonzero().flatten()[0])} 列")
    for extra in ("q_norm.weight", "k_norm.weight"):
        if extra in hf_tensors and extra in conv.tensors:
            want = L.fold_unit_shift(L.permute_norm(hf_tensors[extra], lay))
            if not torch.equal(conv.tensors[extra], want):
                report["mismatch"].append(f"{extra} 与『换座次 + 折算 (1+w)』的结果逐元素不等")
    if conv.gated:
        gate_src = L.split_gated_q(q_full, lay)[1]
        if not torch.equal(conv.tensors["gate.weight"], gate_src):
            report["mismatch"].append("gate.weight 与 q_proj 后半段逐元素不等")
    if not torch.equal(conv.tensors["o_proj.weight"], hf_tensors["o_proj.weight"]):
        report["mismatch"].append("o_proj.weight 与源张量逐元素不等（该块本不该改动）")
    report["ok"] = not report["mismatch"]
    return report


@dataclass
class LayerParity:
    """一层前向对拍的实测结论：最大绝对差、相对差、判定与一行文本凭据。"""

    layer_idx: int
    max_abs_diff: float
    mean_abs_ref: float
    rel_diff: float
    out_shape: tuple[int, ...]
    tol: float
    ok: bool
    notes: tuple[str, ...] = ()

    def as_note(self) -> str:
        """一行文本凭据（进 run notes：层号 / 最大差 / 相对差 / 判定，事后可复跑核对）。

        白话：把这层的比账压成一行字——第几层、两边最大差多少、参考侧平均多重、折算成
        倍数是多少、允许的限度划在哪儿、最后判过没过——一行就能直接进记录本。"""
        flag = "OK" if self.ok else "FAIL"
        return (f"layer{self.layer_idx} out={list(self.out_shape)} max_abs={self.max_abs_diff:.3e} "
                f"mean_ref={self.mean_abs_ref:.3e} rel={self.rel_diff:.3e} tol={self.tol:.1e} {flag}")


def check_layer_parity(
    text_stack: Any, conv: LayerConversion, *, seq: int = 16, seed: int = 0,
    tol: float = 1e-4, eps: float = 1e-6,
) -> LayerParity:
    """把 HF 的这一个注意力子模块复制成 fp32 当参考侧，与自研栈那一路逐位对拍。

    白话：同一份材料、两种做法，做同一道工序，再把两边的成品摆在一起比轻重。参考那侧照出厂
    说明单上的写法走（六个零件各算各的），我们这侧按内核的口径走（一摞打包、换座次、只朝后看）；
    两边都用单精度、都喂同一串随机数字与同一张角度表，差的倍数落在容差内才算接缝合上。
    """
    import copy

    host = text_stack.layers[conv.layer_idx]
    module = getattr(host, "self_attn", None)
    if module is None:
        raise ValueError(f"第 {conv.layer_idx} 层没有 self_attn（该层是线性注意力），换整注意力层再对拍")
    ref = copy.deepcopy(module).float().eval()
    cfg = copy.deepcopy(ref.config)
    cfg._attn_implementation = "eager"
    ref.config = cfg
    hidden = int(ref.q_proj.in_features)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(seq, hidden, generator=g, dtype=torch.float32)
    pos = torch.arange(seq).view(1, 1, -1).expand(3, 1, -1)
    dummy = torch.zeros(1, seq, hidden, dtype=torch.float32)
    cos, sin = (t[0] for t in text_stack.rotary_emb(dummy, pos))
    mask = torch.full((seq, seq), float("-inf"), dtype=torch.float32).triu(1).view(1, 1, seq, seq)
    with torch.no_grad():
        ref_out, _ = ref(
            x.unsqueeze(0), position_embeddings=(cos.unsqueeze(0), sin.unsqueeze(0)),
            attention_mask=mask, past_key_values=None,
        )
        ours = self_stack_attention(conv, x, cos, sin, eps=eps)
    diff = (ref_out[0] - ours).abs()
    max_abs = float(diff.max())
    mean_ref = float(ref_out[0].abs().mean())
    rel = max_abs / mean_ref if mean_ref else max_abs
    return LayerParity(
        layer_idx=conv.layer_idx, max_abs_diff=max_abs, mean_abs_ref=mean_ref, rel_diff=rel,
        out_shape=tuple(ours.shape), tol=tol, ok=rel <= tol, notes=conv.notes,
    )


def full_attention_layers(text_config: Any) -> list[int]:
    """从 config 的 layer_types 里挑出"整注意力"的层号（线性注意力层没有 self_attn）。

    白话：这副身板二十四层里有二十层走的是另一条路（不按"只朝后看"这套做），只有标了
    full_attention 的那几层才是本文件能对拍的对象；先把页码列出来，省得一层层撞墙再猜为什么。
    """
    types = list(getattr(text_config, "layer_types", None) or [])
    return [i for i, t in enumerate(types) if t == "full_attention"]


def verify_attention_layer(snapshot_dir: str | Path, layer_idx: int, text_stack: Any, *,
                           text_config: Any = None, seq: int = 16, tol: float = 1e-4, seed: int = 0,
                           sample_rows: int = 2048) -> dict[str, Any]:
    """一站式验布局：safetensors 直读 → 换算 → 抽样逐元素对拍 → 一层前向对拍，交可进 notes 的报告。

    白话：把上面几步串成一条命令，一趟交三样东西——产物里有哪几块（键名）、搬行的账对不对
    （逐元素）、两道做法算出来的结果差多少（前向对拍）。任何一环没对上，报告里就写着没对上，
    不拿"跑通了"三个字糊弄过去。
    """
    idx = read_index(snapshot_dir)
    cfg = text_config if text_config is not None else getattr(text_stack, "config")
    conv = convert_attention_layer(snapshot_dir, layer_idx, cfg, index=idx)
    base = f"{text_prefix(idx)}layers.{layer_idx}.self_attn."
    names = [base + n for n in ATTN_NAMES if base + n in idx["weight_map"]]
    raw = read_tensors(snapshot_dir, names, idx, dtype=torch.float32)
    short = {k[len(base):]: v for k, v in raw.items()}
    rows = sample_element_match(conv, short, sample_rows=sample_rows, seed=seed)
    parity = check_layer_parity(text_stack, conv, seq=seq, seed=seed, tol=tol,
                                eps=float(getattr(cfg, "rms_norm_eps", 1e-6)))
    lay = conv.layout
    return {
        "layer_idx": layer_idx, "keys": conv.keys(), "row_match": rows, "parity": parity,
        "ok": bool(rows["ok"] and parity.ok), "note": parity.as_note(), "notes": list(conv.notes),
        "layout": {"heads": lay.heads, "kv_heads": lay.kv_heads, "head_dim": lay.head_dim,
                   "rotary_dim": lay.rotary_dim, "group": lay.group},
    }
