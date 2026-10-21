"""p2-01 权重布局层测试：HF 检查点换成自研栈张量布局——先合成身板验纯换算，再用真权重对拍。

口径：本层的换算只做"搬行、换座次"，一个数都不改，所以逐元素断言不必依赖真权重；
名字带 real 的用例走 bench/ms_models 的真快照与真检查点（缺则 skip，不假绿）；
资源纪律：一切前向在 CPU 上做（MPS 归一阶段训练长跑占用）。
"""
from __future__ import annotations

import json
import pytest
import torch

from production import assets
from production.backbone import layout as L
from production.backbone import loader as LD

SNAP = assets.DEFAULT_CACHE_DIR / "models" / "Qwen--Qwen3.5-0.8B" / "snapshots" / "master"
HAS_SNAP = SNAP.is_dir() and bool(list(SNAP.glob("*.safetensors")))
real_only = pytest.mark.skipif(not HAS_SNAP, reason="真快照不在 bench/ms_models：布局对拍要真权重")

#: 与 Qwen3.5-0.8B 实证同族的一副小身板：8 个大头共用 2 套小头，每头 256 宽、其中 64 宽参与转角
LAY = L.HeadLayout(heads=8, kv_heads=2, head_dim=256, rotary_dim=64)


@pytest.fixture(scope="session")
def text_cfg():
    """文本侧 config：0.8B 外面套着多模态壳，布局口径只看里面的语言栈。"""
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(str(SNAP)).get_text_config()


@pytest.fixture(scope="session")
def bb():
    """真载一次 0.8B（CPU/fp16）给两处真权重用例共用——单次载入十几秒，不做两遍。"""
    return assets.load_backbone("qwen3.5-0.8b", device="cpu", dtype=torch.float16)


def test_layout_permutation_is_bijection():
    """座次置换必须是双射：每个来源坐标被搬到唯一一个自研栈位置上，不多不少。"""
    perm = L.rope_pair_permutation(LAY)
    assert len(perm) == LAY.head_dim
    assert sorted(perm) == list(range(LAY.head_dim))
    assert perm[:4] == [0, 1, 2, 3]                      # 真转角的前半原样坐在前排
    assert perm[LAY.half:LAY.half + 4] == [32, 33, 34, 35]   # 另一半取自来源 rotary_dim/2 起的第 32..35 位


def test_layout_permutation_pairs_follow_kernel_convention():
    """配对要换成内核那一种：自研栈第 j 位与第 j+half 位，正是来源的第 j 位与第 j+rotary_dim/2 位。"""
    perm = L.rope_pair_permutation(LAY)
    r_half = LAY.rotary_dim // 2
    for j in range(r_half):
        assert perm[LAY.half + j] - perm[j] == r_half
    assert sorted(perm[:LAY.rotary_dim // 2] + perm[LAY.half:LAY.half + r_half]) == list(range(LAY.rotary_dim))


def test_layout_permutation_rejects_odd_rotary():
    """转角宽度出格（奇数、超过头宽、未转的剩奇数）必须当场拒，不交出半套座次。"""
    for bad in (65, 0, 512):
        with pytest.raises(ValueError):
            L.rope_pair_permutation(L.HeadLayout(heads=8, kv_heads=2, head_dim=256, rotary_dim=bad))


def test_layout_rope_tables_pad_with_identity():
    """角表补到整头：真角留在前面，不转角的那些位配成 cos=1/sin=0（转了也等于没转）。"""
    seq = 5
    cos = torch.full((seq, LAY.rotary_dim), 0.5)
    sin = torch.full((seq, LAY.rotary_dim), -0.25)
    c, s = L.expand_rope_tables(cos, sin, LAY)
    r_half = LAY.rotary_dim // 2
    assert c.shape == (seq, LAY.half) and s.shape == (seq, LAY.half)
    assert torch.equal(c[:, :r_half], cos[:, :r_half]) and torch.equal(s[:, :r_half], sin[:, :r_half])
    assert torch.equal(c[:, r_half:], torch.ones(seq, LAY.half - r_half))
    assert torch.equal(s[:, r_half:], torch.zeros(seq, LAY.half - r_half))


def test_layout_rope_tables_reject_short_and_mismatched():
    """角表太短（连真转角那半都凑不齐）或 cos/sin 末维不等，一律拒收，不拿缺口表往下游喂。"""
    with pytest.raises(ValueError):
        L.expand_rope_tables(torch.zeros(4, 16), torch.zeros(4, 16), LAY)      # 短于 rotary_dim/2=32
    with pytest.raises(ValueError):
        L.expand_rope_tables(torch.zeros(4, 32), torch.zeros(4, 64), LAY)      # 两表末维不等


def test_layout_gated_split_roundtrip():
    """拆门必须能原样装回去：每头前一段是查询、后一段是门，拆完再拼回应当一字不差。"""
    kdim = 6
    q = torch.arange(LAY.heads * 2 * LAY.head_dim * kdim, dtype=torch.float32)
    q = q.reshape(LAY.heads * 2 * LAY.head_dim, kdim)
    pure, gate = L.split_gated_q(q, LAY)
    assert pure.shape == gate.shape == (LAY.heads * LAY.head_dim, kdim)
    rebuilt = torch.stack([pure.view(LAY.heads, LAY.head_dim, kdim),
                           gate.view(LAY.heads, LAY.head_dim, kdim)], dim=1).reshape(-1, kdim)
    assert torch.equal(rebuilt, q)


def test_layout_gated_split_rejects_ungated_rows():
    """行数不是 heads×head_dim×2 就说明这层压根没开门：当场拒，不拿半截口径往下算。"""
    kdim = 4
    plain = torch.zeros(LAY.heads * LAY.head_dim, kdim)
    with pytest.raises(ValueError):
        L.split_gated_q(plain, LAY)
    with pytest.raises(ValueError):
        L.split_gated_q(torch.zeros(LAY.heads * LAY.head_dim * 2 + 1, kdim), LAY)


def test_layout_pack_qkv_row_provenance():
    """打包后的每一行都要能追溯到来源那一行：查询/键换座次、值只补小头，三段分别核对。"""
    kdim, hd, block = 3, LAY.head_dim, LAY.heads * LAY.head_dim
    q_src = torch.arange(block * kdim, dtype=torch.float32).reshape(block, kdim) / 7.0
    k_src = torch.arange(LAY.kv_heads * hd * kdim, dtype=torch.float32).reshape(LAY.kv_heads * hd, kdim) / 11.0
    v_src = torch.arange(LAY.kv_heads * hd * kdim, dtype=torch.float32).reshape(LAY.kv_heads * hd, kdim) / 13.0
    packed = L.pack_qkv(q_src, k_src, v_src, LAY)
    assert packed.shape == (3 * block, kdim)
    q_i = L.rotated_row_index(LAY.heads, LAY, LAY.heads)
    k_i = L.rotated_row_index(LAY.heads, LAY, LAY.kv_heads)
    v_i = L.plain_row_index(LAY.heads, LAY, LAY.kv_heads)
    assert torch.equal(packed[:block], q_src.index_select(0, q_i))
    assert torch.equal(packed[block:2 * block], k_src.index_select(0, k_i))
    assert torch.equal(packed[2 * block:], v_src.index_select(0, v_i))
    perm = torch.tensor(L.rope_pair_permutation(LAY), dtype=torch.long)
    assert (k_i.reshape(LAY.heads, hd)[:4] == perm).all()             # 前四个大头共用第 0 套小头
    assert (k_i.reshape(LAY.heads, hd)[4:] == hd + perm).all()        # 后四个共用第 1 套（连着分组）
    assert torch.equal(v_i.reshape(LAY.heads, hd)[4], torch.arange(hd) + hd)   # 值一头都不转，照原座次
    with pytest.raises(ValueError):
        L.pack_qkv(q_src, k_src, v_src[:hd], LAY)                        # 键/值形状不等


def test_layout_norm_permute_is_bijection_per_head():
    """逐头缩放系数重排只许换座次：每个头内部是一份重排，系数总集合一个不多一个不少。"""
    w = torch.arange(2 * LAY.head_dim, dtype=torch.float32)
    moved = L.permute_norm(w, LAY)
    assert moved.shape == w.shape
    assert sorted(moved.tolist()) == sorted(w.tolist())
    perm = torch.tensor(L.rope_pair_permutation(LAY), dtype=torch.long)
    for h in range(2):
        head = w[h * LAY.head_dim:(h + 1) * LAY.head_dim]
        assert torch.equal(moved[h * LAY.head_dim:(h + 1) * LAY.head_dim], head[perm])


def test_layout_norm_permute_commutes_with_rms_scale():
    """换座次与逐坐标缩放可以调换次序：先重排系数再乘，与乘完再重排结果，逐位相等。"""
    rows = torch.randn(3, LAY.head_dim)
    w = torch.linspace(-0.4, 0.9, LAY.head_dim)
    perm = torch.tensor(L.rope_pair_permutation(LAY), dtype=torch.long)
    rms = (rows ** 2).mean(dim=1, keepdim=True).sqrt()
    left = L.permute_norm(w, LAY) * rms
    right = (w * rms).index_select(1, perm)
    assert torch.allclose(left, right, atol=1e-6)


def test_layout_fold_unit_shift_adds_one_keeps_shape():
    """折算 (1+w)：折完的数比原读数大整 1，形状一位没动（融合件只认"读数直接乘"）。"""
    w = torch.linspace(-0.3, 0.7, LAY.head_dim)
    folded = L.fold_unit_shift(w)
    assert folded.shape == w.shape
    assert torch.allclose(folded - w, torch.ones_like(w), atol=0)


def test_layout_bias_pack_all_or_none():
    """三条偏置要么全有要么全无：全缺交回 None，缺一条当场拒——上层不用猜"是全零还是没有"。"""
    block, kd = LAY.heads * LAY.head_dim, 4
    small = LAY.kv_heads * LAY.head_dim
    qb, kb, vb = (torch.zeros(block, kd), torch.zeros(small, kd), torch.zeros(small, kd))
    assert L.attention_bias_pack(None, None, None, LAY) is None
    with pytest.raises(ValueError):
        L.attention_bias_pack(qb, None, vb, LAY)
    packed = L.attention_bias_pack(qb + 1.0, kb + 2.0, vb + 3.0, LAY)
    assert packed.shape == (3 * block, kd)
    assert float(packed[:block].sum()) == float(block * kd)              # q 段全是 1
    assert packed[2 * block:].eq(3.0).all() and packed[block:2 * block].eq(2.0).all()


def test_loader_text_prefix_probe():
    """页码目录的探前缀：多模态那本把正文记在语言子模型名下，纯文本那本直接记在模型名下。"""
    multi = {"weight_map": {"model.language_model.layers.0.self_attn.q_proj.weight": "a.safetensors"}}
    assert LD.text_prefix(multi) == "model.language_model."
    plain = {"weight_map": {"model.layers.0.self_attn.q_proj.weight": "a.safetensors"}}
    assert LD.text_prefix(plain) == "model."
    with pytest.raises(KeyError):
        LD.text_prefix({"weight_map": {}})                       # 空目录：不猜，当场说
    with pytest.raises(KeyError):
        LD.text_prefix({"weight_map": {"visual.patch_embed.weight": "a.safetensors"}})   # 没有文本层


def test_loader_read_index_roundtrip(tmp_path):
    """页码目录写得进去就读得出来；没有目录时如实交空表，绝不猜别处的页码。"""
    payload = {"weight_map": {"model.layers.0.self_attn.q_proj.weight": "x.safetensors"},
               "metadata": {"total_size": 123}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(payload), encoding="utf-8")
    got = LD.read_index(tmp_path)
    assert got["weight_map"] == payload["weight_map"]
    assert LD.read_index(tmp_path / "nowhere")["weight_map"] == {}
    with pytest.raises(KeyError):
        LD.read_tensors(tmp_path, ["model.layers.0.no_such.weight"], got)   # 点不到名字就报缺页


@real_only
def test_layout_real_conversion_keys_and_row_match(text_cfg):
    """真检查点换算：产物键名齐、形状跟 config、逐元素对拍过，无偏置就如实缺席。"""
    idx = LD.read_index(SNAP)
    conv = LD.convert_attention_layer(SNAP, 3, text_cfg, index=idx)
    assert conv.keys() == ["gate.weight", "k_norm.weight", "o_proj.weight", "q_norm.weight", "qkv.weight"]
    lay = conv.layout
    assert (lay.heads, lay.kv_heads, lay.head_dim, lay.rotary_dim) == (8, 2, 256, 64)
    assert conv.tensors["qkv.weight"].shape == (3 * lay.heads * lay.head_dim, text_cfg.hidden_size)
    assert conv.gated and "qkv.bias" not in conv.tensors       # 开了门、无偏置：键缺席而非全零
    base = f"{LD.text_prefix(idx)}layers.3.self_attn."
    names = [base + n for n in LD.ATTN_NAMES if base + n in idx["weight_map"]]
    raw = LD.read_tensors(SNAP, names, idx, dtype=torch.float32)
    rep = LD.sample_element_match(conv, {k[len(base):]: v for k, v in raw.items()}, sample_rows=512)
    assert rep["ok"], rep["mismatch"]
    assert rep["sampled_rows"] == 512


@real_only
def test_layout_real_forward_parity_vs_hf(bb, text_cfg):
    """整层前向对拍 HF：按 config 挑出整注意力层，自研栈那一路与参考侧的相对差进 1e-3 才算接缝。"""
    full = LD.full_attention_layers(text_cfg)
    assert full[:2] == [3, 7]                     # 0/1/2 是线性注意力，3 起每 4 层一副真注意力
    worst = 0.0
    for layer_idx in full[:2]:
        rep = LD.verify_attention_layer(bb.snapshot.path, layer_idx, bb.body, text_config=text_cfg,
                                        seq=8, tol=1e-3, sample_rows=512)
        assert rep["row_match"]["ok"], rep["row_match"]["mismatch"]
        assert rep["ok"], rep["note"]
        worst = max(worst, rep["parity"].rel_diff)
    assert worst <= 1e-3
