"""p1-06 解码器域五场景测试（`-k config/roundtrip/causal/interface/mps` 对应 tasks 的 5 条孙任务）。

【做什么】把 `sys1/model.py` 的四件事钉死：① 尺寸真从 config 来（不是代码里写死的）；
② 存出去再读回来，同样的输入要吐出逐位一样的结果；③ 任何位置都读不到它后面的内容——
改了后面的字，前面的结果必须一个比特都不动；④ 对外只交“逐位置数字”和“独立输出层权重”，
决策程序拿这两样就够了，别的一概碰不到（MPS fp16 前向+反向作为设备门单列）。
【怎么做】config 用极小档（d=64/L=2/heads=4/vocab=211）秒级跑完；因果性用“改后不看”的属性
断言（`torch.equal` 逐比特），并配两个阳性对照——改位置 i 本身必须变、改最前面的字必须影响
末位，否则“掩码把大家都屏蔽了”这种反向错也能假过；接口门除形状与签名外，还静态扫
`sys1/decision/` 的源码（先把注释与字符串抹掉再扫，文档里举例不算越界），并且用构造样本
自证扫描器“抓得到真越界”；该目录正被别的域并发写入，目录空、文件半截都算无违规，绝不崩。
【为什么】被否方案一：拿手写期望值表对拍——随机权重一换就得重算表，脆且假；这里全部用
闭式性质（行和为一、逐比特不变、形状随 config 线性缩放）。被否方案二：因果性只断言
“误差小于 1e-6”——−inf 屏蔽的数学结论本就是“恒等于零”，留容差等于给未来泄漏开门。
被否方案三：MPS 用例不标 `@pytest.mark.mps` 直接跑——CI 的 linux runner 没有 MPS，主线全红。
被否方案四：静态门只“扫了不报错”——没有自证样本的门是假门，扫描器一旦被改坏就静默放行。
"""
from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from sys1.model import (
    CONFIG_FILE,
    DECISION_CONFIG_FILE,
    WEIGHTS_FILE,
    Decoder,
    ModelConfig,
    apply_rope,
    rope_tables,
)
from sys1.testing.torch_ref import rope_angle_tables, rope_ref

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiny_config(**overrides) -> ModelConfig:
    """极小档配置：CPU 上毫秒级建好，所有用例共用同一份“可控尺寸”。"""
    base = {"d": 64, "L": 2, "heads": 4, "ctx": 128, "vocab": 211, "rope_theta": 10_000.0, "seed": 7}
    base.update(overrides)
    return ModelConfig(**base)


def _tiny_model(**overrides) -> Decoder:
    return Decoder(_tiny_config(**overrides)).eval()


def _ids(batch: int = 2, seq: int = 16, seed: int = 0, vocab: int = 211) -> torch.Tensor:
    """定种子造编号：0 号是 <|pad|>，所以编号从 1 起，别把真空洞混进无掩码用例。"""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, vocab, (batch, seq), generator=g)


# ==================== A1 模型定义：超参一律来自 config ====================


def test_config_shapes_all_derive_from_config():
    """每个子模块的形状都该能由 config 算出来：改任一项形状跟着动，代码里没有第二处尺寸来源。"""
    cfg = _tiny_config(d=48, L=3, heads=4, vocab=131, ctx=64, ffn_mult=2, norm_eps=1e-3)
    model = Decoder(cfg)
    assert cfg.head_dim == 12
    assert model.config is cfg
    assert model.tok_emb.weight.shape == (131, 48)     # 行数 = vocab，宽度 = d
    assert len(model.blocks) == cfg.L                   # 深度 = L
    assert model.lm_head.weight.shape == (131, 48)      # 输出层：d 进、vocab 出
    assert model.lm_head.bias is None
    block = model.blocks[0]
    assert block.attn.qkv.weight.shape == (3 * 48, 48)  # 一条投影打包三个角色
    assert block.attn.proj.weight.shape == (48, 48)
    assert block.attn.scale == pytest.approx(1.0 / (12.0**0.5))  # 尺度 = 1/sqrt(head_dim)
    assert block.mlp.up.weight.shape == (2 * 48, 48)    # 前馈宽度 = d * ffn_mult
    assert block.mlp.down.weight.shape == (48, 2 * 48)
    for norm in (block.norm1, block.norm2, model.final_norm):
        assert norm.eps == cfg.norm_eps                 # 整形环节的小数也由 config 说话


def test_config_validation_rejects_indivisible_and_odd_geometry():
    """切不匀、凑不成双、深度为零、对照表没行、角度底数非正——几种病都必须在构造前拒掉。"""
    for kwargs in (
        {"d": 64, "heads": 6},   # 64 切不到 6 份
        {"d": 24, "heads": 8},   # head_dim=3 是单数，成对转不了
        {"L": 0},
        {"ctx": 0},
        {"vocab": 0},
        {"rope_theta": 0.0},
        {"ffn_mult": 0},
    ):
        bad = _tiny_config(**kwargs)
        with pytest.raises(ValueError):
            bad.validate()
        with pytest.raises(ValueError):
            Decoder(bad)  # 构造器自己也拦，不许绕过 validate 建模型


def test_config_rejects_sequence_longer_than_ctx():
    """位置表只有 ctx 行长：超长序列要在进网之前报错，而不是悄悄截断或除出 NaN。"""
    model = _tiny_model(ctx=16)
    with pytest.raises(ValueError, match="ctx"):
        model(_ids(seq=17))


def test_config_seed_makes_init_reproducible_and_does_not_leak():
    """seed 进 config 的意义：同 config 必得同权重；且建模型不许动调用方的全局随机状态。"""
    saved = torch.random.get_rng_state().clone()
    a, b = Decoder(_tiny_config(seed=42)), Decoder(_tiny_config(seed=42))
    c = Decoder(_tiny_config(seed=43))
    pa, pb, pc = a.state_dict().values(), b.state_dict().values(), c.state_dict().values()
    assert all(torch.equal(x, y) for x, y in zip(pa, pb, strict=True)), "同种子换了权重"
    assert not all(torch.equal(x, y) for x, y in zip(pa, pc, strict=True)), "换种子没换权重"
    assert torch.equal(torch.random.get_rng_state(), saved), "建模型改写了全局随机状态"


def test_config_rope_theta_flows_into_forward():
    """角度表与 p1-04 同一真源；旋转本体另用 fp64 闭式独立算一遍；theta 必须真进前向。

    口径说明（历史遗留，已解决）：`rope_ref` 曾在 **fp32 输入**下有一处就地别名缺陷（`.to(fp32)`
    自转换返回视图，边算边写致后半二次旋转），**现已修复**（先物化两半再写回），回归用例见
    `tests/test_torch_ref.py::test_rope_fp32_input_no_double_rotation_regression`（带阳性对照）。
    因此本用例对 `rope_ref` 的对拍在 fp16 与 fp32 双档进行：fp16 逐位相等，fp32 容差 1e-5
    （两实现的浮点累加顺序不同，容差比逐位更符合实际语义）。

    白话：给同一件事找两个互不相干的证人——一个是我们自己按公式手算的标准答案，另一个是
    内核对拍那边公认没错的那条路；两边都说对，才敢信这块齿轮装对了。
    """
    head_dim, seq, theta = 16, 12, 5_000.0
    mine = rope_tables(seq, head_dim, theta)
    ref = rope_angle_tables(seq, head_dim // 2, theta)  # 那边以 half 入参，式子同一条
    assert torch.equal(mine[0], ref[0]) and torch.equal(mine[1], ref[1]), "角度表有两处真源"
    torch.manual_seed(0)
    x = torch.randn(1, seq, 4, head_dim)
    half = head_dim // 2
    c = mine[0].unsqueeze(1).to(torch.float64)         # (seq,1,half)：头轴广播
    s = mine[1].unsqueeze(1).to(torch.float64)
    x1 = x[0, :, :, :half].to(torch.float64)
    x2 = x[0, :, :, half:].to(torch.float64)
    book = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)  # 独立手算的数学真值
    assert (apply_rope(x, *mine)[0].to(torch.float64) - book).abs().max().item() < 1e-6
    xh = x.to(torch.float16)  # 打包版参考的正确档位：q 进第一槽、k 进第二槽、第三槽不该动
    zeros = torch.zeros(seq, 4, head_dim, dtype=torch.float16)
    packed = torch.stack((xh[0], xh[0] * 0.5, zeros), dim=1)
    got = rope_ref(packed, *mine)
    assert torch.equal(got[:, 0], apply_rope(xh, *mine)[0]), "模型侧旋转与内核侧参考不一致"
    assert torch.equal(got[:, 2], packed[:, 2]), "第三个角色不该被转"
    # fp32 档（缺陷修复后恢复）：同一对拍，容差 1e-5（浮点顺序差异，非缺陷）。
    zeros32 = torch.zeros(seq, 4, head_dim, dtype=torch.float32)
    packed32 = torch.stack((x[0], x[0] * 0.5, zeros32), dim=1)
    got32 = rope_ref(packed32, *mine)
    assert (got32[:, 0] - apply_rope(x, *mine)[0]).abs().max().item() < 1e-5, "fp32 档对拍回规"
    assert torch.equal(got32[:, 2], packed32[:, 2]), "fp32 档第三槽也不该被转"
    ids = _ids()  # 同种子、只换 theta，逐位置结果必须不同——证明它没被硬编码吃掉
    lo = Decoder(_tiny_config(rope_theta=1_000.0)).eval()(ids)
    hi = Decoder(_tiny_config(rope_theta=100_000.0)).eval()(ids)
    assert not torch.equal(lo, hi), "rope_theta 没进前向（被硬编码吃了）"


def test_config_defaults_align_mac_smoke_profile():
    """默认档就是父 design D6 的 Mac 冒烟档（d=512/L=12/heads=8/ctx=1024），改档只改 config。"""
    cfg = ModelConfig()
    assert (cfg.d, cfg.L, cfg.heads, cfg.ctx) == (512, 12, 8, 1024)
    assert cfg.rope_theta == 10_000.0 and cfg.head_dim == 64
    assert cfg.vocab == 16_000, "词表行数须与 p1-03 实盘档一致（接口常数，改这里即可）"


# ==================== A2 save/load：模型目录三件套 ====================


def test_roundtrip_files_layout_on_disk(tmp_path):
    """落盘布局就是“模型目录”：config.json + weights.safetensors + decision_config.json 同目录。"""
    out_dir = _tiny_model().save(tmp_path / "ckpt")
    for name in (CONFIG_FILE, WEIGHTS_FILE, DECISION_CONFIG_FILE):
        assert (out_dir / name).is_file(), f"缺 {name}"
        assert (out_dir / name).stat().st_size > 0, f"{name} 是空文件"
    data = json.loads((out_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    for key in ("d", "L", "heads", "ctx", "vocab", "rope_theta", "seed"):
        assert key in data, f"config.json 少了超参 {key}"
    decision = json.loads((out_dir / DECISION_CONFIG_FILE).read_text(encoding="utf-8"))
    assert decision.get("placeholder") is True, "占位决策配置该自报占位"
    assert not (out_dir / "tokenizer").exists(), "tokenizer/ 归 S0 产物，模型 save 不造它"


def test_roundtrip_forward_bitwise_identical_after_load(tmp_path):
    """spec「配置往返」：同输入、eval 模式下存出去再读回来必须逐比特一致（不是“接近”就行）。"""
    cfg = _tiny_config(d=128, L=2, heads=8)
    model = Decoder(cfg).eval()
    ids = _ids(seq=24, vocab=cfg.vocab)
    before = model(ids)
    reloaded = Decoder.load(model.save(tmp_path / "ckpt"))
    assert reloaded.config == cfg, "config 往返丢字段（seed 与 rope_theta 都在这条里）"
    assert torch.equal(reloaded(ids), before), "权重往返后前向变了——这条落盘路径不该有损"
    assert next(reloaded.parameters()).dtype == torch.float32


def test_roundtrip_keeps_existing_decision_config(tmp_path):
    """decision_config.json 归决策域：已有内容时模型 save 只补缺，绝不覆盖别人家的东西。"""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    own = {"temperature": {"noul": 0.7}, "render_version": "v1"}
    (ckpt / DECISION_CONFIG_FILE).write_text(json.dumps(own), encoding="utf-8")
    _tiny_model().save(ckpt)
    assert json.loads((ckpt / DECISION_CONFIG_FILE).read_text(encoding="utf-8")) == own
    assert (ckpt / CONFIG_FILE).is_file() and (ckpt / WEIGHTS_FILE).is_file()


def test_roundtrip_rejects_unknown_config_key(tmp_path):
    """未知字段当面拒绝：悄悄收下等于“以为改了其实没改”，是最难查的一类错。"""
    ckpt = tmp_path / "ckpt"
    _tiny_model().save(ckpt)
    path = ckpt / CONFIG_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data["num_layers"] = 99  # 手滑写了个旧名字
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="未知字段"):
        Decoder.load(ckpt)


# ==================== A3 因果性属性 ====================


def test_causal_future_tokens_never_leak_into_prefix():
    """spec「因果性属性」：改位置 i 之后的所有 token，位置 ≤ i 的逐位置结果一个比特都不许动。

    断言用逐比特相等而不是 allclose：−inf 屏蔽的数学结论本就是“份额恒等于零”，留容差
    等于给未来泄漏开门（训练指标虚高、上线才崩，正是本项目最贵的事故）。
    """
    model = _tiny_model(L=3)
    ids = _ids(batch=3, seq=20, vocab=model.config.vocab)
    hidden = model(ids)
    for i in range(ids.shape[1] - 1):
        mutated = ids.clone()
        tail = mutated[:, i + 1:]
        mutated[:, i + 1:] = torch.randint_like(
            tail, 1, model.config.vocab, generator=torch.Generator().manual_seed(i)
        )
        got = model(mutated)
        assert torch.equal(got[:, : i + 1], hidden[:, : i + 1]), f"位置 {i} 之后的字影响了前缀"


def test_causal_change_at_position_i_moves_i_but_not_before():
    """阳性对照：只改第 i 位的字，第 i 位必须变（否则“谁都看不见”也能假过），前面不许变。"""
    model = _tiny_model(L=3)
    ids = _ids(batch=2, seq=18, vocab=model.config.vocab)
    hidden = model(ids)
    i = 9
    mutated = ids.clone()
    mutated[:, i] += 1
    got = model(mutated)
    assert torch.equal(got[:, :i], hidden[:, :i]), "改第 9 位却惊动了前 9 位：位置轴装反了"
    assert not torch.equal(got[:, i], hidden[:, i]), "改了字而该位纹丝不动：字压根没进计算"


def test_causal_first_token_propagates_to_last_position():
    """反方向的正面证据：改最前面那个字，末位结果必须跟着动（信息确实在一路往右传）。"""
    model = _tiny_model(L=3)
    ids = _ids(batch=2, seq=18, vocab=model.config.vocab)
    last = model(ids)[:, -1]
    mutated = ids.clone()
    mutated[:, 0] += 5
    assert not torch.equal(model(mutated)[:, -1], last), "首字影响不到末位：掩码大概是空的"


def test_causal_pad_keys_are_never_read():
    """右补洞做不了键：批内补洞不许改真实位置的结果（决策读出取末位的先决条件），也不许出 NaN。"""
    cfg = _tiny_config(ctx=64)
    model = Decoder(cfg).eval()
    real = _ids(batch=2, seq=20, vocab=cfg.vocab)
    bare = model(real)  # 不补洞
    pad = torch.full((2, 12), cfg.pad_token_id, dtype=real.dtype)
    padded = torch.cat((real, pad), dim=1)  # 右边补 12 个洞
    mask = torch.ones_like(padded)
    mask[:, real.shape[1]:] = 0
    wide = model(padded, mask)
    assert torch.isfinite(wide).all(), "补洞行整行被屏蔽会除出 NaN——对角自看漏了"
    diff = (wide[:, : real.shape[1]] - bare).abs().max().item()
    assert diff < 1e-5, f"批内补洞改了真实位置的结果（最大偏差 {diff}）"


# ==================== B1 窄接口固化（“换脑”契约） ====================

# decision/ 只准依赖这两样：forward 交出的逐位置数字，与 lm_head.weight；其余属性访问一律
# 视为越界。白名单只放“装载与设备搬运”类 torch 通用管线，不放结构名。
NARROW_SURFACE = {
    "forward", "lm_head", "config", "save", "load", "state_dict", "load_state_dict",
    "parameters", "named_parameters", "named_buffers", "to", "to_empty", "cpu", "cuda",
    "eval", "train", "half", "float", "double", "device", "dtype",
}
# 模型内部结构属性名：在 decision/ 源码里以 `.名字` 出现即违例。只挑独一无二的名字，通用词
# （head_dim/proj/attn 等）交给下面的“模型对象接收者”规则兜，免得误伤别域自己的同名属性。
INTERNAL_ATTRS = (
    "tok_emb", "final_norm", "qkv", "norm1", "norm2",
    "_visible_mask", "_share_rows", "_tables",
)
# 模型内部符号（结构类与角度工具）：被点取、被调用或被 import 都算越界。
INTERNAL_SYMBOLS = ("DecoderBlock", "CausalAttention", "MLP", "apply_rope", "rope_tables")
_MODEL_RECEIVER = re.compile(
    r"\b(?:self\s*\.\s*)?(?:model|backbone|decoder|net)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)"
)
_IMPORT_MODEL = re.compile(r"from\s+sys1\.model\s+import\s+([^\n(]+)")
_STRIPPED = re.compile(r'#[^\n]*|""".*?"""|"[^"\n]*"', re.DOTALL)


def _decision_sources() -> list[Path]:
    """要扫的决策域源码。目录没落地就返回空表——“无违规”自动成立，本测试绝不因缺目录而崩。"""
    root = REPO_ROOT / "sys1" / "decision"
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.py"))


def _code_only(text: str) -> str:
    """抹掉注释与字符串字面量，只留代码骨架（换成等量空行，报错行号仍对得上）。"""
    return _STRIPPED.sub(lambda m: chr(10) * m.group(0).count(chr(10)), text)


def _scan_decision(text: str, rel: str) -> list[str]:
    """把一份源码扫成违例清单：模型对象上的非窄接口属性、内部结构属性、内部符号的裸用与导入。"""
    found: list[str] = []
    text = _code_only(text)  # 注释/文档里“提了一句”不算越界，只判真代码

    def at(pos: int) -> str:
        return f"{rel}:{text[:pos].count(chr(10)) + 1}"

    for match in _MODEL_RECEIVER.finditer(text):
        attr = match.group(1)
        if attr not in NARROW_SURFACE:
            found.append(f"{at(match.start())} 通过模型对象引用了 .{attr}（窄接口之外）")
    for name in INTERNAL_ATTRS:
        for match in re.finditer(rf"\.{name}\b", text):
            found.append(f"{at(match.start())} 引用了模型内部属性 .{name}")
    for name in INTERNAL_SYMBOLS:
        pattern = rf"\.{name}\b|\b{name}\s*\(|^[ \t]*(?:from|import)[^\n]*\b{name}\b"
        for match in re.finditer(pattern, text, re.MULTILINE):
            found.append(f"{at(match.start())} 出现模型内部符号 {name}")
    for match in _IMPORT_MODEL.finditer(text):
        for piece in match.group(1).split(","):
            name = piece.strip()
            if name.startswith("_") or name in INTERNAL_SYMBOLS:
                found.append(f"{at(match.start())} 从 sys1.model 导入了内部符号 {name}")
    return found


def test_interface_grep_gate_catches_real_violations():
    """自证扫描器不是假门：代码里真越界必须被抓到，注释里的举例不能被抓到。

    静态门最怕两件事——抓不到（等于没设门）与乱抓（别的域写一句说明就被判红）。两头都用
    构造样本钉住：以后改扫描器，这两条会立刻反对。
    """
    bad = "def f(model):\n    return model.blocks[0].qkv\n"
    hits = _scan_decision(bad, "fake.py")
    assert any(".blocks" in h for h in hits), f"模型对象上摸 .blocks 没抓到：{hits}"
    assert any(".qkv" in h for h in hits), f"内部属性 .qkv 没抓到：{hits}"
    assert _scan_decision("from sys1.model import DecoderBlock\n", "fake.py"), "导入内部类没抓到"
    prose = "def f(model):\n    return model.forward(x)  # 别提 model.blocks 这种写法\n"
    assert _scan_decision(prose, "fake.py") == [], f"注释里的举例被误判成越界：{_scan_decision(prose, 'x')}"


def test_interface_grep_gate_is_safe_when_decision_absent_or_empty(tmp_path, monkeypatch):
    """并发安全：decision/ 还没落地、只有空 __init__.py、甚至半截文件，静态门都只能“无违规”不能崩。"""
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    assert _decision_sources() == [], "目录不存在时不该报错，也不该扫到别处的文件"
    empty = tmp_path / "sys1" / "decision"
    empty.mkdir(parents=True)
    (empty / "__init__.py").write_text("", encoding="utf-8")
    assert [p.name for p in _decision_sources()] == ["__init__.py"]
    assert not any(_scan_decision(p.read_text(encoding="utf-8"), p.name) for p in _decision_sources())
    half = empty / "readout.py"  # 别的域写到一半：语法不完整也只能算无违规，不许抛异常
    half.write_text("def f(model):\n    return model.", encoding="utf-8")
    assert _scan_decision(half.read_text(encoding="utf-8"), half.name) == []


def test_interface_forward_signature_and_returns_last_hidden():
    """前向出口只有一条：forward(input_ids, attn_mask=None) -> (B,T,d) 的逐位置数字。"""
    params = [p for p in inspect.signature(Decoder.forward).parameters.values() if p.name != "self"]
    assert [p.name for p in params] == ["input_ids", "attn_mask"], f"前向签名被改宽了：{params}"
    by_name = {p.name: p for p in params}
    assert by_name["input_ids"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert by_name["attn_mask"].default is None, "attn_mask 必须可省（不补洞时不用传）"
    model = _tiny_model()
    ids = _ids(batch=2, seq=16, vocab=model.config.vocab)
    hidden = model(ids, torch.ones_like(ids))
    assert isinstance(hidden, torch.Tensor) and hidden.dim() == 3
    assert hidden.shape == (2, 16, model.config.d)
    assert hidden.shape[-1] != model.config.vocab, "前向不该交整张对照表的分数"
    assert not hasattr(model, "generate"), "模型侧不许有任何生成循环（output_tokens=0 红线）"


def test_interface_lm_head_weight_selects_letter_rows():
    """换脑契约的另一半：输出层权重是独立模块的 .weight，外部按行取字母行就能读出选项分数。"""
    cfg = _tiny_config(vocab=211)
    model = Decoder(cfg).eval()
    weight = model.lm_head.weight
    assert isinstance(weight, torch.nn.Parameter) and weight.shape == (cfg.vocab, cfg.d)
    letters = torch.arange(1, 27)  # 26 个字母行（真编号由 S0 词表给，这里只验“可按行取”）
    rows = weight.index_select(0, letters)
    assert rows.shape == (26, cfg.d)
    assert torch.equal(rows, weight[letters]), "按行取出来的不是那几行本身"
    hidden = model(_ids(batch=2, seq=12, vocab=cfg.vocab))[:, -1]
    scores = hidden @ rows.transpose(0, 1)  # 末位数字 × 字母行 = 选项分数（决策读出的全部依赖）
    assert scores.shape == (2, 26) and torch.isfinite(scores).all()
    probs = F.softmax(scores.float(), dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_interface_decision_program_touches_only_narrow_surface():
    """静态门：`sys1/decision/` 里不许引用模型内部（除逐位置数字与 lm_head.weight）。"""
    violations: list[str] = []
    for path in _decision_sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        violations += _scan_decision(text, str(path.relative_to(REPO_ROOT)))
    assert not violations, "窄接口被越界：\n  " + "\n  ".join(violations)


def test_interface_package_import_stays_torch_free():
    """他域约定：`import sys1` 不许把 torch 拖进来（模型只在自己的模块里依赖 torch）。"""
    code = "import sys, sys1; sys.exit(1 if 'torch' in sys.modules else 0)"
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"import sys1 把 torch 一起拖进来了：{proc.stderr.strip()}"


# ==================== B2 MPS fp16 前向反向 ====================


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="本机无 MPS 设备")
def test_mps_fp16_forward_backward_batch4_seq256():
    """spec「MPS 前向反向」：batch=4、seq=256、fp16 跑一次前向+反向，不许异常、分数不许非有限。

    分数与份额在网内以 fp32 累加（model.py 的口径），所以 fp16 位宽下末位分数仍应有界；
    损失刻意用 fp32 的分数来算——fp16 直算整表分数是常见的溢出源。
    """
    device = torch.device("mps")
    cfg = _tiny_config(d=256, L=4, heads=8, ctx=1024, vocab=4096, seed=0)
    model = Decoder(cfg).to(device, torch.float16)
    model.train()  # 按训练态跑反向（本模型无随机层，态只影响语义读法）
    ids = torch.randint(1, cfg.vocab, (4, 256), device=device)
    mask = torch.ones_like(ids)
    hidden = model(ids, mask)
    assert hidden.shape == (4, 256, cfg.d) and hidden.dtype == torch.float16
    logits = model.lm_head(hidden[:, :-1]).float()  # 窄接口自行拼出“下一位”的分数
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab), ids[:, 1:].reshape(-1))
    assert torch.isfinite(loss), f"fp16 反向冒烟出现非有限损失：{loss.item()}"
    loss.backward()
    grads = dict(model.named_parameters())
    for name in ("tok_emb.weight", "blocks.0.attn.qkv.weight", "blocks.3.mlp.down.weight",
                 "blocks.3.norm2.weight", "lm_head.weight"):
        grad = grads[name].grad
        assert grad is not None, f"{name} 没收到梯度"
        assert torch.isfinite(grad).all(), f"{name} 梯度含 NaN/Inf"
        assert grad.abs().sum().item() > 0, f"{name} 梯度恒为零，链路断了"
