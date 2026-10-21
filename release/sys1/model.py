"""GPT 式因果 decoder：全部超参来自 config，对外只交“逐位置数字 + 独立输出层权重”。

【做什么】接过一串编号（每个编号代表文本里的一个小片段，由 S0 自训的对照表给出），让每个位置
只跟“自己和它左边”的那些片段相互参照，最后给每个位置留下一串可学的数字（长度 d）。整份文件
里没有任何“一个片段接一个片段往外蹦”的生成循环：一次把所有位置并行算完，答案由外面的决策程序
从末位那串数字里读出来。
【怎么做】① 编号先查成 d 维数字串（tok_emb）；② 叠 L 个 pre-norm 块，每块做两件事：先整形再
“只看左边”的加权汇总（残差加回），再先整形再过一遍“升宽—关卡—压回”的前馈（残差加回）；
③ 加权汇总之前给查询与键按位置转个角（RoPE，角度来自 config.rope_theta，只跟“隔多远”有关），
分数与份额一律 fp32 累加、算完才降回工作位宽；④ 末了再整一次形，交出 (批量, 位置, d)；
⑤ 输出层 lm_head 是一个独立的矩阵，谁想读“某个字接在这一位后面自然不自然”，直接按行去它里面
取（decision/ 取的就是 26 个字母那几行）——本文件的 forward 绝不把它跑成整张对照表的分数。
【为什么】被否方案一：post-norm（先加再整形）——浅的能跑，12 层深链一上来就抖，pre-norm 把残差
主干拉成近乎直线才稳（父 design D6 的 L=12 档必须这个）。被否方案二：绝对位置 embedding——
ctx 一超过训练长度就得换表，二阶段换长上下文 backbone 时位置口径全变；RoPE 是相对角，长文同向，
而且与 p1-04 已对拍过的 rope 内核共用同一套角度式。被否方案三：forward 直接返回整表分数——
看着少一次调用，但决策侧就分不清“全分数”与“只取字母行”，换 backbone 时接口必崩，故只交
last_hidden。被否方案四：把超参写在类里图省事——同一arch 不同尺寸就没法复现，seed 也不入配置，
评测数字失去了可追溯性。
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

# ------------------------------------------------------------ 落盘布局（名字即约定）
# p1-09/p1-10 按“模型目录”消费这三件，不感知权重是自训还是载入的。
CONFIG_FILE = "config.json"
WEIGHTS_FILE = "weights.safetensors"
DECISION_CONFIG_FILE = "decision_config.json"
# tokenizer/ 目录由 S0 产物放入（p1-03），模型 save 既不产生也不校验它——避免把词表域拖进本域。
TOKENIZER_DIRNAME = "tokenizer"
# 占位决策配置：本域不解释内容，只保证“模型目录三件套齐活”；已存在则绝不覆盖（p1-07 拥有它）。
DECISION_CONFIG_PLACEHOLDER: dict[str, Any] = {
    "placeholder": True,
    "owner": "p1-07-decision-program",
    "note": "模型 save 只占位；真实读出配置（qtype 温度表等）由决策域写入，占位不覆盖已有文件",
}

# ------------------------------------------------------------ 数值口径（与 p1-04 对齐）
NEG_INF = float("-inf")           # 屏蔽位用 −inf 而不是大负数：份额才能“恒等于零”可断言
SOFTMAX_DENOM_FLOOR = 1e-30       # 防御性兜底：因果口径下不该出现全屏蔽行，出现也不至 NaN


@dataclass
class ModelConfig:
    """模型的全部超参，一处定义、处处生效（代码里不许再出现第二个尺寸来源）。

    Mac 冒烟档（父 design D6，~40M 参数）就是下面这组默认值：d=512 / L=12 / heads=8 /
    ctx=1024。vocab 默认 16000 取自 p1-03 实盘训练档（runs/1003-s0-bpe-16k-realedu-zh-en），
    那是**接口常数**依赖——本文件不 import 词表域，改词表大小只需改这里的一个数字。
    模型本身规模无关：d=64/L=1 跑测试、d=512/L=12 跑训练，走的是同一份代码。
    """

    d: int = 512                      # 每个位置的数字串长度（特征宽）
    L: int = 12                       # 块数（深度）
    heads: int = 8                    # 每块的份数
    ctx: int = 1024                   # 最长序列（位置表长度，超过即拒绝）
    vocab: int = 16_000               # 对照表行数（p1-03 实盘档）
    rope_theta: float = 10_000.0      # RoPE 角度底数（越大转得越慢、能分辨的跨度越长）
    seed: int = 0                     # 初始化种子：同 config 必得同权重（可复现性的模型侧镜像）
    ffn_mult: int = 4                 # 前馈升宽倍数
    norm_eps: float = 1e-5            # 整形环节的兜底小数（与 p1-04 add_ln 的 DEFAULT_EPS 同值）
    pad_token_id: int = 0             # <|pad|> 固定 0 号（p1-03 契约），本域只用于报错提示

    @property
    def head_dim(self) -> int:
        """每份的数字串长度 = d // heads（validate 已保证整除且为双数，RoPE 要成对转）。

        白话：总宽按份数平摊，每份摊到多长就是它；分不匀或凑不成双，前面就报错，绝不带病切。
        """
        return self.d // self.heads

    def to_dict(self) -> dict[str, Any]:
        """摊成可 json 化的普通字典（config.json 的唯一写入口）。

        白话：把这份设置单变成一份普通字典，落盘那步只管照着写字节，不用再回头懂模型内部长什么样。
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, strict: bool = True) -> ModelConfig:
        """从 config.json 的字典还原配置；缺键按默认值补，strict=True 时未知键直接报错。

        白话：把外面那份设置单照单认领，认不出的名字一律当面拒绝而不是悄悄收下——悄悄收下意味着
        “你以为改了、其实没改”，是最难查的一类错；缺的那几项才用默认值补上。
        """
        known = {f.name for f in fields(cls)}
        if strict:
            unknown = sorted(set(data) - known)
            if unknown:
                raise ValueError(f"{CONFIG_FILE} 含未知字段 {unknown}；加字段请同步改 ModelConfig")
        return cls(**{k: v for k, v in data.items() if k in known})

    def validate(self) -> None:
        """构造模型前先过一遍尺寸自洽检查，把“算出鬼结果”提前成一条人话报错。

        白话：开工前把尺子对三遍——每份要能把总宽整除、每对的长度得是双数（旋转是成对转的）、
        序列不能长过位置表、对照表至少得有一行；有一条不对就当场拒绝，绝不带病开工。
        """
        if self.d <= 0 or self.L <= 0 or self.heads <= 0 or self.ctx <= 0 or self.vocab <= 0:
            raise ValueError(
                f"d/L/heads/ctx/vocab 必须为正，实得 d={self.d} L={self.L} "
                f"heads={self.heads} ctx={self.ctx} vocab={self.vocab}"
            )
        if self.d % self.heads:
            raise ValueError(f"d={self.d} 不能被 heads={self.heads} 整除，切份会丢余数")
        if self.head_dim % 2:
            raise ValueError(f"head_dim={self.head_dim} 需为双数（RoPE 按对旋转）；请调 d 或 heads")
        if self.ffn_mult <= 0:
            raise ValueError(f"ffn_mult 必须为正，实得 {self.ffn_mult}")
        if self.rope_theta <= 0.0:
            raise ValueError(f"rope_theta 必须为正（角度底数），实得 {self.rope_theta}")


def rope_tables(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 (seq_len, head_dim//2) 的 cos/sin 两张 fp32 角度表：第 pos 行第 i 列 = cos/sin(pos·freq_i)。

    freq_i = theta^(-2i/head_dim)，i ∈ [0, head_dim//2)。与 p1-04 `rope_angle_tables` 同一式子
    （那边以 half=head_dim//2 入参、这边以 head_dim 入参），所以“模型侧角度”与“内核对拍侧角度”
    永远只有一份真源；分表存、fp32 算，理由同 p1-04：角度在现场算会把超越函数塞进内核，
    查表把它挪到 torch 侧最稳。

    白话：先给每个位置、每一对数字算好要转多少度，横着的余弦和纵着的正弦各抄一张小抄；
    后面每一层都只翻这两张小抄，不在现场重新算——一次算清，处处一致。
    """
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f"head_dim 需为正双数才能成对旋转，实得 {head_dim}")
    half = head_dim // 2
    freq = torch.tensor(
        [theta ** (-2.0 * i / head_dim) for i in range(half)], dtype=torch.float32, device=device
    )
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    angle = pos * freq.unsqueeze(0)
    return torch.cos(angle), torch.sin(angle)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 (B, T, heads, head_dim) 做 rotate-half 旋转；头轴共用同一套角度，值轴不在这里出现。

    cos/sin 为 (T, head_dim//2) 的 fp32 表。旋转在 fp32 里做完再降回 x 的位宽，与 p1-04
    `rope_ref` 严格同口径（那边是 (tokens, 3, heads, dim) 打包版，本函数是拆包后的单份版，
    两者可用同一组 cos/sin 逐元素对上——tests/test_model.py 的角度对拍用例就钉这一点）。

    白话：把每串数字从中间剪成两截，前截减去“后截乘纵向小抄”，后截加上“前截乘横向小抄”；
    位置越靠后转得越多，于是两串一对照，剩下的信息只跟“隔多远”有关，跟它在第几位无关。
    """
    half = x.size(-1) // 2
    c = cos.to(torch.float32).unsqueeze(0).unsqueeze(2)  # (T,half) → (1,T,1,half)：批轴、头轴广播
    s = sin.to(torch.float32).unsqueeze(0).unsqueeze(2)
    x1 = x[..., :half].to(torch.float32)
    x2 = x[..., half:].to(torch.float32)
    out = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
    return out.to(x.dtype)


def _visible_mask(
    seq_len: int, attn_mask: torch.Tensor | None, batch: int, device: torch.device | str | None = None
) -> torch.Tensor:
    """返回 (B,1,T,T) 或 (1,1,T,T) 布尔掩码，"真 = 这一行允许看这一列"。

    三条规矩叠在一起：① 因果（j <= i，绝不看未来）；② 屏蔽位不做键（右 padding 的空洞位
    谁也看不见，保证批内补洞不改已算出的位置）；③ 对角线自看（洞位至少能看自己，避免整行
    全屏蔽除出 NaN；真实位 i 的 j==i 本来就可看，所以这条对真实行的结果零影响）。
    """
    idx = torch.arange(seq_len, device=device)
    causal = (idx.unsqueeze(1) - idx.unsqueeze(0)) >= 0  # [i,j] = (i - j >= 0)
    if attn_mask is None:
        return causal[None, None]
    if attn_mask.shape != (batch, seq_len):
        raise ValueError(f"attn_mask 需为 (B,T)=({batch},{seq_len})，实得 {tuple(attn_mask.shape)}")
    key_ok = attn_mask.to(torch.bool)[:, None, None, :]              # (B,1,1,T)：这一列是不是真 token
    eye = torch.eye(seq_len, dtype=torch.bool, device=device)[None, None]           # (1,1,T,T)：对角自看
    return causal[None, None] & (key_ok | eye)


def _share_rows(scores: torch.Tensor) -> torch.Tensor:
    """沿键轴摊份额：先减每行最大值再取指数、按行和归一（减最大值只为不溢出）。"""
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=-1, keepdim=True).clamp_min(SOFTMAX_DENOM_FLOOR)


class CausalAttention(nn.Module):
    """“只看左边”的加权汇总：一条 qkv 投影 + 一条回投，份数与每份宽度来自 config。"""

    def __init__(self, config: ModelConfig, ops=None) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(config.head_dim)  # 点积尺度：除以 sqrt(每份宽度)
        self.qkv = nn.Linear(config.d, 3 * config.d)
        self.proj = nn.Linear(config.d, config.d)
        self.ops = ops   # 非 None 时前向走可微内核链（autograd 模块门面），形状/设备不合规逐算子回退

    def _kernel_ok(self, x: torch.Tensor, seq: int, visible) -> bool:
        """内核链准入门：无批内补洞掩码、序列与份宽可被 16 整除、设备有内核可用。"""
        if self.ops is None or visible is not None or seq % 16 or self.head_dim % 16:
            return False
        return self.ops.kernels_ready(x.device)

    def _forward_kernels(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """可微内核四连：qkv→packed-rope→滑窗注意力→proj；主干精度纪律由调用方的融合件守住。

        白话：四段工都交给同一套模具——投影先把每行数字折成三摞；前两摞按位置转角；
        然后每个位置只朝后看定长一段把内容加权合拢；最后折回一行宽交出去。中途把数字
        改细（半精度）省力气，两头粗细的换算由自动回账链条接上。
        """
        ops = self.ops
        b, t, d = x.shape
        xh = x.reshape(b * t, d).half()
        qkv = ops.linear(xh, self.qkv.weight.half(), self.qkv.bias.float())
        qkv = qkv.view(b, t, 3, self.heads, self.head_dim).reshape(b * t, 3, self.heads, self.head_dim).contiguous()
        # 内核要 (tokens, half) 的逐位置表：批内同位共享同一行角度，按批展开再续平
        c2 = cos.unsqueeze(0).expand(b, t, cos.size(-1)).reshape(b * t, -1).contiguous().half()
        s2 = sin.unsqueeze(0).expand(b, t, sin.size(-1)).reshape(b * t, -1).contiguous().half()
        qkv = ops.rope(qkv, c2, s2).view(b, t, 3, self.heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2).contiguous()
        k = qkv[:, :, 1].transpose(1, 2).contiguous()
        v = qkv[:, :, 2].transpose(1, 2).contiguous()
        out = ops.attn_sw(q, k, v, window=t)                  # 整窗=全因果（无 pad 场景专用）
        out = out.transpose(1, 2).reshape(b * t, d).contiguous()
        return ops.linear(out, self.proj.weight.half(), self.proj.bias.float()).view(b, t, d)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        visible: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(B,T,d) 进、(B,T,d) 出；cos/sin 是角度表，visible 是 (B,1,T,T) 或 (1,1,T,T) 的可见位。

        分数与份额全程 fp32 累加、只在出口降回工作位宽——与 p1-04 的 gemm/attn 参考同口径，
        将来这两条投影换成 fused 内核时，eager 与内核才有对得上的前提。

        白话：每个位置向它看得见的那些位置报一组相关性分数，把分数摊成“加起来等于一”的份额，
        再按份额把第三摞数字加权合起来；看不见的位置份额恒为零，不是“很小”，是零。
        """
        batch, seq, _ = x.shape
        if self._kernel_ok(x, seq, visible):
            return self._forward_kernels(x, cos, sin).to(x.dtype)
        # 一次投影切成三份：查询、键各按位置转角，值原封不动（p1-04 rope 的“只转前两槽”纪律）
        qkv = self.qkv(x).view(batch, seq, 3, self.heads, self.head_dim)
        q = apply_rope(qkv[:, :, 0], cos, sin)
        k = apply_rope(qkv[:, :, 1], cos, sin)
        v = qkv[:, :, 2]
        q32 = q.transpose(1, 2).to(torch.float32)          # (B,heads,T,head_dim)
        k32 = k.transpose(1, 2).to(torch.float32)
        v32 = v.transpose(1, 2).to(torch.float32)
        scores = (q32 @ k32.transpose(-1, -2)) * self.scale  # (B,heads,T,T) fp32 累加
        if visible is not None:
            scores = scores.masked_fill(~visible, NEG_INF)
        out = _share_rows(scores) @ v32                      # 屏蔽位份额恰为 0 → 未来严格不泄漏
        out = out.transpose(1, 2).reshape(batch, seq, self.heads * self.head_dim)
        return self.proj(out.to(x.dtype))


class MLP(nn.Module):
    """位置无关的前馈：升宽 → gelu（tanh 近似）→ 压回，进出同宽。"""

    def __init__(self, config: ModelConfig, ops=None) -> None:
        super().__init__()
        wide = config.d * config.ffn_mult
        self.up = nn.Linear(config.d, wide)
        self.down = nn.Linear(wide, config.d)
        self.ops = ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """逐位置独立地过一次“升宽—关卡—压回”，形状不变、绝不串到别的位置上去。

        gelu 取 tanh 近似式（不是精确误差函数式），与 p1-04 `gemm_ref` 的 gelu 定义逐字一致。
        内核链只接管两条投影（act=none 口径），gelu 留在普通写法——融合激活的反向未承接，不偷。

        白话：先把每一串数字摊宽好几倍，过一道“负得多的压平、正的留原样但带点弯”的关卡，
        再压回原来的宽度；这一步只替自己这一位说话，不参与位置之间的相互参照。
        """
        if self.ops is not None and self.ops.kernels_ready(x.device) and x.size(-1) % 16 == 0:
            b, t, d = x.shape
            wide = self.up.out_features
            up = self.ops.linear(x.reshape(b * t, d).half(), self.up.weight.half(),
                                 self.up.bias.float()).view(b, t, wide)
            act = F.gelu(up, approximate="tanh")
            return self.ops.linear(act.reshape(b * t, wide), self.down.weight.half(),
                                   self.down.bias.float()).view(b, t, d).to(x.dtype)
        return self.down(F.gelu(self.up(x), approximate="tanh"))


class DecoderBlock(nn.Module):
    """pre-norm 一块：整形→加权汇总→加回主干；再整形→前馈→再加回主干。"""

    def __init__(self, config: ModelConfig, ops=None) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d, eps=config.norm_eps)
        self.attn = CausalAttention(config, ops=ops)
        self.norm2 = nn.LayerNorm(config.d, eps=config.norm_eps)
        self.mlp = MLP(config, ops=ops)
        self.ops = ops

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        visible: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """把主干 x（B,T,d）加工一层后原样宽交出去；两条分支都走“残差加回”，主干只加不减。

        内核链下“残差相加紧接第二次整形”正落在融合件的形状上：add_ln(x, attn_out, norm2)
        一枪出 (y1 供前馈, h1 为主干)；主干全程 fp32 的纪律与融合件双出口天然一致。

        白话：先把主干整形再让位置们相互参照，参照结果加回主干；再整形一次、再过一遍前馈关卡，
        结果又加回主干。主干一路只往上加，堆得再深也不至于把最初那点信息挤没了。
        """
        if self.ops is not None and self.attn._kernel_ok(x, x.size(1), visible):
            y0 = self.norm1(x).half()
            attn_out = self.attn(y0, cos, sin, None).to(x.dtype)
            y1, h1 = self.ops.add_ln(x, attn_out.float(), self.norm2.weight, self.norm2.bias,
                                     self.norm2.eps)
            return h1 + self.mlp(y1).float()
        x = x + self.attn(self.norm1(x), cos, sin, visible)
        x = x + self.mlp(self.norm2(x))
        return x


class Decoder(nn.Module):
    """整条链路：查表 → L 个 pre-norm 块 → 末整形 → 交出逐位置数字（last_hidden）。"""

    def __init__(self, config: ModelConfig, *, kernel_backend: str = "off") -> None:
        super().__init__()
        config.validate()
        if kernel_backend not in ("off", "tilelang"):
            raise ValueError(f"kernel_backend 只认 off/tilelang，实得 {kernel_backend!r}")
        self.kernel_backend = kernel_backend
        self.config = config
        # 初始化只认 config.seed：同 config 必得同权重（“无 run-id 不评测”的模型侧镜像）。
        # 建完就把全局随机状态原样还回去——一个模型的种子不该泄漏给调用方后面的随机数。
        saved_state = torch.random.get_rng_state()
        torch.manual_seed(config.seed)
        try:
            ops = None
            if kernel_backend == "tilelang":
                from sys1.kernels import autograd as ops   # 可微内核门面；单向依赖，不成环
            self.tok_emb = nn.Embedding(config.vocab, config.d)
            self.blocks = nn.ModuleList(DecoderBlock(config, ops=ops) for _ in range(config.L))
            self.final_norm = nn.LayerNorm(config.d, eps=config.norm_eps)
            # lm_head 独立成模块且不参与 forward：decision/ 直接按行取它（26 个字母那几行），
            # 二阶段换 backbone 时“行还是那些行”，读出代码零改动。bias=False 与参考实现一致。
            self.lm_head = nn.Linear(config.d, config.vocab, bias=False)
        finally:
            torch.set_rng_state(saved_state)

    def _tables(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """按当前设备/长度取角度表（一次 forward 只算一遍，逐层复用同一张表）。"""
        return rope_tables(seq_len, self.config.head_dim, self.config.rope_theta, device=device)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """(B,T) 编号进，(B,T,d) 逐位置数字出——本模型对外唯一的前向出口。

        形状与取值都不含“整张对照表的分数”：那是 lm_head 的事，由调用方（p1-07 读出、p1-08
        训练）自己决定取哪几行、取哪个位置。attn_mask 为 (B,T) 的 0/1，1=真 token、0=补洞位；
        传 None 表示整批都没有补洞。补洞位永远做不了键，所以批内补洞不改已算出的位置。

        白话：编号先换成数字串，逐层往前加工，每一层都只许看自己和左边、不看还没发生的位置；
        最后每个位置留下一串可学的数字。这串数字怎么变成答案，由外面读的人定，这里不替它猜。
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids 需为 (B,T) 两维，实得 {tuple(input_ids.shape)}")
        batch, seq = input_ids.shape
        if seq > self.config.ctx:
            raise ValueError(f"序列长 {seq} 超过 config.ctx={self.config.ctx}，位置表不够用")
        cos, sin = self._tables(seq, input_ids.device)
        visible = _visible_mask(seq, attn_mask, batch, input_ids.device)
        x = self.tok_emb(input_ids)
        for block in self.blocks:
            x = block(x, cos, sin, visible)
        return self.final_norm(x)

    def save(self, out_dir: str | Path) -> Path:
        """按“模型目录”布局落盘：config.json + weights.safetensors，同目录补一份决策配置占位。

        已有的 decision_config.json 一律不覆盖（那是 p1-07 的产物）；tokenizer/ 目录本函数不管。
        权重取 detach/cpu/contiguous 的副本，safetensors 不接受共享内存的张量。

        白话：把设置单写成一个 json、把学到的数字写成一份权重文件，放进同一个目录；目录里再留
        一张“决策配置”的空壳占位，已经有人填过就绝不抹掉——别人家的东西不碰。
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / CONFIG_FILE).write_text(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(state, str(out / WEIGHTS_FILE))
        decision = out / DECISION_CONFIG_FILE
        if not decision.exists():  # 占位，不覆盖：真配置的所有权在决策域
            decision.write_text(json.dumps(DECISION_CONFIG_PLACEHOLDER, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
        return out

    @classmethod
    def load(cls, in_dir: str | Path, *, strict: bool = True) -> Decoder:
        """从“模型目录”读回：config.json 建骨架（含 seed 初始化），weights.safetensors 逐位填回。

        读入后一律置为 eval 模式：载入是为了推理/评测，不该带着 dropout 那类随机行为（本模型
        没有 dropout，但模式一致能让“同目录、同输入 → 逐位相同输出”这条往返契约说得出口）。

        白话：先照着设置单空造一个模型，再把目录里那份数字逐位填进对应的位置；对不上号就当场
        报错，绝不“差不多就行”——静默少一层、多一层是最难查的错。
        """
        src = Path(in_dir)
        cfg = ModelConfig.from_dict(json.loads((src / CONFIG_FILE).read_text(encoding="utf-8")))
        model = cls(cfg)
        model.load_state_dict(load_file(str(src / WEIGHTS_FILE)), strict=strict)
        return model.eval()


__all__ = [
    "CONFIG_FILE",
    "DECISION_CONFIG_FILE",
    "MLP",
    "WEIGHTS_FILE",
    "CausalAttention",
    "Decoder",
    "DecoderBlock",
    "ModelConfig",
    "apply_rope",
    "rope_tables",
]
