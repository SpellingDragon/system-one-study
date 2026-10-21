"""production.assets — 二阶段载重接缝：预训练 backbone 的拉取、载入与四道接缝强校验。

【做什么】
    给二阶段一个入口 `load_backbone(name)`：按名字把 ModelScope 上的 Qwen3.5-0.8B（主力）
    或 Qwen3-0.6B（备选）拉到本地缓存目录，装成半精度的因果模型，再交出一个"窄接口"对象
    （只露 `forward(input_ids, attn_mask) -> 逐位置数字` 与输出层权重行），使一阶段冻结的
    `sys1/decision/` 一个字都不用改就能换脑。同时把来源 repo/revision 连同字母编号、隐藏宽度
    等一起打包成可写进 run config 的溯源字典。

【怎么做】
    三步。① `fetch_snapshot()` 取权重目录：优先 ModelScope `snapshot_download`（国内实测
    ~9.5MB/s），失败或没装则退回 huggingface 同名仓库；返回路径与来源标记一起留痕。
    ② `check_seams()` 跑四道接缝校验（spec 的三校验 + design 第四校验）：
      ① 字母单 token：直接复用一个阶段 `sys1.lang.bpe.check_tokenizer` 的判定口径（52 字母逐个
        单独试编，只出一个符号且符号原文就是它本身），外加"渲染文本尾部接字母不吞并"的边界核对；
      ② letter_rows：从输出层权重按 26 个字母行号取出的行数与列宽必须等于 backbone 隐藏宽度；
      ③ 思考关闭：对话模板交出的 prompt 必须以 `decision.THINK_OFF_SUFFIX` 逐字节结尾，
        且 `enable_thinking=True` 时必须不同（否则"关闭"是假的）；
      ④ 多模态编号：config 顶层的 image/video/vision_start/vision_end 四个 token id 必须存在、
        与 tokenizer 里的实际编号一致，且不与 26 个字母编号相撞（p2-08 视觉塔依赖此接口）。
      任一不过抛 `SeamCheckError`，消息里带接缝名，拒载。
    ③ `Backbone` 薄壳包住 HF 模型：把多模态外壳里的纯文本解码器取出来（照 StartLux
    `load_model` 的 `model.language_model` 口径），前向只走 `use_cache=False`。
    权重布局转换不在本文件（那是 `production/backbone/` 的职责），本文件只管"载与接缝"。

【为什么】
    接缝是二阶段一切训练的前提：字母不是一整个符号，26 行读出就无从下手；模板尾部多了思考
    前缀，读点位读到的就不是"该出字母的那一位"。所以校验放在载入路径里、而且只许硬拦。
    被否方案一：让 decision/ 自己适配 backbone——违反 D4-4"决策契约零改动"红线，且把 backbone
    内部细节灌进全项目最重要的解耦点。
    被否方案二：接缝只做单测不做载入时校验——训练脚本可以绕过测试直接 load，一旦绕过就是
    几小时白跑；载入即校验的代价只是一次字母试编（微秒级）。
    被否方案三：用 AutoModelForCausalLM 直接装多模态检查点——检查点的权重名带
    `model.language_model.` 前缀，装错类会得到一屋子随机初始化的层且只有一条警告，
    静默错值比报错贵得多，故按 config 的 architectures 取类。
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# 一阶段冻结件：只读引用，绝不改（decision/ 零改动红线 D4-4）
from sys1.decision.render import (        # 渲染层：排版 + 思考关闭常量 + 版号
    RENDER_VERSION,
    THINK_OFF_SUFFIX,
    from_systemone,
    render as render_messages,
)
from sys1.lang.bpe import (               # 字母单 token 判定口径的原创处
    LETTERS_52,
    LOWERCASE,
    UPPERCASE,
    TokenizerCheckError,
    check_tokenizer as bpe_check,
    letter_rows as bpe_letter_rows,
)

#: 默认权重缓存根（release/bench/ms_models）——一阶段探针的 0.6B 就落在这里，二阶段沿用
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "bench" / "ms_models"

#: 名字别名一律归一到这个小写短码；上游（训练脚本/测试）只需要认这几个字
ALIASES = {
    "qwen3.5-0.8b": "qwen3.5-0.8b",
    "qwen35-0.8b": "qwen3.5-0.8b",
    "qwen3_5-0.8b": "qwen3.5-0.8b",
    "0.8b": "qwen3.5-0.8b",
    "qwen3-0.6b": "qwen3-0.6b",
    "qwen30.6b": "qwen3-0.6b",
    "0.6b": "qwen3-0.6b",
}


@dataclass(frozen=True)
class BackboneSource:
    """一个 backbone 的出处：ModelScope 与 huggingface 上的仓库号 + 版本号。"""

    key: str
    modelscope_repo: str
    hf_repo: str
    revision: str = "master"


#: 选型已定（父 design D9）：Qwen3.5-0.8B 主力（同 StartLux 系、原生 262K、自带视觉塔），0.6B 备选
REGISTRY: dict[str, BackboneSource] = {
    "qwen3.5-0.8b": BackboneSource("qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B"),
    "qwen3-0.6b": BackboneSource("qwen3-0.6b", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B"),
}


class SeamCheckError(RuntimeError):
    """接缝校验不过的拒载信号；消息第一句恒为接缝名，便于日志里一眼定位是哪道缝。"""


def resolve_source(name: str) -> BackboneSource:
    """把用户给的 backbone 名字归一到注册表条目，认不出来就列出可用名字。

    白话：报名字领东西，名字写法有点出入（大小写、点号、下划线）都认得；真认不出来的话，
    把能领的东西挨个报一遍，别让人对着一个空清单猜。

    :raises KeyError: 名字不在注册表里（消息含全部可用短码）。
    """
    raw = str(name).strip().lower().replace(" ", "").replace("_", "-")
    key = ALIASES.get(raw) or ALIASES.get(raw.replace(".", "-"))
    if key is None:
        raise KeyError(f"未知 backbone 名 {name!r}；可选：{sorted(REGISTRY)}")
    return REGISTRY[key]


@dataclass
class Snapshot:
    """一次拉取的结果：本地目录 + 来源 + 版本 + 实测字节量与耗时（run notes 的凭据）。"""

    path: Path
    repo: str
    revision: str
    source: str                      # "modelscope" / "huggingface" / "local"
    bytes_downloaded: int = 0
    seconds: float = 0.0
    endpoint: str = ""
    files: int = 0
    note: str = ""

    def as_config(self) -> dict[str, Any]:
        """摊成能直接写进 run config 的扁平字典（键名带 backbone_ 前缀，避免与训练参数撞名）。

        白话：把这次搬东西的账目摊平成一行一项的条目——从哪家搬的、哪个版本、搬进多少、
        用了多久、落在那个柜子的哪一格——抄进实验记录本就是一份能复查的凭据。"""
        return {
            "backbone_source": self.source,
            "backbone_repo": self.repo,
            "backbone_revision": self.revision,
            "backbone_snapshot": str(self.path),
            "backbone_download_bytes": self.bytes_downloaded,
            "backbone_download_seconds": round(self.seconds, 1),
            "backbone_endpoint": self.endpoint,
            "backbone_files": self.files,
        }


def _dir_bytes(path: Path) -> int:
    """目录里所有真文件的字节数合计（符号链接只数它本身，不跟着跳到缓存别处）。"""
    total = 0
    if not path.exists():        # 首次下载前这一格还不存在：按 0 算，差值才是净增量
        return 0
    for p in path.rglob("*"):
        try:
            if p.is_symlink():
                total += p.stat().st_size
            elif p.is_file():
                total += p.stat().st_size
        except OSError:  # 拉取过程中可能瞬时出现半截文件，读不到就跳过，不影响合计口径
            continue
    return total


def fetch_snapshot(
    name: str,
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    prefer: str = "modelscope",
) -> Snapshot:
    """取回 backbone 的本地目录：ModelScope 优先、huggingface 兜底，并如实记字节量与耗时。

    白话：先去国内那个仓库把东西搬到自家柜子里；柜子已经满（文件齐）就不重复搬，只报"这次
    没新搬东西"。国内那个搬不动（服务没装、连不上、没这个条目）才去国外那个试，两条都写清
    是从哪边拿到的——事后报告里"这权重到底哪来的"必须一句话能答。

    :param name: backbone 名（见 `resolve_source` 的别名表）。
    :param cache_dir: 本地缓存根，默认 `release/bench/ms_models`。
    :param revision: 版本号；None 用注册表里登记的那个。
    :param allow_patterns: 只取匹配的文件（如先只要分词器那几个小文件）。
    :param prefer: "modelscope"（默认）或 "huggingface"（显式换边）。
    :returns: `Snapshot`（路径、来源、实测字节量/耗时/镜像端点）。
    :raises FileNotFoundError: 两条远端路径都失败（消息含两边的原始报错摘要）。
    """
    src = resolve_source(name)
    rev = revision or src.revision
    root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    patterns = list(allow_patterns) if allow_patterns else None

    order = ["modelscope", "huggingface"] if prefer == "modelscope" else ["huggingface", "modelscope"]
    errs: list[str] = []
    for side in order:
        repo = src.modelscope_repo if side == "modelscope" else src.hf_repo
        try:
            if side == "modelscope":
                return _pull_modelscope(repo, rev, root, patterns, side)
            return _pull_huggingface(repo, rev, root, patterns, side)
        except Exception as exc:  # noqa: BLE001 —— 兜底要的就是"这条不行换下一条"，异常留到最后合并上报
            errs.append(f"{side}({repo}): {type(exc).__name__}: {str(exc)[:180]}")
    raise FileNotFoundError(f"两条远端都没能取到 {name}；逐条报错：\n  " + "\n  ".join(errs))


def _pull_modelscope(repo: str, rev: str, root: Path, patterns: list[str] | None, side: str) -> Snapshot:
    """走 ModelScope 的 snapshot_download；先记目录规模，回来算差值。"""
    from modelscope import snapshot_download  # 延迟 import：没装时由调用侧转去 huggingface

    scope = _repo_cache_dir(root, repo)
    before = _dir_bytes(scope)
    t0 = time.time()
    kw: dict[str, Any] = {"cache_dir": str(root), "revision": rev}
    if patterns:
        kw["allow_patterns"] = patterns
    path = Path(snapshot_download(repo, **kw))
    seconds = time.time() - t0
    return Snapshot(
        path=path, repo=repo, revision=rev, source=side,
        bytes_downloaded=max(0, _dir_bytes(_repo_cache_dir(root, repo)) - before), seconds=seconds,
        endpoint=_ms_endpoint(), files=_snapshot_files(path),
    )


def _pull_huggingface(repo: str, rev: str, root: Path, patterns: list[str] | None, side: str) -> Snapshot:
    """走 huggingface_hub 兜底（HF_ENDPOINT 环境变量决定镜像端点，如实记下）。"""
    import os

    from huggingface_hub import snapshot_download as hf_download

    scope = _repo_cache_dir(root, repo)
    before = _dir_bytes(scope)
    t0 = time.time()
    kw: dict[str, Any] = {"cache_dir": str(root), "revision": rev if rev != "master" else "main"}
    if patterns:
        kw["allow_patterns"] = patterns
    path = Path(hf_download(repo, **kw))
    seconds = time.time() - t0
    return Snapshot(
        path=path, repo=repo, revision=rev, source=side,
        bytes_downloaded=max(0, _dir_bytes(_repo_cache_dir(root, repo)) - before), seconds=seconds,
        endpoint=os.environ.get("HF_ENDPOINT", "https://huggingface.co"), files=_snapshot_files(path),
    )


def _snapshot_files(path: Path) -> int:
    """快照目录里的文件个数（给 run notes 一个"齐不齐"的粗判据）。"""
    return sum(1 for p in path.rglob("*") if p.is_file() or p.is_symlink())


def _ms_endpoint() -> str:
    """取 ModelScope 实际使用的服务地址；取不到就返回官方默认值。"""
    import os

    override = os.environ.get("MODELSCOPE_DOMAIN") or os.environ.get("MODELSCOPE_ENDPOINT")
    if override:
        return str(override)
    try:
        from modelscope.hub import constants

        val = getattr(constants, "DEFAULT_MODELSCOPE_DATA_ENDPOINT", None)
        if val:
            return str(val)
    except Exception:  # noqa: BLE001
        pass
    return "https://www.modelscope.cn"


def _repo_cache_dir(root: Path, repo: str) -> Path:
    """缓存根里"只属于这个仓库"的那一块（ModelScope 的目录名是 org--name）。

    白话：柜子里搬来了好几家的东西时，要算这次搬进多少，就得只开这一家的那格称重；
    拿整柜说事会把邻居刚搬进来的也算成自己的，数字大得没边还没法复现。
    """
    for cand in (root / "models" / repo.replace("/", "--"), root / repo.replace("/", "--"), root / repo):
        if cand.exists():
            return cand
    return root


# ---------------------------------------------------------------- 接缝校验（四道，硬拦）
#: 第四校验要看的多模态编号字段（父 design：Qwen3.5 系 config 顶层自带这四个 id）
MM_TOKEN_FIELDS = ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")
#: 多模态编号在分词器里的名字（id 与名字必须互相对上，否则 p2-08 的图文装配会错位）
MM_TOKEN_NAMES = {
    "image_token_id": "＜image_pad＞",
    "video_token_id": "＜video_pad＞",
    "vision_start_token_id": "＜vision_start＞",
    "vision_end_token_id": "＜vision_end＞",
}
SEAM_LETTERS = "字母单 token"
SEAM_LETTER_ROWS = "letter_rows 维度"
SEAM_THINK_OFF = "思考关闭前缀"
SEAM_MM_IDS = "多模态编号冲突"


@dataclass
class SeamReport:
    """四道接缝的实测产出：字母编号、字母行形状、模板尾部与快照指纹、多模态编号表。"""

    letter_ids: tuple[int, ...]                 # A–Z 共 26 个编号，顺序即渲染层字母序
    letters_52: dict[str, int] = field(default_factory=dict)
    head_shape: tuple[int, int] = (0, 0)        # (vocab, 隐藏宽度)
    prompt_sha: str = ""                        # 关闭思考后 canonical prompt 的 sha256 前 16 位
    prompt_tokens: int = 0                      # canonical prompt 编出的符号个数
    think_off_suffix: str = ""                  # 实测到的模板尾部（应与 decision 的常量同文）
    mm_token_ids: dict[str, int] = field(default_factory=dict)

    def as_config(self) -> dict[str, Any]:
        """摊平成 run config 字段（接缝证据跟着 run 走，报告里能指回来）。

        白话：四道关卡各自量到的数——二十六个字母的号、挑出来的行有多宽、套壳尾巴的指纹
        与长度、贴图那几个记号的号——一并摊平，报告里每句结论都能指回这里核对。"""
        return {
            "seam_letter_ids": list(self.letter_ids),
            "seam_head_vocab": self.head_shape[0],
            "seam_hidden_size": self.head_shape[1],
            "seam_prompt_sha256_16": self.prompt_sha,
            "seam_prompt_tokens": self.prompt_tokens,
            "seam_think_off_suffix": self.think_off_suffix,
            "seam_mm_token_ids": dict(self.mm_token_ids),
        }


def _sha16(text: str) -> str:
    """字符串的 sha256 前 16 位（快照指纹；全文太长不进 config）。"""
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _tokenize_len(tokenizer: Any, text: str) -> list[int]:
    """按"不加任何额外特殊符"的口径把文字编成编号列表（返回普通 list，便于逐位比）。"""
    out = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(out, dict):
        return list(out["input_ids"])
    if hasattr(out, "ids"):
        return list(out.ids)
    return list(out)


def check_letters(tokenizer: Any, letters: str = LETTERS_52) -> dict[str, int]:
    """接缝①：字母单 token——复用一阶段 `sys1.lang.bpe.check_tokenizer` 的判定口径。

    白话：五十两个大小写字母挨个单独送去试编，每个都只该弹出一个符号、而且弹出来的原文
    就是它自己；再把渲染好的那截文字尾巴上接一个字母重编一遍，长度必须正好多出一位——
    少多一位都说明字母被前文"吞并"成了半个词，读点位就读不到干净的字母了。

    :param tokenizer: HF 分词器（`AutoTokenizer.from_pretrained` 的那个对象）。
    :param letters: 要抽查的字母串，默认 52 个（大写是决策读点，小写是一阶段词表纪律）。
    :returns: 字母 → 编号的映射（大写 26 个的有序列由 `letter_ids_from_mapping` 取）。
    :raises SeamCheckError: 任一字母不达标（消息含接缝名与逐字母实测细节）。
    """
    try:
        mapping = bpe_check(tokenizer.backend_tokenizer, letters)
    except TokenizerCheckError as exc:
        raise SeamCheckError(f"{SEAM_LETTERS} 失败：{exc}") from exc
    return mapping


def letter_ids_from_mapping(mapping: dict[str, int]) -> tuple[int, ...]:
    """把字母映射摊成 A–Z 的 26 个编号（顺序 = decision 渲染层的字母序）。"""
    return tuple(bpe_letter_rows(mapping, UPPERCASE))


def check_letter_boundary(tokenizer: Any, prompt: str, letter_ids: Sequence[int]) -> None:
    """接缝①的边界半边：prompt 尾部接任一字母，必须只多出该字母那一格。

    白话：在已经排好的那段文字末尾贴一个字母再重新试编，结果应当是"原来那一串 + 一个字母"，
    多出来的必须恰是那个字母自己的编号；要是末尾两格被搅在一起（比如字母并进了换行里），
    读点位拿到的就不是纯字母，26 行读出立刻失去意义。

    :raises SeamCheckError: 出现吞并（消息含接缝名与首个出问题的字母）。
    """
    base = _tokenize_len(tokenizer, prompt)
    for idx, letter in enumerate(UPPERCASE):
        merged = _tokenize_len(tokenizer, prompt + letter)
        if merged != base + [letter_ids[idx]]:
            got = merged[len(base):]
            raise SeamCheckError(
                f"{SEAM_LETTERS} 失败：字母 {letter!r} 接在渲染文字尾部后被并成 {got}"
                f"（期望 [{letter_ids[idx]}]），说明读点位上的字母不干净"
            )


def check_letter_rows(head_weight: torch.Tensor, letter_ids: Sequence[int], hidden_size: int) -> torch.Tensor:
    """接缝②：按 backbone 的隐藏宽度重建 26 行字母方向，并交出这批行。

    白话：从整张对照表里按编号挑出 26 行，逐行长度得和这副身板一样宽；宽了窄了都说明
    编号与权重的对应关系错位，挑出来的行配不上钥匙，读出的分数就是废数。

    :param head_weight: `(vocab, d)` 输出层权重。
    :param letter_ids:  26 个字母编号。
    :param hidden_size: backbone 文本栈的隐藏宽度（Qwen3.5-0.8B 实证 1024）。
    :returns: `(26, d)` 的字母行（float32 副本，供读点直接点积）。
    :raises SeamCheckError: 形状不符或编号越界（消息含接缝名）。
    """
    if head_weight.dim() != 2:
        raise SeamCheckError(f"{SEAM_LETTER_ROWS} 失败：输出层权重应为两维 (vocab,d)，实得 {tuple(head_weight.shape)}")
    vocab, width = int(head_weight.shape[0]), int(head_weight.shape[1])
    if width != hidden_size:
        raise SeamCheckError(
            f"{SEAM_LETTER_ROWS} 失败：输出层宽 {width} 与 backbone 隐藏宽度 {hidden_size} 不符"
        )
    out_of_range = [i for i in letter_ids if not 0 <= i < vocab]
    if out_of_range:
        raise SeamCheckError(f"{SEAM_LETTER_ROWS} 失败：字母编号 {out_of_range} 越出词表行数 {vocab}")
    ids = torch.tensor(list(letter_ids), dtype=torch.long)
    rows = head_weight.detach().index_select(0, ids.to(head_weight.device)).float()
    if tuple(rows.shape) != (len(letter_ids), hidden_size):
        raise SeamCheckError(f"{SEAM_LETTER_ROWS} 失败：取出的行形状 {tuple(rows.shape)} 不是 ({len(letter_ids)},{hidden_size})")
    return rows


def check_think_off(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, str]:
    """接缝③：对话模板必须交出"思考关闭"的固定尾巴，且开关确实能改变它。

    白话：把排好的信交给模板套上壳，套出来的末尾那截必须与一阶段钉死的那个"不想了"的记号
    一字不差；再把开关扳到"要想"，尾巴应当换成另一截——两截一样才说明开关是摆设，
    那时读点位读到的就是思维链中途，整份读出作废。

    :returns: `(关闭态全文, 实测尾巴)`。
    :raises SeamCheckError: 尾巴不符或开关无效（消息含接缝名）。
    """
    off = _apply_chat_template(tokenizer, messages, enable_thinking=False)
    if not off.endswith(THINK_OFF_SUFFIX):
        raise SeamCheckError(
            f"{SEAM_THINK_OFF} 失败：模板尾部是 {off[-len(THINK_OFF_SUFFIX) - 8:]!r}，"
            f"不是 decision 钉死的 {THINK_OFF_SUFFIX!r}"
        )
    on = _apply_chat_template(tokenizer, messages, enable_thinking=True)
    if on == off:
        raise SeamCheckError(f"{SEAM_THINK_OFF} 失败：enable_thinking 开关不改变模板输出，无法确认已关闭思考")
    return off, THINK_OFF_SUFFIX


def check_mm_token_ids(config: Any, tokenizer: Any, letter_ids: Sequence[int]) -> dict[str, int]:
    """接缝④：多模态编号存在、与分词器对得上、且不与 26 个字母编号相撞。

    白话：这副身板自带"贴图的位置"那几个记号。载入时先把它们登记在册：号子得在，
    号子与分词器那边的写法要互相认得出，还不能和字母抢同一个号——抢了的话，图文装配
    以后会把图片占位当成选项字母，视觉那条路就白铺。

    :raises SeamCheckError: 缺字段、编号对不上或撞号（消息含接缝名与冲突项）。
    """
    letters = set(letter_ids)
    found: dict[str, int] = {}
    for field_name in MM_TOKEN_FIELDS:
        value = getattr(config, field_name, None)
        if value is None:
            raise SeamCheckError(f"{SEAM_MM_IDS} 失败：config 顶层缺 {field_name}（视觉塔接口无从登记）")
        token_name = MM_TOKEN_NAMES[field_name].translate(_FULLWIDTH)
        tid = int(value)
        if tid in letters:
            raise SeamCheckError(f"{SEAM_MM_IDS} 失败：{field_name}={tid} 与字母编号相撞")
        got = tokenizer.convert_tokens_to_ids(token_name)
        if isinstance(got, int) and got >= 0 and got != tid:
            raise SeamCheckError(f"{SEAM_MM_IDS} 失败：{field_name}={tid} 与分词器里 {token_name!r}={got} 不一致")
        found[field_name] = tid
    if len(set(found.values())) != len(found):
        raise SeamCheckError(f"{SEAM_MM_IDS} 失败：多模态编号彼此重复 {found}")
    return found


#: 特殊符名字里的全角尖括号 → 真尖括号（写成全角是为了不让本文件的注释被误当成控制符）
_FULLWIDTH = {ord("＜"): "<", ord("＞"): ">"}


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], *, enable_thinking: bool) -> str:
    """套对话模板取文字（不做编号），thinking 开关按参数给。"""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )


# ---------------------------------------------------------------- 窄接口对象
#: 快照用的 canonical 请求：一问一答的最小题面（固定文字，跨 run 可比）
CANONICAL_ROW_SPEC = {
    "type": "choice",
    "instructions": "Which option fits the evidence?",
    "criteria": {"a": "sunny", "b": "rain", "c": "cloudy"},
}
CANONICAL_STATE = "weather: clear sky at 07:00, humidity 40%"


def canonical_prompt(tokenizer: Any) -> tuple[str, int, str]:
    """生成 canonical prompt（关闭思考）并交出实测指纹：文字、符号个数、sha 前 16 位。

    白话：拿那道最小题面走一遍排版和套壳，把得到的整段文字称个重量（编出多少符号、
    指纹是什么），日后模板或排版动过一个字节，这个数就会变，快照测试当场就能发现。

    :returns: `(prompt 文字, 符号个数, sha256 前 16 位)`。
    """
    row = from_systemone(CANONICAL_STATE, dict(CANONICAL_ROW_SPEC))
    messages, _ = render_messages(row)
    text = _apply_chat_template(tokenizer, messages, enable_thinking=False)
    return text, len(_tokenize_len(tokenizer, text)), _sha16(text)


@dataclass
class Backbone:
    """载好的 backbone + 四道接缝证据 + 决策程序要的窄接口（前向与输出层行）。

    对 `sys1/decision/` 而言，本对象只有两样东西有用：逐位置的数字 `(B,T,d)` 与输出层权重
    `(vocab,d)`；模型内部（哪几层是线性注意力、塔在哪儿）一概不往外露——这就是换脑时
    decision/ 零改动的物理保证。
    """

    model: Any
    body: Any
    tokenizer: Any
    snapshot: Snapshot
    seam: SeamReport
    dtype: torch.dtype
    device: str
    hidden_size: int
    letter_ids: tuple[int, ...]
    letter_rows: torch.Tensor

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """窄接口前向：`(B,T)` 编号 + `(B,T)` 可见位 → `(B,T,d)` 逐位置数字（不回填任何生成物）。

        白话：把一排排字送进去，只收"每个位置读完之后留下的那串数字"，不收写好的字；
        批次里被垫了空洞的行照样收，读点位由调用方按每行真长度去取。

        :param input_ids: `(B,T)`  long 编号。
        :param attn_mask: `(B,T)`  0/1 可见位；None 表示整行都是真符号。
        :returns: `(B,T,d)`，dtype 与载入精度一致。
        """
        with torch.no_grad():
            dev = next(self.body.parameters()).device   # 入参跟随模型设备（CPU 构造/MPS·NPU 载入跨设备必配）
            kwargs: dict[str, Any] = {"input_ids": input_ids.to(dev), "use_cache": False, "return_dict": True}
            if attn_mask is not None:
                kwargs["attention_mask"] = attn_mask.to(dev)
            out = self.body(**kwargs)
        return out.last_hidden_state

    def encode_prompt(self, row: dict[str, Any]) -> tuple[list[int], list[str]]:
        """把一行渲染成"套壳关闭思考"的编号列，返回 `(ids, 字母对应的候选代号)`。

        白话：把那道题排成信、套上壳、编成号，读的人拿这串号送前向；读点位就是最后一个号
        的位置，末尾没有任何垫的空洞。

        :param row: `from_systemone()` 产出的渲染行。
        :returns: `(编号列表, order)`，`order[i]` 是字母 `LETTERS[i]` 代表的候选代号。
        """
        messages, order = render_messages(row)
        text = _apply_chat_template(self.tokenizer, messages, enable_thinking=False)
        return _tokenize_len(self.tokenizer, text), order

    def provenance_config(self) -> dict[str, Any]:
        """run config 要的溯源 + 接缝字典（repo/revision 必含，spec "来源可溯"场景）。

        白话：把"东西从哪家搬来、搬了多少、过了哪几道检查"合成一张说明单一次交齐，
        省得写记录时东抄一行西抄一行，最要紧的出处反而漏掉。"""
        cfg: dict[str, Any] = {
            "backbone_name": self.snapshot.repo,
            "backbone_dtype": str(self.dtype).replace("torch.", ""),
            "backbone_device": self.device,
            "backbone_hidden_size": self.hidden_size,
            "render_version": RENDER_VERSION,
        }
        cfg.update(self.snapshot.as_config())
        cfg.update(self.seam.as_config())
        return cfg


def load_backbone(
    name: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
    allow_patterns: Sequence[str] | None = None,
    with_visual: bool = False,
    prefer: str = "modelscope",
) -> Backbone:
    """载入 backbone：拉快照 → 装模型 → 跑四道接缝 → 交窄接口对象。

    白话：先把东西搬到自家柜子，再照检查点自己声明的那个类把它拼起来（拼错类会得到
    一屋子随机初始化的层，所以这里不图省事用通用入口）；拼好先过四道关卡——字母能不能
    一个符号表示、挑出的字母行宽窄对不对、模板尾巴是不是"不想了"那一截、图片记号有没有
    和字母抢号。四道全过才把东西交出去，任何一道不过就拒载并说清是哪道。

    :param name: backbone 名（见 `resolve_source`）。
    :param device: 落位设备；MPS 被他用占用时走 CPU 也能跑通前向（0.8B 级）。
    :param dtype: 载入精度，默认 fp16（MPS 上 bf16 受限，见 kernels/README 收敛口径）。
    :param cache_dir: 本地缓存根，默认 `release/bench/ms_models`。
    :param revision: 版本号覆盖（None 用注册表）。
    :param allow_patterns: 只取部分文件（如先只取分词器）。
    :param with_visual: True 则保留完整多模态外壳（p2-08 用），False 只取纯文本栈。
    :param prefer: 远端优先顺序。
    :returns: `Backbone`。
    :raises SeamCheckError: 任一道接缝不过。
    :raises FileNotFoundError: 两条远端都取不到。
    """
    import transformers
    from transformers import AutoTokenizer

    snap = fetch_snapshot(
        name, cache_dir=cache_dir, revision=revision, allow_patterns=allow_patterns, prefer=prefer
    )
    tokenizer = AutoTokenizer.from_pretrained(str(snap.path))
    config = transformers.AutoConfig.from_pretrained(str(snap.path))
    # 按检查点自己声明的类装（StartLux load_model 同口径），而不是 AutoModelForCausalLM 猜一个
    cls_name = (config.architectures or ["Qwen3_5ForConditionalGeneration"])[0]
    model_cls = getattr(transformers, cls_name, None) or transformers.AutoModelForCausalLM
    model = model_cls.from_pretrained(str(snap.path), dtype=dtype)
    model.eval()
    model.to(device)

    text_config = config.get_text_config()
    hidden = int(getattr(text_config, "hidden_size"))
    body = getattr(model.model, "language_model", model.model) if not with_visual else model

    mapping = check_letters(tokenizer)
    letter_ids = letter_ids_from_mapping(mapping)
    head_weight = model.get_output_embeddings().weight
    rows = check_letter_rows(head_weight, letter_ids, hidden)
    canonical_row = from_systemone(CANONICAL_STATE, dict(CANONICAL_ROW_SPEC))
    messages, _ = render_messages(canonical_row)
    _, suffix = check_think_off(tokenizer, messages)
    text, ntok, sha = canonical_prompt(tokenizer)
    check_letter_boundary(tokenizer, text, letter_ids)
    mm = check_mm_token_ids(config, tokenizer, letter_ids)

    seam = SeamReport(
        letter_ids=letter_ids, letters_52=mapping, head_shape=(int(head_weight.shape[0]), hidden),
        prompt_sha=sha, prompt_tokens=ntok, think_off_suffix=suffix, mm_token_ids=mm,
    )
    return Backbone(
        model=model, body=body, tokenizer=tokenizer, snapshot=snap, seam=seam, dtype=dtype,
        device=str(device), hidden_size=hidden, letter_ids=letter_ids, letter_rows=rows,
    )
