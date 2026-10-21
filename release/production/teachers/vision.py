"""视觉教师：图文请求 → 选项分布；主路吃离线伪标包，在线产标只在打标那一次打开。

【做什么】
    输入是 (一张图, 一段请求文字, 候选清单)，输出每个候选的概率。两条来路：
    ① 回放路（默认）：以前产好的分布从 parquet 包里取，一次在线请求都不发；
    ② 产标路：外部视觉大模型（本项目 = GLM-5.3-Flash）对每个候选给 0–10 的信心分，
      折成 soft 分布，产完立刻落盘。按 §11-4 红线，外部只做教师离线产标，
      训练/评测/服务全链不得依赖在线接口——所以本模块默认 mode="pack"。

【怎么做】
    三段。图像先定指纹与传输编码（字节或文件路径都收，PIL 对象现场存成 PNG，
    指纹进缓存键：换图必换键，绝不把旧标签扣在新图上）。主路提示词要求"逐候选输出
    `A=8` 这样的 0–10 信心分"，`parse_confidences` 按候选表逐行收，收到 0–10 之外的
    数或漏收任一候选即判不可用；折份额走 `confidence_distribution`，给每个候选垫一点
    底再线性摊平，"8 vs 4" 就是 2:1，模型报的信心比例被原样带进伪标。
    主路失败（接口没按格式答）才退到 answer 协议：约束作答、抽值、折成近 one-hot；
    抽不出就重试一轮，两轮都败返回 None，由调用方按缺失处理。
    每次真打接口都在 `TeacherStats.api_calls` 与用量表上加一笔——预算与熔断有数可依。

【为什么】
    本地 9B 级 VLM 在家用内存里跑不动（单跑纯推理 ~18GB，还要与训练争），而视觉伪标
    只吃"结果"不吃"过程"，一次产完反复回放即可。被否方案一：训练/评测里在线调用——
    §11-4 明确禁止，且网络一断整条链就停摆；被否方案二：只让外部模型答一个字母——
    丢掉其余候选的信心信息，蒸馏与校准都无稠密信号可用，故逐选项置信为主路、
    单答案只作兜底；被否方案三：把 0–10 直接 softmax——指数放大让 8 与 7 的差被夸成
    数倍，伪标比原模型自信得多，线性摊平才不篡改信心的比例。
"""
from __future__ import annotations

import base64
import io
import math
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .cache import DistCache, image_digest
from .protocol import extract_answer, hard_distribution, retry_prompt
from .text import TeacherStats

__all__ = [
    "GlmApiBackend",
    "MissingApiKeyError",
    "TeacherUnreachableError",
    "VisionTeacher",
    "confidence_distribution",
    "parse_confidences",
]

# 教师标识进缓存键与账本：换模型必然 miss（版本污染比慢更可怕）
DEFAULT_MODEL_ID = "glm-5.3-flash"
DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
API_KEY_ENV = "ZAI_API_KEY"          # 只读环境变量，值绝不写进任何文件与日志
CONF_SCALE = 10.0                   # 逐候选信心的量纲（0–10）
# 回复预算：GLM-5.3-Flash 是"先想后答"的模型，实测 64 的预算全被思考吃掉、正文交空串
# （reasoning_tokens 62/64），因此默认给到 512；用量按 reasoning_tokens 单列，成本可核。
DEFAULT_MAX_TOKENS = 512
CONF_FLOOR = 0.5                    # 摊平时的垫底：0 分候选也留一线份额，防纯零伪标
SOURCE_PACK = "offline_pack"
SOURCE_API = "api_label"

# `A=8` / `A : 8.5` / `A - 10` 三种写法都收；值必须落在 0–10
_CONF_LINE = re.compile(r"(?P<key>[A-Za-z][A-Za-z0-9_]*)\s*[=:：\-]\s*(?P<val>\d+(?:\.\d+)?)")


class TeacherUnreachableError(RuntimeError):
    """产标路走不通（网络/接口报错/格式三次都不对）——把"拿不到标签"说成显式失败。"""


class MissingApiKeyError(TeacherUnreachableError):
    """环境变量里没有密钥：直接停，绝不猜测或回退到任何内置密钥。"""


def encode_image(image: Any) -> tuple[str, str]:
    """任意图像入参 → `(base64 文本, 内容指纹)`；字节/路径/PIL 对象三种写法都收。

    白话：先把图变成"一串能寄出去的字符"和一个"这图长什么样的身份证号"。身份证号
    由图的字节内容算出来，所以同名换了图，号码必变，不会把旧标签错扣到新图上。
    """
    if isinstance(image, (bytes, bytearray, memoryview)):
        raw = bytes(image)
    elif isinstance(image, (str, Path)) and Path(image).is_file():
        raw = Path(image).read_bytes()
    elif hasattr(image, "save"):  # PIL 之类可保存对象：本仓未装 PIL 时走不到这里
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        raw = buffer.getvalue()
    else:
        raise ValueError(f"无法识别的图像入参 {type(image).__name__}（要 bytes / 文件路径 / 可 save 的图像对象）")
    if not raw:
        raise ValueError("图像字节为空")
    return base64.b64encode(raw).decode("ascii"), image_digest(raw)


def parse_confidences(text: str, option_keys: Sequence[str]) -> dict[str, float] | None:
    """从接口回复里收"逐候选 0–10 信心分"；缺任一候选或越界即判不可用（返回 None）。

    白话：教师照格式回一行行 `A=8` 就照单全收；只要少答了一个候选，或者答了 11 分
    这种出格数，整份都算不作数——宁可重来，也不给半截标签。
    """
    keys = [str(k) for k in option_keys]
    if not isinstance(text, str) or not text.strip():
        return None
    got: dict[str, float] = {}
    for match in _CONF_LINE.finditer(text):
        key = match.group("key").upper() if len(match.group("key")) == 1 else match.group("key")
        if key not in keys:
            continue  # 回复里夹带的额外行（如 "NONE=0"）不参与
        value = float(match.group("val"))
        if not 0.0 <= value <= CONF_SCALE:
            return None
        got[key] = value
    if len(got) != len(keys):
        return None
    return {k: got[k] for k in keys}


def confidence_distribution(confidences: dict[str, float], *, floor: float = CONF_FLOOR) -> dict[str, float]:
    """把 0–10 信心分线性摊成份额（和为 1）；垫一点底，全 0 时退成均匀。

    白话：教师给每个候选打个十分制信心，这里把信心直接摊成百分比份额——8 分就是
    4 分的两倍，不额外放大也不压缩。每个候选先垫一小笔，免得"0 分"被读成"绝不可能"。
    """
    if not confidences:
        raise ValueError("confidences 不能为空")
    if floor < 0.0 or not math.isfinite(floor):
        raise ValueError(f"floor 必须是非负有限数，实得 {floor!r}")
    keys = list(confidences)
    weights = [float(confidences[k]) + floor for k in keys]
    total = sum(weights)
    if total <= 0.0:  # floor=0 且全 0 分：没任何信息可用，退均匀并如实留痕
        even = 1.0 / len(keys)
        return {k: even for k in keys}
    return {k: w / total for k, w in zip(keys, weights)}


class GlmApiBackend:
    """外部视觉大模型客户端：只管"发一次图文请求、拿回文字与用量"，不碰缓存不折份额。

    白话：这是那台"外部打分机器"的接线端子。它把图和问题寄出去、把回信收回来，
    顺便记下这次花了多少字。寄件地址与密钥都从环境变量取，本机不留副本。
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = API_KEY_ENV,
        timeout: float = 60.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stats: TeacherStats | None = None,
        budget: int | None = None,
        session: Any = None,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.max_tokens = int(max_tokens)
        self.stats = stats if stats is not None else TeacherStats()
        self.budget = budget
        self.usage: dict[str, int] = {
            "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "seconds": 0}
        self._session = session

    # 上层的"逐候选置信 / answer 协议"两条提示词都从这里寄出
    def chat(self, content: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        """发一次多模态对话，返回 `(回复正文, 用量字典)`；失败抛 TeacherUnreachableError。

        白话：把图和话一起寄出去，等回信。寄之前先确认存放密钥的地方有密钥、这次寄件
        还没超出预算；回信不好就直接说"这条路走不通"，并记下这次花了多少字数、多久。
        """
        import os

        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise MissingApiKeyError(f"环境变量 {self.api_key_env} 未设置——产标路拒绝用任何内置密钥顶上")
        if self.budget is not None and self.usage["calls"] >= self.budget:
            raise TeacherUnreachableError(f"本次产标预算 {self.budget} 次已用尽（熔断，防超支）")
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        session = self._session or _requests_session()
        started = time.time()
        try:
            response = session.post(self.base_url, json=payload, headers={"Authorization": f"Bearer {api_key}"}, timeout=self.timeout)
        except Exception as exc:  # 网络层异常类型太杂，统一收口成显式不可达
            raise TeacherUnreachableError(f"接口请求失败：{type(exc).__name__}: {exc}") from exc
        elapsed = time.time() - started
        if response.status_code != 200:
            # 错误正文只截前 200 字，绝不打印请求头（里面有密钥）
            raise TeacherUnreachableError(f"接口返回 HTTP {response.status_code}: {response.text[:200]}")
        body = response.json()
        text = _content_of(body)
        usage = _usage_of(body)
        self.stats.api_calls += 1
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        self.usage["completion_tokens"] += int(usage.get("completion_tokens", 0))
        details = usage.get("completion_tokens_details") or {}
        self.usage["reasoning_tokens"] += int(details.get("reasoning_tokens", 0))
        self.usage["seconds"] += round(elapsed, 3)
        return text, {"usage": usage, "seconds": elapsed}

    def confidences(self, image_b64: str, prompt: str, option_keys: Sequence[str]) -> tuple[dict[str, float] | None, dict[str, Any]]:
        """主路：要逐候选 0–10 信心分；返回 `(信心表或 None, 这次请求的用量)`。

        白话：先按"给每个候选打个十分制信心"的格式问一次。教师照格式答就把表交回去，
        不照格式就交 None——格式对不对由 `parse_confidences` 当场验，不猜。
        """
        keys = [str(k) for k in option_keys]
        question = (
            "For EACH option below, give a confidence score from 0 to 10 that it is the correct answer "
            "to the question. Reply with one line per option in exactly the form `KEY=score` "
            f"(integers or decimals, no other text). Options: {', '.join(keys)}\n\n{prompt}"
        )
        text, info = self.chat(_content(image_b64, question))
        return parse_confidences(text, keys), {**info, "raw": text[:400]}

    def answer(self, image_b64: str, prompt: str, option_keys: Sequence[str], *, qtype: str = "choice") -> tuple[str | None, dict[str, Any]]:
        """兜底路：按 answer 协议要一个确定值；抽不出时重试一轮，仍不出返回 None。

        白话：主路没拿到整齐的分数表时才走这里——只要求教师"用尖括号把答案包起来"，
        第一次不守格式就补一句更硬的提醒再问一次，两次都不守就算了。
        """
        keys = [str(k) for k in option_keys]
        asked = ""
        for attempt in (0, 1):
            base = prompt if attempt == 0 else retry_prompt(prompt, keys, qtype=qtype)
            question = f"{base}\n\nReply with exactly <answer>VALUE</answer> and nothing else."
            asked = question
            text, info = self.chat(_content(image_b64, question))
            value = extract_answer(text, qtype=qtype, option_keys=keys)
            if value is not None:
                return value, {**info, "raw": text[:400]}
        return None, {"raw": asked[:200]}


class VisionTeacher:
    """视觉教师门面：`score(image, prompt, option_keys)`；mode 决定能否打接口。

    白话：训练与评测只认这一个入口。默认它是"只翻账本"的回放模式——账本里没有就明确
    报缺，绝不自作主张联网；只有打标的那次运行才把它开成产标模式。
    """

    MODES = ("pack", "api")

    def __init__(
        self,
        cache: DistCache | None = None,
        *,
        mode: str = "pack",
        backend: GlmApiBackend | None = None,
        model_id: str | None = None,
        stats: TeacherStats | None = None,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode 必须是 {self.MODES} 之一，实得 {mode!r}")
        if mode == "api" and backend is None:
            raise ValueError("mode='api' 必须给 backend（回放模式则不需要任何在线依赖）")
        self.cache = cache
        self.mode = mode
        self.backend = backend
        self.model_id = model_id or (backend.model_id if backend is not None else DEFAULT_MODEL_ID)
        self.stats = stats if stats is not None else (backend.stats if backend is not None else TeacherStats())

    def score(
        self,
        image: Any,
        prompt: str,
        option_keys: Sequence[str],
        *,
        qtype: str = "choice",
        use_cache: bool = True,
    ) -> dict[str, float]:
        """图文请求 → 份额分布。命中包内标签时零在线请求；未命中按 mode 决定产标或报缺。

        白话：先按"图 + 问题 + 候选"去账本里查。查到了就原样交出，一分钱算力不花；
        查不到时，回放模式老实说"包里没这条"，产标模式才真的寄出去问外部教师。
        """
        keys = [str(k) for k in option_keys]
        if not keys:
            raise ValueError("option_keys 不能为空")
        digest, image_b64 = _image_pair(image)
        cached = None
        if self.cache is not None and use_cache:
            cached = self.cache.lookup(self.model_id, prompt, keys, image_hash=digest)
            if cached is not None:
                self.stats.cache_hits += 1
                return {k: float(cached[k]) for k in keys if k in cached} or cached
            self.stats.cache_misses += 1
        if self.mode == "pack":
            raise TeacherUnreachableError(
                f"离线伪标包缺这条标签（{self.model_id}/{digest}）：请先跑产标批次回填，训练评测链不得在线补算"
            )
        dist = self._label(image_b64, prompt, keys, qtype=qtype)
        if self.cache is not None and use_cache:
            self.cache.record(
                model_id=self.model_id, prompt=prompt, option_keys=keys, dist=dist,
                image_hash=digest, source=SOURCE_API, meta={"labeler": self.model_id},
            )
        return dist

    def flush(self) -> int:
        """把本次产标的行落盘成伪标包（一次批次结束必须调，否则结果只在内存里）。

        白话：这一批打好的标签要誊到账本文件里才算存住；不誊，程序一关就全没了。
        """
        if self.cache is None:
            return 0
        return self.cache.flush()

    def _label(self, image_b64: str, prompt: str, keys: list[str], *, qtype: str) -> dict[str, float]:
        """产标一条：主路要信心表，拿不到就退 answer 协议，两轮都败则显式失败。"""
        conf, info = self.backend.confidences(image_b64, prompt, keys)
        if conf is not None:
            dist = confidence_distribution(conf)
            return dist
        value, _ = self.backend.answer(image_b64, prompt, keys, qtype=qtype)
        if value is None:
            raise TeacherUnreachableError(f"外部教师两次都未按格式作答，无法产标（候选 {keys}）")
        return hard_distribution(value, keys)


def _image_pair(image: Any) -> tuple[str, str]:
    """统一取 `(指纹, base64)`：回放模式其实用不到 base64，但入参口径要与产标一致。"""
    if isinstance(image, tuple) and len(image) == 2 and all(isinstance(x, str) for x in image):
        # 允许调用方直接交 (digest, base64)——大批量产标时可复用已编码结果
        return str(image[0]), str(image[1])
    b64, digest = encode_image(image)
    return digest, b64


def _content(image_b64: str, question: str) -> list[dict[str, Any]]:
    """拼多模态消息体：图（base64 不带 data: 前缀）+ 文本问题，顺序固定。"""
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", "text": question},
    ]


def _requests_session() -> Any:
    """取一个 requests 会话（延迟 import：只吃离线包时完全不需要网络栈）。"""
    import requests

    return requests.Session()


def _content_of(body: dict[str, Any]) -> str:
    """从回复体里取正文文本；结构不合预期就交空串（由上层判"格式不对"）。"""
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    content = message.get("content", "")
    if isinstance(content, list):  # 部分接口把多段内容拆成 list 返回
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)


def _usage_of(body: dict[str, Any]) -> dict[str, Any]:
    """取用量（token 账），供成本报表逐段核账用。"""
    usage = body.get("usage") if isinstance(body, dict) else None
    return usage if isinstance(usage, dict) else {}


def pack_rows(cache_root: str | Path, *, limit: int = 5) -> list[dict[str, Any]]:
    """看一眼包里有什么（前 limit 行摊成 dict），供冒烟脚本与 run notes 记证据。

    白话：不用把整包读进内存，先掀开盖子数几行，看看键、模型名、份额都齐不齐。
    """
    directory = Path(cache_root)
    files = sorted(directory.glob("*.parquet"))
    rows: list[dict[str, Any]] = []
    for path in files:
        table = pq.read_table(path)
        for i in range(min(table.num_rows, max(0, limit - len(rows)))):
            rows.append({col: table.column(col)[i].as_py() for col in table.column_names})
        if len(rows) >= limit:
            break
    return rows
