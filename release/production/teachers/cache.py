"""教师分布缓存：把"算过一次的结果"落成 parquet 行，让同一请求第二次零成本命中。

【做什么】
    教师（文本或视觉）每打一次分，产出的是一份"每个候选各占多少"的分布。本模块把这份
    分布连同它的身份三件套——模型名、请求正文、候选键——算成一个短哈希键写进 parquet；
    下次同样的三件套再来，直接读表返回，一次前向都不发。同一个目录还能装外部产好的
    伪标包，离线回放即可，不再打任何在线请求。

【怎么做】
    三段。第一段是键：`sha256(model_id + prompt + sorted(keys))` 截 16 位十六进制，视觉
    请求另把图像指纹并进材料（不给就与文本请求同构，互不污染）。第二段是读写：首次访问
    时扫目录下所有 `*.parquet` 合并成内存字典（行很小，几千行不过几兆），写侧只追加新
    分片文件、绝不回头改旧文件——同一片目录里多进程各写各的也不会互相覆盖。第三段是
    落盘：`record()` 先进待写缓冲，`flush()` 用 pyarrow 把缓冲写成 `shard-*.parquet`
    （先写 .tmp 再原子改名，读者永远看不到半截文件）。
    数据流：`lookup/record` → 命中即返 → 未命中由教师算完 `record` 回写 → `flush`。

【为什么】
    教师打分是本项目最贵的一段（本地 4B 级前向 + 外部产标按次计费），而三栈训练与评测
    会对同一批请求反复取用同一份分布——不缓存等于把同一笔钱付三遍，父设计 D3 因此把
    "缓存先行"列为并发前提。被否方案一：jsonl 追加文件——读侧要逐行扫、写侧无法去重，
    行数千以后启动就慢；parquet 自带列式压缩与按列读取，pyarrow 已在依赖里。
    被否方案二：键里只放 prompt——换教师不 miss，旧分布会冒充新教师的答案，属于静默
    数据污染，比慢一百倍都严重；所以 model_id 必须进键。
    被否方案三：一个大 parquet 就地重写——并发写互踩且写一半崩了整包报废，改为只追加。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["CacheMissError", "DistCache", "image_digest", "make_key"]

# 键截断长度：16 位十六进制 = 64 bit，几万行的碰撞概率可忽略，目录名/表列都短。
KEY_HEX_LEN = 16
# 表列顺序写死在这里：伪标包与缓存共用同一套列，多出的字段留空串，读侧按列取。
COLUMNS = ("key", "model_id", "prompt", "option_keys", "dist", "image_hash", "source", "created_at", "meta")


class CacheMissError(KeyError):
    """只读模式（离线伪标包）下查无此键——教师不许自作主张去算，必须把话挑明。"""


def _digest(material: str | bytes) -> str:
    """把任意字符串或字节压成 16 位十六进制（缓存键与图像指纹共用同一压法）。"""
    raw = material if isinstance(material, bytes) else material.encode("utf-8", "surrogatepass")
    return hashlib.sha256(raw).hexdigest()[:KEY_HEX_LEN]


def make_key(
    model_id: str,
    prompt: str,
    option_keys: Sequence[str],
    *,
    image_hash: str | None = None,
) -> str:
    """请求三件套 → 缓存键；候选顺序不参与键值，视觉请求额外并入图像指纹。

    白话：先报"哪个模型、问了什么、给了哪几个候选"，把它们搅成一小串字母数字当门牌；
    候选的排列顺序不算新门牌，因为答案与顺序无关。看图请求再多加一项"图的指纹"。
    """
    if not isinstance(model_id, str) or not model_id:
        raise ValueError(f"model_id 必须是非空字符串（换教师必须 miss），实得 {model_id!r}")
    if not isinstance(prompt, str):
        raise ValueError(f"prompt 必须是字符串，实得 {type(prompt).__name__}")
    keys = sorted(str(k) for k in option_keys)
    if not keys:
        raise ValueError("option_keys 不能为空——没有候选就没有分布")
    if len(set(keys)) != len(keys):
        raise ValueError(f"option_keys 含重复键：{list(option_keys)}")
    material = "\x1f".join([model_id, prompt, ",".join(keys), image_hash or ""])
    return _digest(material)


def image_digest(image: bytes | str | os.PathLike) -> str:
    """图像内容（字节或文件路径）→ 16 位指纹；同名换图必然换键，不会读到旧标签。

    白话：给图片算一个"内容身份证号"，内容动一个字就换号。路径读回来算，字节直接算，
    这样包里的图和产标时的图能严丝合缝对上。
    """
    if isinstance(image, (bytes, bytearray, memoryview)):
        raw = bytes(image)
    else:
        raw = Path(image).read_bytes()
    if not raw:
        raise ValueError("图像字节为空，无法取指纹")
    return _digest(raw)


@dataclass
class DistCache:
    """一个缓存目录的句柄：内存索引 + 待写缓冲 + 分片落盘。

    白话：把一堆 parquet 小文件当成一本可查的账来看。账本只往后写、不改旧页，
    `read_only=True` 时更是只许看不许写（离线伪标包就该这样，防止把别人的产标改脏）。
    """

    root: str | Path = "bench/teacher_cache"
    read_only: bool = False
    index: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    pending: list[dict[str, Any]] = field(default_factory=list, repr=False)
    hits: int = 0
    misses: int = 0
    _loaded: bool = field(default=False, repr=False)

    # ── 载入 ────────────────────────────────────────────────────────────
    def load(self) -> "DistCache":
        """扫目录下所有 `*.parquet` 建索引（懒执行，重复调用只在首次真扫）。

        白话：第一次用时把账本全翻一遍、记住每行门牌号；之后再翻就不动它，新写入的行
        由内存里的待写缓冲直接顶上，所以刚存的东西马上就能查到。
        """
        if self._loaded:
            return self
        directory = Path(self.root)
        rows: list[dict[str, Any]] = []
        if directory.is_dir():
            for path in sorted(directory.glob("*.parquet")):  # 名字排序 → 后写的分片覆盖先写的
                rows.extend(self._read_parquet(path))
        for row in rows:
            key = row.get("key")
            if key:
                self.index[str(key)] = row
        self._loaded = True
        return self

    @staticmethod
    def _read_parquet(path: Path) -> list[dict[str, Any]]:
        """读一个 parquet 成 dict 行列表；缺列在调用侧按缺省值补，不让整包报废。"""
        table = pq.read_table(path)
        columns = table.column_names
        rows = [{col: table.column(col)[i].as_py() for col in columns} for i in range(table.num_rows)]
        for row in rows:  # 列名集合由产标方决定，本层只硬要求 key 与 dist 两样
            row.setdefault("key", None)
            row.setdefault("model_id", "")
            row.setdefault("dist", None)
        return rows

    # ── 查 ──────────────────────────────────────────────────────────────
    def lookup(
        self,
        model_id: str,
        prompt: str,
        option_keys: Sequence[str],
        *,
        image_hash: str | None = None,
    ) -> dict[str, float] | None:
        """按请求三件套查分布：命中返回 {候选: 概率}，未命中返回 None 并记一笔 miss。

        白话：拿着"模型+问题+候选"去问有没有算过的结果。有就把当年那份原样交出来，
        没有就说一声"没查到"，让问的人自己去算——本层绝不猜一个数糊弄。
        """
        key = make_key(model_id, prompt, option_keys, image_hash=image_hash)
        return self.get(key)

    def get(self, key: str) -> dict[str, float] | None:
        """按键取分布；命中/未命中各记一笔，是"零前向"断言的直接数据源。

        白话：拿着门牌号去取那一行结果。取到了和没取到都要在册子上留一笔，事后查
        "重复请求到底有没有少走算力"就看这两个数。
        """
        self.load()
        row = self.index.get(key)
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return _as_dist(row.get("dist"))

    def __contains__(self, key: str) -> bool:
        """`key in cache` 的读法（测试与教师按这个口径问，不额外记账）。"""
        self.load()
        return key in self.index

    def __len__(self) -> int:
        """已索引的条目数（含尚未落盘、但已在内存里的待写行）。"""
        self.load()
        return len(self.index) + len([p for p in self.pending if p["key"] not in self.index])

    @property
    def stats(self) -> dict[str, int]:
        """命中/未命中/待写计数快照（写进 run 的 metrics 行用）。

        白话：一句话交代"查中几次、查空几次、草稿本上还压着几行没誊"，方便事后核账。
        """
        self.load()
        return {"hits": self.hits, "misses": self.misses, "pending": len(self.pending), "entries": len(self.index)}

    # ── 写 ──────────────────────────────────────────────────────────────
    def record(
        self,
        *,
        model_id: str,
        prompt: str,
        option_keys: Sequence[str],
        dist: dict[str, float],
        image_hash: str | None = None,
        source: str = "",
        meta: dict[str, Any] | None = None,
    ) -> str:
        """把一次打分结果存进缓冲并返回它的键；键由三件套现算，调用方不必自己拼。

        白话：算完一题就把"题目 + 答案份额"记进草稿本，同时告诉你是哪一页门牌号。
        记下来的东西马上能查到（内存里就有），但还没落盘——落盘要等 `flush`。
        """
        _reject_if_read_only(self)
        key = make_key(model_id, prompt, option_keys, image_hash=image_hash)
        row = {
            "key": key,
            "model_id": model_id,
            "prompt": prompt,
            "option_keys": json.dumps(sorted(str(k) for k in option_keys), ensure_ascii=False),
            "dist": json.dumps(_normalize_dist(dist), ensure_ascii=False),
            "image_hash": image_hash or "",
            "source": source,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "meta": json.dumps(meta or {}, ensure_ascii=False),
        }
        self._stage(row)
        return key

    def put(self, key: str, model_id: str, dist: dict[str, float], **extra: Any) -> str:
        """低层写入：键已由外部算好的行（伪标包合并、跨机搬运）直接进缓冲。

        白话：门牌号你已经写好了，我就只管把这一行抄进草稿本，不再重算牌号。
        """
        _reject_if_read_only(self)
        row: dict[str, Any] = {col: extra.get(col, "") for col in COLUMNS}
        row.update({"key": key, "model_id": model_id, "dist": json.dumps(_normalize_dist(dist), ensure_ascii=False)})
        row.update(extra)
        self._stage(row)
        return key

    def _stage(self, row: dict[str, Any]) -> None:
        """把一行放进待写缓冲并立刻可见（同进程内"写完就能查到"靠这一步）。"""
        self.pending.append(row)
        self.index[str(row["key"])] = row
        self._loaded = True

    def flush(self) -> int:
        """把缓冲写成一个新分片文件，返回写出行数（空缓冲直接返回 0，不造空文件）。

        白话：草稿本上的账攒够了就誊成一页新账本，先誊到临时纸、再一次性替换，
        这样别人来翻账时要么看见完整一页、要么什么都没看见，绝不会翻到半张纸。
        """
        if not self.pending:
            return 0
        directory = Path(self.root)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d%H%M%S")
        name = f"shard-{stamp}-{os.getpid()}-{len(self.pending)}.parquet"
        tmp = directory / (name + ".tmp")
        # 列全部按字符串落表：分布与候选清单以 json 文本存，读侧解码，避免跨语言的类型坑
        arrays = {col: pa.array([str(row.get(col, "")) for row in self.pending], type=pa.string()) for col in COLUMNS}
        pq.write_table(pa.table(arrays), tmp)
        os.replace(tmp, directory / name)  # 原子改名：读者不会看到写到一半的文件
        written = len(self.pending)
        self.pending.clear()
        return written

    def merge(self, other_root: str | Path) -> int:
        """把另一个目录（例如外部产好的伪标包）并进本缓存，返回并进的新行数。

        白话：别人算好的一摞账本，按门牌号收进我们这本里；已有同牌号的就不动，
        免得把好不容易存下的结果被来历不明的一页顶掉。
        """
        self.load()
        added = 0
        directory = Path(other_root)
        paths: Iterable[Path] = sorted(directory.glob("*.parquet")) if directory.is_dir() else [directory]
        for path in paths:
            for row in self._read_parquet(path):
                key = row.get("key")
                if not key or key in self.index:
                    continue
                clean = {col: row.get(col, "") for col in COLUMNS}
                clean["key"] = key
                self._stage(clean)
                added += 1
        return added


def _reject_if_read_only(cache: DistCache) -> None:
    """只读缓存（离线包）拒写——错误消息带上目录名，方便调用方定位是哪一包。"""
    if cache.read_only:
        raise CacheMissError(f"缓存以只读方式打开（{cache.root}），拒绝写入新行")


def _normalize_dist(dist: dict[str, float]) -> dict[str, float]:
    """把分布的键按字典序固定、数值转 float（json 往返后与写入时逐字节一致）。"""
    return {str(k): float(dist[k]) for k in sorted(dist, key=str)}


def _as_dist(value: Any) -> dict[str, float] | None:
    """读侧解码：json 字符串或 dict 都收，解不出分布就返回 None（坏行不假装命中）。"""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict) or not value:
        return None
    return {str(k): float(v) for k, v in value.items()}
