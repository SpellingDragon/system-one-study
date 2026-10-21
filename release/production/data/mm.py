"""图文样本装配：MMBench 行 → 模型输入三件套（p2-08 A2）。

【做什么】
    把 registry 装好的图文决策行（question/options/image/answer）经两段式
    processor 配方编成 input_ids/pixel_values/image_grid_thw，附答案边车，
    评测与训练共用同一装配口。

【怎么做】
    先 apply_chat_template(tokenize=False) 渲染含 vision 占位的文本，再交
    processor(images, text) 一次完成占位展开（A1 探针 run 0224 实测配方）；
    格数账以 config 为准（image_pad=248056，非方图 smart_resize 自适应）。

【为什么】
    图输入口径必须与主干渲染一致，否则加/去图对照失真、视觉增益不可归因；
    曾试一站式 messages API（transformers 5.18 阻，kwargs 须入
    processor_kwargs）与 square-448 硬账（真图非方格，grid 自适应才对），
    皆被探针否掉。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

# —— 常量：来自 A1 探针（run 1006-p2-08-a1-multimodal-probe-0224），config 侧互证 ——
MM_MARK = {"vision_start": 248053, "vision_end": 248054, "image_pad": 248056, "video_pad": 248057}


def build_messages(image_ref: Any, question: str, options: list[str]) -> list[dict[str, Any]]:
    """白话：把图与题面拼成 chat 消息体（图在前、题在后，要求单字母作答）。

    :param image_ref: PIL.Image 或图片路径
    :param question: 题干文字
    :param options: 候选项文字列表（渲染进题面，答案仍按字母序对应）
    :returns: 单轮 user 消息列表，供 processor 模板渲染
    """
    opt_line = " ".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
    return [{"role": "user", "content": [
        {"type": "image", "image": image_ref} if not isinstance(image_ref, str) else {"type": "image", "image": Path(image_ref)},
        {"type": "text", "text": f"{question}\n{opt_line}\nAnswer with one letter."},
    ]}]


def encode_row(row: dict[str, Any], processor: Any, image_root: Path | str) -> dict[str, Any]:
    """白话：一行 MMBench 记录变一帖模型输入——两段式装配，交出编号/像素/网格三件套与答案边车。

    输入：row（含 question/options/image/answer/qtype）、processor（AutoProcessor 产物）、
    image_root（图片目录）。输出 dict 含 input_ids/pixel_values/image_grid_thw（processor
    原样键）、gold 字母、letter_keys、with_image 开关位可回退纯文本（加/去图对照复用同函数）。
    """
    root = Path(image_root)
    imgs = []
    if row.get("image") and (root / str(row["image"])).exists():
        from PIL import Image
        imgs = [Image.open(root / str(row["image"])).convert("RGB")]
    msgs = build_messages(imgs[0] if imgs else None, row["question"], row["options"])
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if imgs:
        enc = processor(images=imgs, text=text, return_tensors="pt")
    else:
        enc = processor(text=text, return_tensors="pt")
    letters = [chr(65 + i) for i in range(len(row["options"]))]
    gold = str(row.get("answer", "")).strip()
    if gold.isdigit():  # MMBench 数字下标答案 → 字母
        gold = letters[int(gold)]
    out = {"id": row.get("id"), "qtype": row.get("qtype", "choice"),
           "gold": gold, "letter_keys": letters, "has_image": bool(imgs)}
    for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        if k in enc:
            out[k] = enc[k]
    return out


def iter_subset(subset_dir: Path | str, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """白话：按行读出 registry 装好的 MMBench-CN 子集账本（index.jsonl），可截前 N 题。"""
    f = Path(subset_dir) / "index.jsonl"
    for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
        if limit is not None and i >= limit:
            return
        yield json.loads(line)
