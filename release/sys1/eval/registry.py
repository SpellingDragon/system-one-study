"""sys1/eval/registry.py — 评测集注册表：把【用哪些集、哪个版本、几道题】钉成一张能复跑的清单。

【做什么】
    这里是六轴评测全部数据来源的**唯一登记处**：每个集记着来源种类（git / archive / HF / 魔搭 /
    合成）、pin（提交短码 + 实测解析出的全长散列）、分割（test/train/dev）、题量种子与采样口径、
    它服务哪条轴。对外只交三件事：`export_versions()` 交版本清单（报告附录直接抄，PRODUCTION §6-2）、
    `fetch` 子命令一键把数据拉到盘上并装配成评测能吃的行（一行一个 `{id, task, sample}` 信封，
    与 P1 预测出口同一种形状）、`needle_plan()` 交合成长文针的固定档位与种子。所有真实下载字节量、
    文件数、样本数、逐题型条数（choice/noul/score）与 sha256 都落 `bench/eval_data/manifest.json`，
    下游（p2-04 双基线、p2-07/08/09 三扩展、p2-11 verifier、p2-13 NPU 评测）一律从这张清单取数。

【怎么做】
    ① 注册表是模块级常量 `REGISTRY: {id: Pin}`。三件套的 pin 由变更文档定死（Intern-Decision
    @`2f81580`、jevbench@`7ce310c7`、typed-decisions@`f7a2487e`），并把本次实测解析出的全长散列一并
    写死——短码会被仓库历史里的同名对象撞车，全长才叫 pin。
    ② 拉取按来源分派六条路（第六条是 p2-09 的派生路：不下载，按已 pin 的副本现推）：git 路走 `init + fetch --depth 1 <full-sha> + checkout FETCH_HEAD`；
    本机实测 git 协议对 github.com 直连被 TLS 掐断，于是同一份内容改走 archive 路——下载 URL 里
    直接带那串全长散列（`codeload.github.com/<repo>/tar.gz/<full-sha>`，实测 200 且真取到 24MB/4.6MB），
    按内容取版本与浅检等价，落盘留 `_source.tar.gz` 原包与散列可复验。HF 路走
    `snapshot_download(endpoint=...)`，官方端点不通就换镜像端点（实测靠 hf-mirror 真取到 824881 字节）；
    魔搭路走 `modelscope.snapshot_download(repo_type="dataset")`；合成路不下载，只按 seed 产出确定的
    针位计划。每条路下载完立刻量字节（目录增量 / 归档包体积 / git 对象包体积）、算 sha256、记解析结果。
    ③ 装配把原始形态折成统一信封：一题一条（P1 契约一行一题），候选写进 criteria、档位键重写成
    `"0".."n-1"`、非真即假键固定 `"false"/"true"`，人工份额原样带过（多人投票的分布不摊平），并给
    每条写上 `qtype` 与 `task`——下游按【题型 × 候选数】分桶全靠这两个字段。含图的集（MMBench-CN）
    自持副本写成【子集 parquet + 逐行索引 + 图文件】三件，因为把图 base64 进 jsonl 会撑到百兆级；
    子集一律固定 seed、题量下限 `MIN_SUBSET`。
    ④ 幂等靠清单：同一 id 若 pin 的散列与盘上记录一致且文件齐，就报 cached 不重复下载；`--force`
    才重来。失败不静默：`fetch` 返回 {ok, failed}，CLI 把失败集逐个列名并以非零码退出。

【为什么】
    被否方案一：各域自己写拉数脚本（p2-04 拉 typed、p2-09 拉 CMMLU）——同一份数据在两个域里是两个
    版本，跨域数字加不出总分，双基线对照表的第一前提（同分母）当场失效；故 pin 只此一处。
    被否方案二：只记短码不记全长散列、也不记字节与条数——看着像同一个提交不等于同一个提交，且
    无法证明真下过。本次开工实测：github.com 的 git 通道 TLS 被掐、huggingface.co 不可达、ModelScope
    查无 Intern-Decision（API 回不存在的数据集、搜索 TotalCount=0）；若不留 resolved 散列与实测条数，
    可得性主张就只是许愿。
    被否方案三：把固定 seed 子集当全量报数——子集与全量分布不同，必须靠 `sampler` 字段与 `MIN_SUBSET`
    下限把口径写在清单里，报告脚注据此披露。
    被否方案四：网络受阻时用合成数据或离线旧副本顶替真实下载——这是上一阶段被点名的假绿形态（R14：
    外部资产可得性必须实测），故每条下载路径都留真实字节量，拉不到就报失败集并退出非零。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

#: 三件套 pin（变更文档定死；改这里=改契约，须同步 spec 与报告脚注）
PIN_INTERN = "2f81580"
PIN_JEV = "7ce310c7"
PIN_TYPED = "f7a2487e"
#: 本次实测解析出的全长散列（api.github.com 与 HF revision 接口各查一次得到，防短码撞车）
FULL_INTERN = "2f815802058b2464144218859ea9221c1bc0d2a8"
FULL_JEV = "7ce310c7262ed49cc85853339a8a42459298e3f3"
FULL_TYPED = "f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8"

#: 本机实测的可达性兜底（2026-10-05 复测）：git 协议对 github.com 直连被 TLS 掐断
#: （`RPC failed; curl 35 LibreSSL SSL_connect: SSL_ERROR_SYSCALL`），huggingface.co 不可达；
#: 于是 git 内容改走 codeload 源码归档（URL 里带全长提交散列=按内容钉版本，实测 200），
#: HF 走镜像端点，魔搭境内直连可用。归档里的散列就是 pin，与浅检等价且可用包散列复验。
GH_GIT_MIRROR = "https://gh-proxy.com/"
CODELOAD_ARCHIVE = "https://codeload.github.com/{repo}/tar.gz/{sha}"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
ARCHIVE_TIMEOUT = 900                 # 单包下载上限（秒）：24MB 实测 <25s，留足抖动

MIN_SUBSET = 200                      # 子集自持副本的题量下限（design §子集策略）
#: 中文决策信封的转写口径版号（规则住在 sys1/eval/chinese.py；两侧不一致即有用例判红）
ZH_DECISION_VERSION = "zh_decision_v1"
NEEDLE_SEED = 20261005                # 合成针的固定 seed（档位/针数见 needle_plan）
NEEDLE_BUCKETS = (8192, 32768, 131072, 262144)
QTYPES = ("choice", "noul", "score")  # 与 sys1.data.schema 的三枚取值同源
#: 训练数据专用登记轴（D1）：不并进六轴评测表——训练复用评测题面=训测同集假分（p2-05 教训）
TRAIN_AXIS = "train"
#: 中文 train 语料的自持上限（D3）：上游 CLUE train 档盘上实测 tnews 53360 行 / ocnli 50437 行
#: （`pyarrow.parquet.ParquetFile(...).metadata.num_rows`）：本域按固定 seed 只留这一段
#: （副本体积与装配内存都按这个数封顶）；要全量请显式改 sampler
ZH_TRAIN_SUBSET = MIN_SUBSET * 10

REPO_ROOT = Path(__file__).resolve().parents[2]        # release/（P2 主场根目录）
DATA_DIR = REPO_ROOT / "bench" / "eval_data"           # gitignored：数据产物落点
RAW_DIR_NAME = "raw"
ASSEMBLED_DIR_NAME = "assembled"
MANIFEST_NAME = "manifest.json"


class FetchError(RuntimeError):
    """某个集拉不下来时抛出（消息带原因）；`fetch` 兜住它、汇总进 failed 而不是半路崩。"""


class AssembleError(RuntimeError):
    """原始数据在盘上但装不出合法信封时抛出——这说明来源形态变了，比拉不到更该停下来。"""


@dataclass(frozen=True)
class Pin:
    """一个评测集的登记项：来源、pin、分割、种子、采样口径、服务轴、装配器名字。

    字段全为可 json 化的简单值，注册表因此能原样导出成报告用的清单。`sources` 是**按优先级**
    排列的候选来源（镜像兜底就写在登记项上，而不是藏在 if 分支里），`patterns` 是给下载器的
    文件白名单（把只取那一档 zip 这种省钱决定显式记下来，复跑者一眼能看到）。
    """

    id: str
    kind: str                                             # git | archive | hf | modelscope | synthetic
    repo: str                                             # org/name 形式的仓库标识
    revision: str                                         # pin 的短码（报告口径）
    split: str                                            # test / train / dev / n/a（合成）
    full: str = ""                                        # 实测解析出的全长散列
    seed: int | None = None
    sampler: str = "full"                                 # full | seeded-subset:n | needle:...
    qtypes: tuple[str, ...] = QTYPES
    axis: str = "quality"
    sources: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    assembler: str = ""
    note: str = ""
    derived_from: str = ""      # kind="derived" 时填：题面原件的集 id（零流量，按它现推）

    def as_dict(self) -> dict[str, Any]:
        """摊成清单条目（登记侧的字段；下载侧的实测字段由 `export_versions` 合并）。

    白话：把登记时写死的那几样——叫什么、从哪种来源来、钉在哪个版本、取哪一档、随机种子多少、
    服务哪条轴——原样摊平成一个字典，好让导出与报告接口直接取用。这里只搬运不加工：真拉了多少
    字节、装出多少条题是账本里的事实，由读账本那一层并进来，两处各管各的才不会互相改写。
    """
        return {
            "id": self.id, "source": self.kind, "repo": self.repo, "revision": self.revision,
            "revision_full": self.full, "split": self.split, "seed": self.seed,
            "sampler": self.sampler, "qtypes": list(self.qtypes), "axis": self.axis,
            "sources": list(self.sources), "patterns": list(self.patterns),
            "assembler": self.assembler, "note": self.note,
            "derived_from": self.derived_from,
        }


REGISTRY: dict[str, Pin] = {
    # ── 三件套（质量轴与把握轴的单一分母）────────────────────────────────────
    "typed-decisions": Pin(
        id="typed-decisions", kind="hf", repo="LocalLLaMA/typed-decisions", revision=PIN_TYPED,
        split="test", full=FULL_TYPED, sampler="full:test", axis="quality",
        sources=(HF_MIRROR_ENDPOINT, "https://huggingface.co"), patterns=("all/*",),
        assembler="typed",
        note="400 案例×5 问=2000 决策；卡面 Uniform 行 KL/TV/Brier=0.444/0.381/0.238 的锚点集。"
             "实测经 hf-mirror 真取 824881 字节（test 222140 + train 598824，逐件 sha256 入账）。"
             "两路候选都探过：① 本地工作区只有打分器与示例代码（StartLux-Decision/eval/"
             "typed_decisions.py、laya/examples/32_typed_decisions_workflow.py 等），没有任何数据件"
             "副本；② 魔搭镜像 theGAI/typed-decisions 存在（API 200），留作 HF 双端点皆失败时的兜底。"),
    "typed-decisions-train": Pin(
        id="typed-decisions-train", kind="hf", repo="LocalLLaMA/typed-decisions", revision=PIN_TYPED,
        split="train", full=FULL_TYPED, sampler="full:train", axis=TRAIN_AXIS,
        sources=(HF_MIRROR_ENDPOINT, "https://huggingface.co"), patterns=("all/train-*.parquet",),
        assembler="typed-train",
        note="typed-decisions 的 train 档训练登记项（D1 补登）：与 test 同 pin f7a2487e，上游单提交"
             "同时带 train/test 两档；train 实测 1200 案例×5 问=6000 决策，qtype "
             "choice/noul/score=1800/1800/2400 且逐条带人工份额（probabilities 齐全、求和≈1）。"
             "挂 train 轴、不进六轴评测表；train 案例号与 test 实测零交集（盘上 parquet 对拍），"
             "id 级隔离由 tests/test_registry.py 的 split_isolation 用例把关。"),
    "intern-decision": Pin(
        id="intern-decision", kind="archive", repo="InternLM/Intern-Decision", revision=PIN_INTERN,
        split="test", full=FULL_INTERN, sampler="full:test", axis="quality",
        sources=(CODELOAD_ARCHIVE.format(repo="InternLM/Intern-Decision", sha=FULL_INTERN),
                 GH_GIT_MIRROR + CODELOAD_ARCHIVE.format(repo="InternLM/Intern-Decision", sha=FULL_INTERN)),
        patterns=("benchmarks/*",), assembler="intern",
        note="accuracy-v1 各套件（含 typed_decisions 离线副本，作 HF 不可达兜底）+ known-distribution"
             "-pilot-v1（inputs+references 按 (id,field) 合装解析真值）。指定源 ModelScope 查无此资产"
             "（可复跑取证：GET /api/v1/datasets/InternLM/Intern-Decision 回 HTTP 404；"
             "GET /api/v1/datasets?Query=Intern-Decision 回 TotalCount=0、Data=[]），而本机 git 通道"
             "对 github.com 的 TLS 被掐（RPC failed; curl 35），故按全长散列取 codeload 源码归档；"
             "URL 内含散列即版本凭证，包体 sha256 入账可复验。装出 12447 条 = 上游 manifest 声明的 "
             "12351 决策 + known-dist 96 条，逐套件条数与 qtype 均可对账。train 分区探查"
             "（2026-10-05）：pin 归档内只有 test 档数据件，不登记 Intern train 项——"
             "取证与结论见 `INTERN_TRAIN_PROBE`（run 底账原样引用）。"),
    "jevbench": Pin(
        id="jevbench", kind="archive", repo="fstandhartinger/jevbench", revision=PIN_JEV,
        split="test", full=FULL_JEV, sampler="full", axis="calibration",
        sources=(CODELOAD_ARCHIVE.format(repo="fstandhartinger/jevbench", sha=FULL_JEV),
                 GH_GIT_MIRROR + CODELOAD_ARCHIVE.format(repo="fstandhartinger/jevbench", sha=FULL_JEV)),
        patterns=(), assembler="jev",
        note="JevBench 公开三层与上游打分器；hard 层的 ECE 走 10 桶口径（与 typed 的 15 桶禁止混报）。"),

    # ── 扩展轴（中文 / 长文 / 多模态；一律固定 seed 子集自持）─────────────────
    "cmmlu-subset": Pin(
        id="cmmlu-subset", kind="modelscope", repo="modelscope/cmmlu", revision="master", split="test",
        full="cmmlu_v1_0_1", seed=NEEDLE_SEED, sampler=f"seeded-subset:{MIN_SUBSET}",
        qtypes=("choice",), axis="quality", sources=("modelscope://modelscope/cmmlu",),
        patterns=("cmmlu_v1_0_1.zip",), assembler="cmmlu",
        note="中文多选题的题面原件；决策化信封已交派生集 cmmlu-decision（转写口径 zh_decision_v1），本集只留原件与版本凭证。实测该仓"
             "就一个 1.08MB 的 zip（含全学科 csv）。train 分割探查（2026-10-06，D3）：仓内三件与"
             "归档内部都只有 dev/test 两档，查无 train——不登记 CMMLU train 项、不拿 dev 改名凑数，"
             "取证见 `ZH_TRAIN_PROBE`（run 底账原样引用）。"),
    "clue-subset": Pin(
        id="clue-subset", kind="modelscope", repo="opencompass/clue", revision="master", split="validation",
        full="opencompass-clue-parquet", seed=NEEDLE_SEED, sampler=f"seeded-subset:{MIN_SUBSET * 2}",
        qtypes=("choice", "noul"), axis="quality", sources=("modelscope://opencompass/clue",),
        patterns=("tnews/validation-*.parquet", "ocnli/validation-*.parquet"), assembler="clue",
        note="tnews→choice(15 类)、ocnli→noul。取 validation 不取 test：实测 test 档 label 整列"
             "为 -1（官方隐藏答案——tnews 10000 行、ocnli 3000 行无一带值），没有真值的题面"
             "只能当语料不能当考卷。ModelScope 官方 clue 仓只带加载脚本无数据文件"
             "（实测文件清单只有 clue.py/dataset_infos.json/README），故取 opencompass/clue 的 parquet 镜像。"
             "决策化信封见派生集 clue-decision（zh_decision_v1）。"),
    # ── 中文决策信封（p2-09 派生集：零流量，按上面两份题面原件现推）─────────────
    "cmmlu-decision": Pin(
        id="cmmlu-decision", kind="derived", repo="sys1:eval.chinese", revision=ZH_DECISION_VERSION,
        split="test", full="cmmlu_v1_0_1", seed=NEEDLE_SEED, sampler="derived:cmmlu-subset",
        qtypes=("choice",), axis="quality", sources=(f"derived://{ZH_DECISION_VERSION}/cmmlu-subset",),
        patterns=(), assembler="zh-cmmlu", derived_from="cmmlu-subset",
        note="CMMLU 200 题决策化成 choice 信封（k=4，criteria=A/B/C/D 选项文本，问法换中文知识问法）。"
             "转写口径见 `sys1/eval/chinese.py`；题面原件与版本凭证都在 cmmlu-subset 那一格，本集"
             "不重复下载、不另钉版本。"),
    "clue-decision": Pin(
        id="clue-decision", kind="derived", repo="sys1:eval.chinese", revision=ZH_DECISION_VERSION,
        split="validation", full="opencompass-clue-parquet", seed=NEEDLE_SEED,
        sampler="derived:clue-subset", qtypes=("choice", "noul"), axis="quality",
        sources=(f"derived://{ZH_DECISION_VERSION}/clue-subset",), patterns=(), assembler="zh-clue",
        derived_from="clue-subset",
        note="CLUE 400 题决策化：tnews→choice(15 类)、ocnli→noul(蕴含=true，中立/矛盾=false)。"
             "候选文字沿用源 parquet 自带的类别码 \"100\"..\"116\"（盘上没给码↔类别名对照，"
             "档位随原件取 validation。"),
    # ── 中文 train 语料（D3 · p2-09 隐患裁决 C5 前置闸之二：训练档与考卷必须两格各占一行）──
    "clue-train-subset": Pin(
        id="clue-train-subset", kind="modelscope", repo="opencompass/clue", revision="master",
        split="train", full="opencompass-clue-parquet", seed=NEEDLE_SEED,
        sampler=f"seeded-subset:{ZH_TRAIN_SUBSET}", qtypes=("choice", "noul"), axis=TRAIN_AXIS,
        sources=("modelscope://opencompass/clue",),
        patterns=("tnews/train-*.parquet", "ocnli/train-*.parquet"), assembler="clue-train",
        note="CLUE 题面原件的 **train 档**（D3 补登）：上游确有三个档位——实测仓内 "
             "tnews/train-00000-of-00001.parquet 3399930 字节、ocnli/train-00000-of-00001.parquet "
             "2440405 字节（`HubApi.get_dataset_files` 列举）；盘上装出实测 tnews 53360 行、"
             "ocnli 50437 行（`pyarrow.parquet.ParquetFile(f).metadata.num_rows`）。只取"
             f"固定 seed 的 {ZH_TRAIN_SUBSET} 条自持（原件只当语料凭据，绝不当考卷），决策化交"
             "派生集 clue-train-decision。与 validation 那两份（clue-subset/clue-decision）分属两"
             "轴，id 里的档位段（train/validation）天然互斥，隔离由 split_isolation 用例把关。"
             "口径提示（读底账前先说明白，免得把两格当成两批题）：原件档那格的 qtype 是**折法提示**"
             "——`_assemble_clue` 只在 label 取值 <=2 时才标 noul，而 ocnli 原件在盘上给出 0/1/2 三值，"
             "于是 2000 条全记成 choice（1008 条 ocnli + 992 条 tnews）；真正按折法落题型的是决策档那一格"
             "（noul 1008 / choice 992，与原件的任务分布逐位对上，两档 id 集合实测全等）。"
             "三口一致以决策档为准：pin.qtypes=(choice,noul) ⊇ 决策档实出题型 ⊇ manifest.qtype_counts。"),
    "clue-train-decision": Pin(
        id="clue-train-decision", kind="derived", repo="sys1:eval.chinese",
        revision=ZH_DECISION_VERSION, split="train", full="opencompass-clue-parquet",
        seed=NEEDLE_SEED, sampler="derived:clue-train-subset",
        qtypes=("choice", "noul"), axis=TRAIN_AXIS,
        sources=(f"derived://{ZH_DECISION_VERSION}/clue-train-subset",), patterns=(),
        assembler="zh-clue-train", derived_from="clue-train-subset",
        note="CLUE train 档决策化出来的训练信封（零流量，按 clue-train-subset 那份原件现推）：折法"
             "与考卷那一格完全同规则（tnews→choice k=15、ocnli→noul），只有档位不同——同一套规则"
             "分别喂 train 与 validation，才谈得上「练的题与考的题不是同一批」。挂 train 轴、经 "
             "load_train_records() 消费，永不进六轴评测表（p2-05 训测同集假分的教训在这里同样生效）。"),
    "mmbench-cn-subset": Pin(
        id="mmbench-cn-subset", kind="modelscope", repo="lmms-lab/MMBench_CN", revision="master",
        split="dev", full="mmbench-cn-dev-parquet", seed=NEEDLE_SEED,
        sampler=f"seeded-subset:{MIN_SUBSET}", qtypes=("choice",), axis="multimodal",
        sources=("modelscope://lmms-lab/MMBench_CN",), patterns=("data/dev-*.parquet",),
        assembler="mmbench",
        note="图文决策；dev 档实测 96.7MB（test 141MB），只取 dev 且落 200 题自持副本（parquet+索引+图）。"
             "探测显示 hf-mirror 上 MMBench 系仓库需授权（401），故走魔搭。"),
    "longbench-zh": Pin(
        id="longbench-zh", kind="modelscope", repo="iic/longbench", revision="master", split="test",
        full="longbench-data-zip", seed=NEEDLE_SEED, sampler=f"seeded-subset:{MIN_SUBSET}",
        qtypes=("choice", "noul", "score"), axis="longctx", sources=("modelscope://iic/longbench",),
        patterns=("LongBench/data.zip",), assembler="longbench",
        note="长文中文任务真实分布；实测该仓 LongBench/data.zip 为 11.4MB。针（受控合成长文）"
             "由 needle-synthetic 另记一格，两者合起来构成长文轴有数可跑的证据。"),
    "needle-synthetic": Pin(
        id="needle-synthetic", kind="synthetic", repo="sys1:eval.registry.needle_plan",
        revision=str(NEEDLE_SEED), split="n/a", full=f"seed={NEEDLE_SEED}", seed=NEEDLE_SEED,
        sampler="needle:4buckets", qtypes=("choice",), axis="longctx", sources=(), patterns=(),
        assembler="needle",
        note="多针（颜色-数字对）插入长干扰文；档位 8K/32K/128K/256K 与每档针数由 seed 定，"
             "正文生成器归 p2-07（长文轨），本域只登记确定的针位表，不入库、零流量。"),
}

#: Intern-Decision train 分区探查结论（D1 · 2026-10-05，pinned-sha 归档内取证的原文摘录）：
#: 查无 train 数据档，故不登记 Intern train 项、只交 typed train——不硬造分区凑数，run 底账原样引用。
INTERN_TRAIN_PROBE = {
    "verdict": "上游无 train 分区：pin 2f81580 归档内只有 test 档数据，不登记 Intern train 项（只交 typed train）",
    "evidence": [
        "benchmarks/ 下数据件全为 test 档：accuracy-v1/{agnews,toolace,wildjailbreak,typed_decisions}/"
        "test.jsonl + jevbench/{easy,original,hard}.jsonl + known-distribution-pilot-v1/{inputs,"
        "references}.jsonl；整个归档 `find -iname 'train*'` 只命中训练脚本（scripts/train.sh、"
        "configs/training/qwen35.py、tests/check_training_runtime.py），零 train 数据件",
        "docs/DATA.md 原文：'Training data, private calibration/validation records, media, and their "
        "preparation pipelines are excluded.'（中文同段：不发布训练数据、内部校准/验证记录、图片及其处理流程）",
        "benchmarks/accuracy-v1/manifest.json 与 known-distribution-pilot-v1/manifest.json 均声明 "
        "\"training_ready\": false",
    ],
    "probed_at": "2026-10-05",
}

#: 中文集 train 分割探查结论（D3 · 2026-10-06）：CMMLU 查无 train 档，CLUE 有 train 档。
#: 三条取证原样抄自 `.probe_p203_d3_train.py` 的运行输出（一条命令一条事实，可复跑），
#: run 底账引用 verdict；不登记 CMMLU train 项，也不拿 dev 档改名冒充 train——dev 是少样本
#: 示例档，改名叫 train 就是给考卷同源的数据刷上"练习册"的标签，训测隔离当场失效。
#: 2026-10-07 复验注：那个脚本一度不在 release/ 下（结论退化成"抄来的话"，不可复跑即无效），
#: 已按同样三条命令重建脚本并复跑对拍，数字与下面 evidence 逐位相符——见 `reverified` 那一行。
ZH_TRAIN_PROBE = {
    "verdict": ("CMMLU 上游无 train 分割：modelscope/cmmlu 仓只有 README.md/cmmlu.py/"
                "cmmlu_v1_0_1.zip 三件，归档内部只有 dev/(67 个 csv) 与 test/(67 个 csv)，"
                "`train` 路径 0 个——故不登记 CMMLU train 项；中文 train 语料只交 "
                "CLUE(tnews/ocnli) train 档（已登记 clue-train-subset / clue-train-decision）"),
    "absent": ("cmmlu",),
    "present": ("clue-tnews", "clue-ocnli"),
    "evidence": [
        "盘上取证（零流量）：`python -c \"import zipfile; "
        "zipfile.ZipFile('bench/eval_data/raw/cmmlu-subset/cmmlu_v1_0_1.zip').namelist()\"` 的顶层目录"
        "只有 ['dev', 'test']；按后缀数 csv：dev=67 / test=67 / 含 train 的路径=0（按条目数是 "
        "dev=68 / test=69，多出来的是目录行本身——两个口径都记着，免得复跑数字对不上被当成编造）",
        "上游列举（魔搭 API）：`HubApi().get_dataset_files(repo_id='modelscope/cmmlu', "
        "revision='master', recursive=True)` 回 3 件：README.md 406 B、cmmlu.py 5066 B、"
        "cmmlu_v1_0_1.zip 1078656 B —— 仓内没有任何 train 数据件，也没有第二个归档",
        "上游列举（魔搭 API）：`get_dataset_files(repo_id='opencompass/clue', revision='master', "
        "recursive=True)` 回 49 件，其中 11 个任务带 train 档，本域要的两件是 tnews/"
        "train-00000-of-00001.parquet（3399930 B）与 ocnli/train-00000-of-00001.parquet"
        "（2440405 B）——CLUE 确有 train 档，据此登记并真拉装配",
    ],
    "reverified": ("2026-10-07 复跑 `release/.probe_p203_d3_train.py` 逐位对拍：盘上 cmmlu_v1_0_1.zip "
                   "顶层只有 dev(68 条目/67 csv)、test(69/67)，含 train 的路径 0 个；raw/clue-train-subset "
                   "下 tnews/train-00000-of-00001.parquet 3,399,930 B / 53,360 行，ocnli 同档 2,440,405 B "
                   " / 50,437 行；魔搭列举 modelscope/cmmlu files=3 train_files=0、opencompass/clue "
                   "files=49 train_files=11——三条全对得上，故结论与登记都不撤销"),
    "probed_at": "2026-10-06",
}

#: 中文 train 语料那一格（D3）：原件档 + 决策化档，都只挂 train 轴，不进六轴评测表
ZH_TRAIN_IDS_TUPLE = ("clue-train-subset", "clue-train-decision")

#: fetch 的分组开关（CLI 上的 --typed/--intern/... 在这里展开成 id，避免按钮与集名两处维护）
FLAG_TO_IDS = {
    "typed": ("typed-decisions",),
    "train": ("typed-decisions-train",),
    "intern": ("intern-decision",),
    "jev": ("jevbench",),
    "cn": ("cmmlu-subset", "clue-subset"),
    "zh": ("cmmlu-decision", "clue-decision"),
    "zh-train": ZH_TRAIN_IDS_TUPLE,          # D3 中文 train 语料（原件 + 决策信封两格）
    "mm": ("mmbench-cn-subset",),
    "long": ("longbench-zh",),
    "needle": ("needle-synthetic",),
}
EXTRA_IDS = ("cmmlu-subset", "clue-subset", "cmmlu-decision", "clue-decision",
             "mmbench-cn-subset", "longbench-zh", "needle-synthetic")
THREE_SHEET_IDS = ("typed-decisions", "intern-decision", "jevbench")
#: 训练侧登记集（D1 交 typed train，D3 补中文档）：fetch --all 一并带上，底账才完整
TRAIN_IDS = ("typed-decisions-train",) + ZH_TRAIN_IDS_TUPLE


# ---------------------------------------------------------------- 清单读写（实测事实的落点）
def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """读回 manifest.json（读不到回一份空骨架，不抛错——还没拉过是合法状态）。

    白话：这本账记着每个集真拉了多少字节、多少个文件、装出多少条题、散列是多少。第一次跑
    之前账是空的，这不算错，调用方据此知道还没下载；账本坏了（半行 json）也照样回空骨架，
    让复跑的人能从头重写，而不是被一本破账卡死。
    """
    p = Path(path) if path is not None else DATA_DIR / MANIFEST_NAME
    if not p.is_file():
        return {"sets": {}, "fetch": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"sets": {}, "fetch": {}}
    if not isinstance(data, dict):
        return {"sets": {}, "fetch": {}}
    data.setdefault("sets", {})
    data.setdefault("fetch", {})
    return data


def save_manifest(manifest: dict[str, Any], *, path: str | Path | None = None) -> Path:
    """把清单写盘（原子替换：先写临时文件再改名，中途断电也不会留下半本账）。

    白话：这本账是后面所有报告的底账，写坏一半比不写更糟——报告里会出现说不清拉没拉过的集。
    所以先落到同目录的临时文件，落成功才换掉正本；缩进留给人读，diff 时一眼看出改了哪行。
    """
    dst = Path(path) if path is not None else DATA_DIR / MANIFEST_NAME
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(dst)
    return dst


# ---------------------------------------------------------------- 版本导出（报告附录的真源）
def export_versions(*, manifest: dict[str, Any] | None = None,
                    ids: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """导出每个集的 id/revision/split/seed 清单，并合入盘上实测到的条数、字节与散列。

    注册表是应然（该用哪个版本），清单是实然（真拉到了多少）。这个函数把两份并成一列，
    未下载的集照样出现在结果里、实测字段为 None——报告因此能区分没注册与注册了还没拉。

    白话：交一张这批分数用的是什么数据的对照表：一集一行，写清它叫啥、从哪种来源来、钉在
    哪个版本、取哪一档、随机种子是多少、装出多少道题。没下过的集也照样占一行，只是那一行的
    条数与字节先空着——空着是实话，删掉这一行就变成假装齐全。
    """
    man = manifest if manifest is not None else load_manifest()
    rows: list[dict[str, Any]] = []
    for pid, pin in REGISTRY.items():
        if ids is not None and pid not in ids:
            continue
        row = pin.as_dict()
        got = (man.get("sets") or {}).get(pid) or {}
        row["status"] = got.get("status", "absent")
        row["resolved"] = got.get("resolved") or (pin.full if pin.kind != "synthetic" else None)
        # 字节有两个口径：真走网络的累计量（bytes_downloaded）与本轮量（bytes_this_run）；
        # 报告要的是前者，取不到才退回老账本里唯一的 bytes
        row["bytes"] = got.get("bytes_downloaded") or got.get("bytes")
        row["bytes_this_run"] = got.get("bytes_this_run")
        row["bytes_on_disk"] = got.get("bytes_on_disk")
        row["files"] = got.get("files")
        row["samples"] = got.get("samples")
        row["qtype_counts"] = got.get("qtype_counts")
        row["gold_coverage"] = got.get("gold_coverage")
        row["envelope_ready"] = got.get("envelope_ready")
        row["sha256"] = got.get("sha256") or got.get("assembled_sha256")
        row["local_path"] = got.get("raw_path") or got.get("assembled")
        row["assembled"] = got.get("assembled")
        row["error"] = got.get("error")
        rows.append(row)
    return rows


def format_versions_table(rows: list[dict[str, Any]] | None = None) -> str:
    """把版本清单压成等宽文本表（报告与 run notes 直接抄这张）。

    白话：一行一个集，把名字、来源、钉住的版本、取哪一档、随机种子、几条题、几个字节排齐；
    没拉到的条数与字节写成短横线而不是零。零会被读成这个集里没有题，短横线才读得出还没去拿。
    """
    rows = rows if rows is not None else export_versions()
    head = (f"{'id':<20} {'source':<11} {'revision':<12} {'split':<5} {'seed':<10} "
            f"{'axis':<12} {'samples':>8} {'bytes':>11} status")
    lines = [head, "-" * len(head)]
    for r in rows:
        seed = "-" if r.get("seed") is None else str(r["seed"])
        samples = "-" if r.get("samples") is None else str(r["samples"])
        nbytes = "-" if r.get("bytes") is None else f"{r['bytes']:,}"
        lines.append(f"{r['id']:<20} {r['source']:<11} {str(r['revision']):<12} {r['split']:<5} "
                     f"{seed:<10} {r['axis']:<12} {samples:>8} {nbytes:>11} {r.get('status', 'absent')}")
    return "\n".join(lines)


def registry_ids(*, groups: tuple[str, ...] = (), ids: tuple[str, ...] = (),
                 all_sets: bool = False) -> tuple[str, ...]:
    """把 CLI 开关还原成一组注册表 id（顺序稳定、去重、认不出来当场报错而不是忽略）。

    白话：命令行上 --typed、--intern 这些按钮按下去，这里负责翻译成到底要动哪几个集。
    拼错按钮名会停下来问，而不是当作没这回事悄悄少拉一个集——少拉一个集，报告里的数据
    就比声称的少一格，而这种漏在数字上看不出来。
    """
    picked: list[str] = []
    if all_sets:
        picked += list(THREE_SHEET_IDS) + list(EXTRA_IDS) + list(TRAIN_IDS)
    for flag in groups:
        if flag not in FLAG_TO_IDS:
            raise KeyError(f"未知分组 {flag!r}，可选：{sorted(FLAG_TO_IDS)}")
        picked += list(FLAG_TO_IDS[flag])
    for pid in ids:
        if pid not in REGISTRY:
            raise KeyError(f"未注册的集 id {pid!r}，可选：{sorted(REGISTRY)}")
        picked.append(pid)
    seen: dict[str, None] = {}
    for pid in picked:
        seen.setdefault(pid, None)
    return tuple(seen)


# ---------------------------------------------------------------- 下载（六条路 + 兜底候选源）
def _sha256(path: Path) -> str:
    """算一个文件的 sha256（分块读，百兆级的大文件也不会一次吞进内存）。"""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_snapshot(root: Path) -> dict[str, int]:
    """列出目录下所有文件的相对路径与大小（用来算这次真下了多少字节）。"""
    if not root.exists():
        return {}
    return {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*") if p.is_file()}


def _git_head(cwd: Path) -> str | None:
    """取该工作树当前检出的提交散列；拿不到就回 None（不是 git 树或 git 命令缺失）。"""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd), capture_output=True,
                             text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout or "").strip() or None


def _run_git(args: list[str], cwd: Path, timeout: int = 600) -> str:
    """跑一条 git 命令；非零退出即抛 FetchError（消息带上命令与最后一行 stderr）。"""
    proc = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise FetchError(f"git {' '.join(args)} 退出码 {proc.returncode}："
                         f"{(tail[-1] if tail else '')[:220]}")
    return proc.stdout or ""


def _pack_bytes(dst: Path) -> int:
    """从 git 自己的账（count-objects -v）取对象包字节数——浅检的真实流量就是这个。"""
    try:
        out = _run_git(["count-objects", "-v"], dst, timeout=30)
    except FetchError:
        return sum(_dir_snapshot(dst).values())
    kb = 0
    for line in out.splitlines():
        if line.startswith("size-pack:"):
            kb = int(float(line.split(":", 1)[1].strip()))
            break
    return kb * 1024


def _fetch_git(pin: Pin, dst: Path) -> dict[str, Any]:
    """git 路：只钉住那一个提交做浅检（候选源不通逐个换），回实测信息字典。

    白话：与其把整条仓库历史拖回家，不如只要那一次提交的样子——先空手建一个本地仓，
    把远端挂上，只把那一个提交取下来，再摆成能读的文件。第一个远端地址连不上就换下一个
    （本机实测 github 直连的 git 通道被 TLS 掐断，故此路在当前网络下会失败并落到 archive 路）。
    取完核对一下检出的散列跟登记的是不是同一串，不是就当失败；最后量一下对象包多大，
    这个数就是这次真花掉的流量。
    """
    dst.mkdir(parents=True, exist_ok=True)
    if _git_head(dst) == pin.full:
        return {"resolved": pin.full, "bytes": _pack_bytes(dst), "cached": True,
                "files": len(_dir_snapshot(dst))}
    if not (dst / ".git" / "config").is_file():
        _run_git(["init", "-q"], dst)
    _run_git(["remote", "remove", "origin"], dst)      # 复跑先摘干净，避免挂着失败的源
    errors: list[str] = []
    for url in pin.sources:
        try:
            _run_git(["remote", "add", "origin", url], dst)
            _run_git(["fetch", "-q", "--depth", "1", "origin", pin.full], dst)
            _run_git(["checkout", "-q", "--force", "FETCH_HEAD"], dst)
            break
        except FetchError as exc:
            errors.append(f"{url} → {exc}")
            _run_git(["remote", "remove", "origin"], dst)
    else:
        raise FetchError(f"git 全部候选源均失败：{' | '.join(errors) or '无候选源'}")
    resolved = _git_head(dst) or ""
    if resolved != pin.full:
        raise FetchError(f"检出散列 {resolved} 与 pin {pin.full} 不符：拒绝把别的版本当这个集")
    return {"resolved": resolved, "bytes": _pack_bytes(dst), "cached": False,
            "files": len(_dir_snapshot(dst)), "source": pin.sources[0]}


def _archive_urls(pin: Pin) -> tuple[str, ...]:
    """把登记项的候选源整理成按全长散列取源码归档的 URL 列（顺序即优先级）。"""
    return tuple(pin.sources) or (CODELOAD_ARCHIVE.format(repo=pin.repo, sha=pin.full),)


def _fetch_archive(pin: Pin, dst: Path) -> dict[str, Any]:
    """archive 路：下载 pin 指定的那一个提交的源码归档并解包（候选源不通逐个换）。

    白话：本机的 git 通道被网络掐了，但同一份内容还有个按提交号打包的取法：把那一长串
    提交号写进取件地址，拿回来的就是那一次提交的全部内容——提交号写在地址里，取错版本一眼
    就能看出来。解包前先把这坨字节原样留着，算个散列记进账本；下次复跑看到散列对得上就直接
    用本地副本，不再白跑一趟网络。
    """
    marker = dst / ".pinned_sha"
    blob = dst / "_source.tar.gz"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == pin.full and blob.is_file():
        # 归档原包留在盘上：命中缓存时报的仍是当初真下载的那份字节（同版本复跑取较大者，不翻倍）
        return {"resolved": pin.full, "bytes": blob.stat().st_size,
                "cached": True, "files": len(_dir_snapshot(dst)),
                "sha256": {blob.name: _sha256(blob)}}
    dst.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    used = ""
    for url in _archive_urls(pin):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sys1-eval-registry"})
            with urllib.request.urlopen(req, timeout=ARCHIVE_TIMEOUT) as resp:  # noqa: S310
                payload = resp.read()                                       # 域固定、散列写在 URL 里
            if not payload:
                raise FetchError(f"{url} 返回空包")
            blob.write_bytes(payload)
            used = url
            break
        except (urllib.error.URLError, OSError, FetchError) as exc:
            errors.append(f"{url} → {type(exc).__name__}: {exc}")
    else:
        raise FetchError(f"归档全部候选源均失败：{' | '.join(errors) or '无候选源'}")
    with tarfile.open(blob, "r:gz") as tf:
        tf.extractall(dst, filter="data")        # 解包丢弃绝对路径/设备类条目（stdlib 的 data 过滤器）
    marker.write_text(pin.full + "\n", encoding="utf-8")
    return {"resolved": pin.full, "bytes": blob.stat().st_size, "cached": False,
            "files": len(_dir_snapshot(dst)), "source": used,
            "sha256": {blob.name: _sha256(blob)}}


def _fetch_hf(pin: Pin, dst: Path) -> dict[str, Any]:
    """HF 路：按 pin 的 revision 取白名单文件，一个端点不通就换下一个（本机实测靠镜像）。

    白话：数据在架子上，但这个网络到不了官方那层——于是先问登记在册的第一个端点，问不通就
    问第二个，两边拿回来的都是同一个版本号（那串长散列）的东西才算数。只要点名要的那几个
    文件，其余一概不碰，省流量也省得把无关版本混进副本。
    """
    from huggingface_hub import snapshot_download

    before = _dir_snapshot(dst)
    endpoints = list(pin.sources) or ["https://huggingface.co"]
    last_err: Exception | None = None
    for endpoint in endpoints:
        try:
            snapshot_download(pin.repo, repo_type="dataset", revision=pin.full,
                              local_dir=str(dst), allow_patterns=list(pin.patterns),
                              endpoint=endpoint)
        except Exception as exc:  # noqa: BLE001  网络/端点故障形态太多，候选源试到底
            last_err = exc
            continue
        after = _dir_snapshot(dst)
        gained = sum(v for k, v in after.items() if before.get(k) != v)
        picked = [k for k in after if _match_any(k, pin.patterns) and not k.startswith(".")]
        return {"resolved": pin.full, "bytes": gained, "cached": gained == 0,
                "files": len(picked), "endpoint": endpoint,
                "sha256": {k: _sha256(dst / k) for k in picked}}
    raise FetchError(f"HF 全部端点失败（试过 {endpoints}）：{type(last_err).__name__}: {last_err}")


def _fetch_modelscope(pin: Pin, dst: Path) -> dict[str, Any]:
    """魔搭路：按登记的仓库 id 取白名单文件（境内直连可用，本域扩展集的主来源）。

    白话：这个来源在国内不用改道，问一句就答。和上面一样，只取注册表里点名要的那几件——
    比如 CMMLU 只要那一个压缩包（一兆多点），不把整个仓的加载脚本和历史都搬回来。
    """
    from modelscope import snapshot_download as ms_download

    before = _dir_snapshot(dst)
    try:
        ms_download(repo_id=pin.repo, repo_type="dataset", allow_patterns=list(pin.patterns),
                    local_dir=str(dst))
    except Exception as exc:  # noqa: BLE001  与 HF 路同处置：兜住形态、报给上层换招
        raise FetchError(f"魔搭拉取失败 {pin.repo}：{type(exc).__name__}: {exc}") from exc
    after = _dir_snapshot(dst)
    gained = sum(v for k, v in after.items() if before.get(k) != v)
    picked = [k for k in after if _match_any(k, pin.patterns) and not k.startswith(".")]
    return {"resolved": pin.full, "bytes": gained, "cached": gained == 0, "files": len(picked),
            "sha256": {k: _sha256(dst / k) for k in picked}}


def _fetch_synthetic(pin: Pin, dst: Path) -> dict[str, Any]:
    """合成路不下载：按 seed 产出确定的针位计划落盘（字节只算这份计划本身）。

    白话：这一格的数据不是从哪儿拿来，是按一个固定随机数捏出来的：种子一定，针插在哪、
    写什么颜色、几分，全都定死，谁复跑都得到同一份。所以这里零流量是事实，不是偷懒。
    """
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / "needle_plan.json"
    out.write_text(json.dumps(needle_plan(seed=pin.seed or NEEDLE_SEED), ensure_ascii=False,
                              indent=2) + "\n", encoding="utf-8")
    return {"resolved": f"seed={pin.seed}", "bytes": out.stat().st_size, "cached": False,
            "files": 1, "sha256": {out.name: _sha256(out)}}


def _fetch_derived(pin: Pin, dst: Path) -> dict[str, Any]:
    """派生路不下载：只按已 pin 的源副本现推，把"从哪份原件、哪一版推的"写成一页凭据落盘。

    白话：这一格的题不是从网上再拿一遍，而是从盘上那份题面原件誊出来的（中文子集已经 fetch
    并钉过版本，再下一遍只得多花流量、还多出第二个版本凭证谁也不认）。所以这里零字节是事实：
    要证明的只有两件事——原件在不在（不在就报命令让人先去 fetch --cn）、原件是哪一版（散列抄下来
    存进 derived_from.json）。装配规则变了不要指望这条路自动重跑：幂等照旧，改过口径请 --force。
    """
    src = REGISTRY.get(pin.derived_from)
    if src is None:
        raise FetchError(f"派生集 {pin.id} 的源副本 {pin.derived_from!r} 未登记，无从现推")
    origin = DATA_DIR / ASSEMBLED_DIR_NAME / src.id / f"{src.id}.jsonl"
    if not origin.is_file():
        raise FetchError(f"派生集 {pin.id} 的题面原件不在盘上：{origin}（先跑 "
                         f"`python -m sys1.eval.registry fetch --sets {src.id}`）")
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / "derived_from.json"
    out.write_text(json.dumps({"from": src.id, "revision": src.revision, "revision_full": src.full,
                               "split": src.split, "seed": src.seed, "assembler": pin.assembler,
                               "origin_bytes": origin.stat().st_size, "origin_sha256": _sha256(origin)},
                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"resolved": f"derived:{src.id}@{src.revision}", "bytes": 0, "cached": True, "files": 1,
            "endpoint": "local", "source": str(origin), "sha256": {out.name: _sha256(out)}}


FETCHERS = {"git": _fetch_git, "archive": _fetch_archive, "hf": _fetch_hf,
            "modelscope": _fetch_modelscope, "synthetic": _fetch_synthetic,
            "derived": _fetch_derived}


def _match_any(name: str, patterns: tuple[str, ...]) -> bool:
    """文件名是否落在登记的文件白名单里（fnmatch 语义；空白名单=全都算）。"""
    if not patterns:
        return True
    base = name.split("/")[-1]
    return any(fnmatch(name, p) or fnmatch(base, p) for p in patterns)


# ---------------------------------------------------------------- 装配（原始形态 → 统一信封）
def _envelope(rid: str, qtype: str, state: str, instructions: str, criteria: dict[str, str],
              target: dict[str, float], *, task: str = "") -> dict[str, Any]:
    """造一行评测信封 `{id, task, sample{state, questions, targets}}`（一题一行，P1 契约）。

    白话：不管题目原来长什么样，最后都誊成同一张表头：这道题的编号、它是哪一类、题面材料
    是什么、要问的那一句是什么、候选有哪几个及各叫什么名字、人工给各候选分了几成。下游
    读数、判分、分桶只认这一种形状；一题占一行是因为预测出口一次只处理一道题。
    """
    return {
        "id": rid,
        "task": task or qtype,
        "qtype": qtype,
        "sample": {
            "state": state,
            "questions": {"q": {"type": qtype, "instructions": instructions, "criteria": criteria}},
            "targets": {"q": target},
        },
    }


def _typed_keys(question: dict[str, Any]) -> tuple[list[str], list[str]]:
    """取一题的候选代号与说明两列（noul 固定 false/true，score 重写成 0..n-1）。"""
    qtype = question.get("type", "choice")
    criteria = question.get("criteria")
    if qtype == "noul" and not criteria:
        return ["false", "true"], ["不成立", "成立"]
    if isinstance(criteria, list):
        if qtype == "noul":
            return ["false", "true"], ([str(c) for c in criteria][:2] or ["不成立", "成立"])
        return [str(i) for i in range(len(criteria))], [str(c) for c in criteria]
    if isinstance(criteria, dict) and criteria:
        if qtype == "score":
            try:
                ordered = sorted(criteria, key=lambda k: float(k))
            except (TypeError, ValueError):
                ordered = list(criteria)
            return [str(i) for i in range(len(ordered))], [str(criteria[k]) for k in ordered]
        return [str(k) for k in criteria], [str(v) for v in criteria.values()]
    if qtype == "noul":
        return ["false", "true"], ["不成立", "成立"]
    raise AssembleError(f"题目认不出候选清单：type={qtype} criteria={type(criteria).__name__}")


def _noul_key(label: Any) -> Any:
    """把是非题的各种口语答案归到候选码 false/true 上（认不出的原样交回）。

    白话：有的集在答案里写 yes/no，有的写 true/false，还有的直接写 1/0；这一类题的候选码按
    契约只有 false 与 true 两个，所以先把口语说法换成码，免得把明明答对的一条记成"答案不在
    候选里"，再退成一份假均匀分布。认不出的仍原样交回，让上层按原口径记账，不做猜测。
    """
    text = str(label).strip().lower() if label is not None else ""
    if text in ("yes", "true", "1", "成立", "是"):
        return "true"
    if text in ("no", "false", "0", "不成立", "否"):
        return "false"
    return label


def _first_present(*values: Any) -> Any:
    """按给出的顺序取第一个"确实给了"的值；数字 0、布尔 false 都算给了，只有 None/空串算没给。

    白话：上游几个字段名轮着当答案用，顺手写 `a or b or c` 很自然，可答案本身就是 0 的时候
    （分级题最低那一档就是 0），Python 会把 0 当"没给"继续往后找，最后什么也没找到，这条就
    被记成了"说不准"。所以这里逐个看"是不是压根没写"，而不是看"好不好像是假"。
    """
    for v in values:
        if v is not None and v != "":
            return v
    return None


def _noul_align(keys: list[str], probs: dict[str, Any],
                label: Any) -> tuple[dict[str, Any], Any]:
    """是非题里把 yes/no 写法的份额与答案挪到候选码 false/true 上（键本就对得上就原样交回）。

    白话：同一类"要不要人工看一眼"的题，候选码按契约只有 false 与 true，可有的包在答案里
    写 yes/no。写的是 no、码要的是 false，对不上号时上层只会看见"答案不在候选里"，于是发一份
    五五开出去——那份原本是六四开的多人投票份额就这么没了，而且看不出来。这里先把说法换成码。
    """
    if set(str(k) for k in keys) != {"false", "true"}:
        return probs, label
    moved = ({str(_noul_key(k)): v for k, v in probs.items()}
             if any(str(k).strip().lower() in ("yes", "no") for k in probs) else probs)
    return moved, _noul_key(label)                      # 答案与份额两条腿都要挪，只挪一条仍会错位


def _target_from_probs(keys: list[str], probs: dict[str, Any], label: Any) -> dict[str, float]:
    """把人工份额按候选序折成 {代号: 份额}；份额缺失时退化成标签那一格为一。

    多人投票的分布照录不摊平（评测侧要的就是原始份额）；只有分布确实没给才用硬标签补一份
    独一份，且这种情况在条数与题型上仍看得清，不影响分母。
    """
    out: dict[str, float] = {}
    for k in keys:
        try:
            out[k] = float(probs.get(str(k), 0.0))
        except (TypeError, ValueError):
            out[k] = 0.0
    if sum(out.values()) <= 0.0:
        hit = str(label) if label is not None else ""
        if hit in out:
            out[hit] = 1.0
        elif keys:
            out = {k: 1.0 / len(keys) for k in keys}      # 标签不在候选里：均匀兜底并如实留痕
    return out


def _json_field(value: Any) -> dict[str, Any]:
    """上游常把结构体写成 json 字符串存一列；这里还原成 dict（本来就是 dict 就原样过）。"""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _maybe_json(value: Any) -> Any:
    """能还原成结构就还原（上游的 state 列常是 json 字符串），否则原样交回。"""
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "{[":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _state_text(state: Any) -> str:
    """题面材料统一收成一段字符串（结构体压 json；空值给占位——契约不接受空白 state）。"""
    if isinstance(state, str):
        return state.strip() or "(empty state)"
    if state is None:
        return "(empty state)"
    return json.dumps(state, ensure_ascii=False)


def _instructions_of(spec: dict[str, Any]) -> str:
    """题干：上游几种写法（instructions / question / query / prompt）都认，都没有给一句人话占位。"""
    text = spec.get("instructions") or spec.get("question") or spec.get("query") or spec.get("prompt")
    return str(text).strip() if text else "请就上述内容给出判断。"


def _assemble_typed(raw: Path, split: str = "test") -> list[dict[str, Any]]:
    """typed-decisions 的指定档 parquet → 逐决策信封（test 2000 决策 / train 6000 决策口径）。

    白话：原始文件里一行是一单生意，一单带五道小题。评测出口一次只吃一题，所以这里把
    每单拆成五行，逐行说清这题是哪一类、题干怎么说、候选有哪些、人工给各候选打了几成。
    题号用原单号加小题名拼出来，跨次复跑永远指同一道题，也不会跟别的集撞号。档名（test/train）
    由调用方点名，取错档等于拿复习题当考卷，所以文件名与档名必须对得上才开读。
    """
    import pyarrow.parquet as pq

    files = sorted(raw.glob(f"all/{split}-*.parquet")) or sorted(raw.rglob(f"{split}-*.parquet"))
    if not files:
        raise AssembleError(f"typed-decisions 的 {split} parquet 不在盘上：{raw}")
    records: list[dict[str, Any]] = []
    for path in files:
        for row in pq.read_table(path).to_pylist():
            questions = _json_field(row.get("questions"))
            golds = _json_field(row.get("gold"))
            state = _state_text(_maybe_json(row.get("state")))
            case_id = str(row.get("id"))
            for qid, q in questions.items():
                keys, texts = _typed_keys(q)
                gold = golds.get(qid) or {}
                probs = gold.get("probabilities") or {}
                target = _target_from_probs(keys, probs, gold.get("label"))
                criteria = {k: (str(t).strip() or k) for k, t in zip(keys, texts)}
                qtype = q.get("type", "choice")
                records.append(_envelope(f"{case_id}/{qid}", qtype, state, _instructions_of(q),
                                         criteria, target, task=f"typed-{qtype}"))
    return records


def _assemble_typed_train(raw: Path) -> list[dict[str, Any]]:
    """typed-decisions train 档 → 逐决策信封（训练消费口，形状与 test 信封逐字段一致）。

    白话：还是同一套拆题的手艺，只是从 train 档文件里取件；每条记录的档位名（split=train）
    由登记项统一盖戳（见 `_store_assembled`），训练侧从这个口拿题，评测侧从 quality 轴永远
    拿不到它们——两档的隔离断言由测试兜底，谁把考卷混进练习册都会当场红。
    """
    return _assemble_typed(raw, split="train")


def _rows_from_object(suite: str, obj: dict[str, Any], fallback_id: str) -> list[dict[str, Any]]:
    """把一条记录折成零或多条信封：带 questions 字典的拆多条，单题形态折一条。

    白话：不同来源的题目堆放方式就两种——一种是一单里塞好几问，另一种是一行一问。这里把两种
    都按同一张表头誊出来，编号一律写成套名加单号加问名，这样跨套也不会撞号，回溯时一眼看出
    它来自哪个集、哪一单、哪一问。
    """
    if obj.get("questions") and isinstance(obj["questions"], (dict, str)):
        questions = _json_field(obj["questions"])
        golds = _json_field(obj.get("gold") or obj.get("targets") or obj.get("answer") or {})
        state = _state_text(_maybe_json(obj.get("state") or obj.get("context")))
        out: list[dict[str, Any]] = []
        for qid, q in questions.items():
            if not isinstance(q, dict):
                continue
            keys, texts = _typed_keys(q)
            gold = _first_present(golds.get(qid)) or {}
            probs = gold.get("probabilities") if isinstance(gold, dict) else {}
            label = _first_present((gold or {}).get("label"), (gold or {}).get("answer"))
            probs, label = _noul_align(keys, probs or {}, label)
            target = _target_from_probs(keys, probs, label)
            criteria = {k: (str(t).strip() or k) for k, t in zip(keys, texts)}
            qtype = q.get("type", "choice")
            out.append(_envelope(f"{suite}:{obj.get('id', fallback_id)}:{qid}", qtype, state,
                                 _instructions_of(q), criteria, target, task=f"{suite}-{qtype}"))
        return out
    # 单题形态有两种摆法：题面字段直接摊在行上（agnews 旧式），或收在 question 里
    # （jevbench 公开层与 Intern-Decision 内嵌副本都是后者）——两种都按同一张表头誊。
    spec = obj.get("question") if isinstance(obj.get("question"), dict) else obj
    qtype = spec.get("type") or obj.get("type") or obj.get("qtype") or obj.get("task") or "choice"
    if qtype not in QTYPES:
        qtype = "noul" if qtype in ("bool", "nli", "yes_no") else "choice"
    options = (spec.get("criteria") or spec.get("options")
               or obj.get("criteria") or obj.get("options") or {})
    if isinstance(options, list):
        options = {str(i): str(v) for i, v in enumerate(options)}
    if not options and obj.get("labels"):
        options = {str(x): str(x) for x in obj["labels"]}
    if qtype == "noul" and not options:
        options = {"false": "不成立", "true": "成立"}
    keys, texts = _typed_keys({"type": qtype, "criteria": options})
    label = _first_present(spec.get("answer"), obj.get("answer"), obj.get("expected"),
                           obj.get("label"))
    probs = _json_field(_first_present(spec.get("targets"), obj.get("targets"),
                                       obj.get("probabilities")) or {})
    if isinstance(probs, dict):
        inner = _first_present(probs.get("probabilities"))
        if isinstance(inner, dict):                            # {label, probabilities} 整包写法
            probs, label = inner, _first_present(probs.get("label"), label)
    probs, label = _noul_align(keys, probs, label)
    target = _target_from_probs(keys, probs, label)
    criteria = {k: (str(t).strip() or k) for k, t in zip(keys, texts)}
    state = _maybe_json(obj.get("state") or obj.get("context") or obj.get("input"))
    return [_envelope(f"{suite}:{obj.get('id', fallback_id)}", qtype, _state_text(state),
                      _instructions_of(spec), criteria, target, task=f"{suite}-{qtype}")]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读一个 jsonl（跳过空行；遇到非 json 行抛 AssembleError 说清是哪一行）。"""
    out: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssembleError(f"{path.name} 第 {lineno} 行不是合法 json：{exc}") from exc
        if isinstance(obj, dict):
            out.append(obj)
    return out


#: 仓库里"数据放这儿"的目录名；jevbench 的 results/ 存的是各模型的跑分输出，不是题目
_DATA_DIR_NAMES = ("benchmarks", "datasets", "data")
_EXCLUDED_PARTS = {".git", "results", "runs", "tests", "docs", "scripts", "src", "assets",
                   "node_modules"}


def _suite_files(raw: Path, hint: str) -> list[Path]:
    """找出一个检出目录里"是题目"的 jsonl：只取数据目录下的，工程件与跑分结果一律不算。

    白话：一个源码包里到处是 .jsonl——模型的跑分输出、单元测试的夹具都这格式。把它们当题目
    装进评测集会造出一堆假题面，所以只认数据目录（benchmarks/、datasets/、data/）那一层里的；
    要是整个包就是平铺的一两层数据（老式仓库常见），才放宽到浅层文件。一条都没找到就明说。
    """
    found: list[Path] = []
    for path in sorted(raw.rglob("*.jsonl")):
        rel = path.relative_to(raw)
        if _EXCLUDED_PARTS & set(rel.parts):
            continue
        if any(part in _DATA_DIR_NAMES for part in rel.parts) or len(rel.parts) <= 2:
            found.append(path)
    if not found:
        raise AssembleError(f"{hint} 检出里没有 jsonl 数据件（来源形态变了？根目录：{raw}）")
    return found


def _assemble_intern(raw: Path) -> list[dict[str, Any]]:
    """Intern-Decision 检出 → 逐决策信封（按套件名分组，一题一行）。

    白话：这个仓把好几套题放在一起，每套一个 jsonl，形状与一单五题不同——它行行是一道题。
    这里照同一张表头重新誊一遍，让评测出口不用认两种形状。万一上游把字段改名了，这里不猜，
    当场停下来说清是哪个文件哪一行读不懂，免得装出一堆空题面的假数据。
    """
    records: list[dict[str, Any]] = []
    for path in _suite_files(raw, "Intern-Decision"):
        if _KNOWN_DIR in path.parts:                          # 这套单独走题面-真值合装
            continue
        suite = path.parent.name if path.parent != raw else path.stem
        for obj in _read_jsonl(path):
            records.extend(_rows_from_object(f"intern-{suite}", obj, path.name))
    for kroot in sorted(p for p in raw.rglob(_KNOWN_DIR) if p.is_dir()):
        if (kroot / "inputs.jsonl").is_file():
            records.extend(_assemble_known_dist(kroot))
    if not records:
        raise AssembleError(f"Intern-Decision 装出零条：{raw}")
    return records


#: 已知分布试点的目录名：答案不是人投票，而是按题面规则算出来的准确概率
_KNOWN_DIR = "known-distribution-pilot-v1"


def _assemble_known_dist(root: Path) -> list[dict[str, Any]]:
    """已知分布试点：inputs 的题面 + references 的解析真值分布 → 逐决策信封。

    白话：绝大多数评测集的标准答案是"人选的"，只有这一套是给每题写了条随机规则（公平骰子六面
    各六分之一、有放回的连击概率等），真值概率能算出来、算得准。它对参考答案另存一个文件，靠
    同一个编号对上——这里逐条对号入座；对不上就当场停，绝不装一条没有真值的题。把握度轴要量的
    就是"嘴上几成把握"配不配，拿到真值已知的题，这条轴才算量到了根上。
    """
    ref_path = root / "references.jsonl"
    if not (root / "inputs.jsonl").is_file() and not ref_path.is_file():
        raise AssembleError(f"已知分布试点缺件：{root}")
    gold: dict[tuple[str, str], dict[str, Any]] = {}
    for obj in _read_jsonl(ref_path):
        gold[(str(obj.get("id")), str(obj.get("field") or "decision"))] = obj
    out: list[dict[str, Any]] = []
    for obj in _read_jsonl(root / "inputs.jsonl"):
        rid = str(obj.get("id"))
        state = _state_text(_maybe_json(obj.get("state")))
        questions = _json_field(obj.get("questions") or {})
        for qid, q in questions.items():
            if not isinstance(q, dict):
                continue
            ref = gold.get((rid, qid))
            if ref is None:
                raise AssembleError(f"已知分布试点 {rid}:{qid} 在 references 里没有真值，拒装")
            keys, texts = _typed_keys(q)
            target = _target_from_probs(keys, ref.get("gold_probs") or {}, None)
            criteria = {k: (str(t).strip() or k) for k, t in zip(keys, texts)}
            out.append(_envelope(f"intern-known-dist:{rid}:{qid}", q.get("type") or "choice", state,
                                 _instructions_of(q), criteria, target, task="known-dist"))
    return out


def _assemble_jev(raw: Path) -> list[dict[str, Any]]:
    """jevbench 检出里的公开三层 → 逐决策信封（easy/original/hard 由 task 名前缀分桶）。

    白话：这个仓把公开层三份题集放在 datasets/public 下，文件名就是层名；它的 results 目录里
    堆的是各家模型的答卷，不是题，一步都不许碰。层名进题号与 task，hard 层的长政策题自然分
    成一桶，打分时和短题不混着算。
    """
    records: list[dict[str, Any]] = []
    for path in _suite_files(raw, "jevbench"):
        parent = path.parent.name
        tier = path.stem if parent in ("public", "datasets") else parent
        for obj in _read_jsonl(path):
            records.extend(_rows_from_object(f"jev-{tier}", obj, path.name))
    if not records:
        raise AssembleError(f"jevbench 装出零条：{raw}")
    return records


def _assemble_cmmlu(raw: Path) -> list[dict[str, Any]]:
    """CMMLU 压缩包 → 固定 seed 的 choice 子集自持副本（≥`MIN_SUBSET` 题，原字段照录）。

    白话：这是几十门学科的中文多选题，整包上万道。本域只需要一个**能复跑的子集**当登记凭据：
    把每题的学科、题干、四个候选与正确项原样抄下来，用固定随机数抽两百多条，谁复跑都抽到
    同一批。把题面改成决策格式是中文轨（p2-09）的活，这里不越界改内容，只保证抽法可复现。
    """
    rows = _read_csv_members(raw)
    if not rows:
        raise AssembleError(f"CMMLU 副本里没读到 csv：{raw}")
    # 包里 dev 是少样本示例、test 才是考卷；混着抽会让"split=test"这句登记话不成立
    test_rows = [r for r in rows if str(r.get("__split")) == "test"]
    if len(test_rows) >= MIN_SUBSET:
        rows = test_rows
    out: list[dict[str, Any]] = []
    for r in rows:
        subject = str(r.get("__subject") or "unknown")
        question = (r.get("question") or r.get("Question") or "").strip()
        options = {k: (r.get(k) or "").strip() for k in ("A", "B", "C", "D") if (r.get(k) or "").strip()}
        answer = (r.get("answer") or r.get("Answer") or "").strip()
        if len(question) < 2 or len(options) < 2 or not answer:
            continue
        out.append({"id": f"cmmlu:{subject}:{r.get('__split')}/{r.get('__row')}", "qtype": "choice",
                    "subject": subject, "question": question, "options": options, "answer": answer})
    if len(out) < MIN_SUBSET:
        raise AssembleError(f"CMMLU 可用题不足下限：{len(out)} < {MIN_SUBSET}")
    return _seeded_subset(out, MIN_SUBSET, NEEDLE_SEED)


def _assemble_clue(raw: Path, *, n: int = MIN_SUBSET * 2) -> list[dict[str, Any]]:
    """CLUE 镜像 parquet（tnews/ocnli）→ 固定 seed 子集，并逐条标好决策化的目标题型。

    白话：tnews 是十五类新闻文本（以后转成多选题），ocnli 是一句对不对（以后转成是非题）。
    这里只把原样字段抄进自持副本、逐条标上以后要转成哪一类，具体怎么写成信封交给中文轨；
    副本先钉住四百条起，免得日后抽得更小却还挂着同一句覆盖充分。`n` 是这一档的自持上限：
    考卷那一格取 400（validation），train 那一格取 `ZH_TRAIN_SUBSET`——同一套抄法只换个上限，
    才不至于出现"两份副本由两套规则抄出来"这种事。
    """
    import pyarrow.parquet as pq

    out: list[dict[str, Any]] = []
    skipped_unlabeled = 0
    for f in sorted(p for p in raw.rglob("*.parquet") if ".git" not in p.parts):
        rel = f.relative_to(raw).parts
        task = rel[-2] if len(rel) >= 2 else f.stem
        split_tag = f.stem.split("-")[0]                   # 文件名首段就是档位（validation-00000-…）
        rows = pq.read_table(f).to_pylist()
        space = sorted({str(r.get("label")) for r in rows if r.get("label") is not None})
        # 是非题按契约只有 false/true 两格：这一档真给出三个取值时就不是是非题，别硬塞
        hint = "noul" if (task in ("ocnli", "csl", "chid", "afqmc") and len(space) <= 2) else "choice"
        for i, row in enumerate(rows):
            item = {k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
                    for k, v in row.items()}
            if row.get("label") is None or str(row.get("label")) == "-1":
                skipped_unlabeled += 1                        # 隐藏答案的档：只能当语料，不当考卷
                continue
            item.update({"id": f"clue:{task}:{split_tag}:{i}", "task_name": task,
                         "qtype": hint, "label_space": space,
                         "gold": str(row.get("label"))})
            out.append(item)
    if not out:
        raise AssembleError(f"CLUE 副本里没有带人工答案的 parquet 行（无真值的档一律不收，"
                            f"跳过 {skipped_unlabeled} 行）：{raw}")
    return _seeded_subset(out, n, NEEDLE_SEED)


def _assemble_clue_train(raw: Path) -> list[dict[str, Any]]:
    """CLUE train 档（tnews/ocnli 的 train parquet）→ 题面原件自持副本（D3 补登）。

    白话：还是 `_assemble_clue` 那门抄题的手艺，只是从 train 那一份文件里取件、上限换成
    `ZH_TRAIN_SUBSET`；每条的 id 里带着档位段（train），与 validation 那份天然撞不上号。
    这一格交的是"题面长什么样"的原件凭据，能直接喂训练侧的决策信封在派生集那一格。
    """
    return _assemble_clue(raw, n=ZH_TRAIN_SUBSET)


def _assemble_longbench(raw: Path) -> list[dict[str, Any]]:
    """LongBench data.zip → 中文任务固定 seed 子集（长文轴的真实分布副本）。

    白话：整包十一兆，里面按任务切成一个个 jsonl，每个任务差不多两百条。这里挑出中文那几档
    （读解、摘要、检索类），原样抄成自持副本并标好以后按哪一类判分。长文针在另一格登记，
    两者一起构成长文轴有数可跑的证据。
    """
    zh_prefix = ("multifieldqa_zh", "dureader", "vcsum", "lsht", "passage_retrieval_zh", "trec_zh")
    out: list[dict[str, Any]] = []
    for member in _zip_jsonl_members(raw, ("data/",)):
        task = member["task"]
        if not task.startswith(zh_prefix):
            continue
        for i, obj in enumerate(member["rows"]):
            if not isinstance(obj, dict):
                continue
            item = dict(obj)
            # longctx 是轴名不是题型：选项/检索类能直接判分归 choice，开放问答与摘要
            # 本域不硬塞成三类之一（那是 p2-07/p2-09 的决策化活），显式标 untyped 让分桶数露馅
            qtype = "choice" if task.startswith(("passage_retrieval", "trec")) else "untyped"
            item.update({"id": f"longbench:{task}:{i}", "task_name": task, "qtype": qtype,
                         "axis": "longctx", "length_chars": len(str(obj.get("context", ""))),
                         "gold": _first_present(obj.get("answers"), obj.get("answer"),
                                               obj.get("label"))})
            out.append(item)
    if not out:
        raise AssembleError(f"LongBench 副本里没落到中文任务（期望前缀：{zh_prefix}）")
    return _seeded_subset(out, MIN_SUBSET, NEEDLE_SEED)


def _assemble_mmbench(raw: Path) -> dict[str, Any]:
    """MMBench-CN dev parquet → 固定 seed 的图文子集（子集 parquet + 逐行索引 + 图文件）。

    白话：整包九十多兆里大半是图片。抽两百题出来，把这三样一起留下：一份只含这二百题的
    parquet、一份给人读的索引（哪道题配哪张图、四个候选、正确项）、以及真图片本身。这样
    多模态轴离线也能复跑，不必每次都回去搬整包；图片名与题号对得上，事后要查也查得着。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = sorted(p for p in raw.rglob("*.parquet") if "dev" in p.name and ".git" not in p.parts)
    if not src:
        raise AssembleError(f"MMBench-CN 的 dev parquet 不在盘上：{raw}")
    rows = pq.read_table(src[0]).to_pylist()
    picked = _seeded_subset(list(range(len(rows))), MIN_SUBSET, NEEDLE_SEED)
    out_rows: list[dict[str, Any]] = []
    images: list[bytes] = []
    for i in picked:
        row = dict(rows[i])
        image = row.get("image")
        if isinstance(image, dict):
            image = image.get("bytes")
        name = f"{row.get('index', i)}.png"
        if isinstance(image, (bytes, bytearray)):
            images.append(bytes(image))
            row["image"] = name                       # 副本里图片以文件名引用，不内嵌大字节
        row["qtype"] = "choice"
        out_rows.append(row)
    if not out_rows:
        raise AssembleError(f"MMBench-CN dev parquet 抽不出题目行（列名变了？）：{src[0]}")
    return {"__subset_rows__": out_rows, "__images__": images,
            "__table__": pa.Table.from_pylist(out_rows)}


def _assemble_needle(raw: Path) -> list[dict[str, Any]]:
    """合成针：把针位计划展开成可核对的针清单（长干扰文正文由 p2-07 的生成器承载）。

    白话：这一格只负责该在哪几档长度、每档插几根针、针上写什么颜色几分这三件事，全部由
    一个固定随机数决定；把长干扰文真正拼出来是长文轨的活。所以这里交出的是一张确定的针位
    表——同一颗种子，任何人复跑都得到逐字相同的针，指不出第二个版本。
    """
    plan = needle_plan(seed=NEEDLE_SEED)
    colors = ["红", "橙", "黄", "绿", "青", "蓝", "紫", "粉", "灰", "黑"]
    rows: list[dict[str, Any]] = []
    for bucket in plan["buckets"]:
        rng = random.Random(bucket["seed"])
        for n in range(bucket["needles"]):
            rows.append({"id": f"needle:{bucket['ctx']}:{n}", "qtype": "choice",
                         "ctx": bucket["ctx"], "seed": bucket["seed"],
                         "pos_frac": round((n + 1) / (bucket["needles"] + 1), 4),
                         "axis": "longctx",
                         "needle_color": rng.choice(colors), "needle_value": rng.randint(100, 999)})
    return rows


def needle_plan(seed: int = NEEDLE_SEED) -> dict[str, Any]:
    """合成长文针的固定档位计划：四个上下文档，每档一个派生 seed 与针数。

    白话：把针插多长、每档几根写成一张不随机器、不随时间变的表：档位取 8K/32K/128K/256K，
    每档的种子由总种子和档号算出来，针数随档位阶梯递增。表里给的是确定数字，复跑同一颗种子
    必然得到同一张表，长文轴以后所有曲线都以此为准绳。
    """
    buckets = [{"ctx": ctx, "seed": (seed + 7919 * i) % (1 << 31), "needles": 2 ** (i + 1), "probes": 4}
               for i, ctx in enumerate(NEEDLE_BUCKETS)]
    return {"kind": "needle", "seed": seed, "buckets": buckets,
            "metric": "needle_recall(exact-string)", "generator": "p2-07 long-context"}


def _assemble_zh_cmmlu(raw: Path) -> list[dict[str, Any]]:
    """中文 CMMLU → 决策信封：规则与版号住在 p2-09 的 `sys1/eval/chinese.py`，这里只挂个名。"""
    from sys1.eval import chinese                           # 懒 import：chinese 反向 import 本模块

    return chinese.assemble_cmmlu_decision(raw)


def _assemble_zh_clue(raw: Path) -> list[dict[str, Any]]:
    """中文 CLUE → 决策信封（tnews 多选题 + ocnli 是非题）：同上，口径只在一处。"""
    from sys1.eval import chinese                           # 懒 import：避免模块级循环引用

    return chinese.assemble_clue_decision(raw)


def _assemble_zh_clue_train(raw: Path) -> list[dict[str, Any]]:
    """中文 CLUE **train 档** → 训练信封：折法住在 p2-09 的 `chinese.py`，这里只挂个名（D3）。"""
    from sys1.eval import chinese                           # 懒 import：与同族钩子一致

    return chinese.assemble_clue_train_decision(raw)


ASSEMBLERS = {"typed": _assemble_typed, "intern": _assemble_intern, "jev": _assemble_jev,
              "typed-train": _assemble_typed_train,
              "cmmlu": _assemble_cmmlu, "clue": _assemble_clue, "mmbench": _assemble_mmbench,
              "longbench": _assemble_longbench, "needle": _assemble_needle,
              "zh-cmmlu": _assemble_zh_cmmlu, "zh-clue": _assemble_zh_clue,
              "clue-train": _assemble_clue_train, "zh-clue-train": _assemble_zh_clue_train}


# ---------------------------------------------------------------- 装配的公共小件
def _seeded_subset(rows: list[Any], n: int, seed: int) -> list[Any]:
    """按固定 seed 抽最多 n 条（不足 n 条全留），返回顺序按原序，跨机器可复现。"""
    if len(rows) <= n:
        return list(rows)
    picked = random.Random(seed).sample(range(len(rows)), n)
    return [rows[i] for i in sorted(picked)]


def _read_csv_members(raw: Path) -> list[dict[str, Any]]:
    """把盘上的 csv（含 zip 包内的 csv）逐行读成带 __subject/__row/__split 的字典列表。"""
    sources: list[tuple[str, str, Any]] = []
    for path in sorted(raw.rglob("*.zip")):
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if not name.endswith(".csv"):
                    continue
                parts = Path(name).parts                     # 包内一层目录就是档位（dev/ test/）
                split = str(parts[-2]) if len(parts) >= 2 else "root"
                sources.append((split, Path(name).stem,
                                io.TextIOWrapper(zf.open(name), encoding="utf-8", errors="replace")))
    for path in sorted(raw.rglob("*.csv")):
        if path.is_file():
            split = path.parent.name or "root"
            sources.append((split, path.stem,
                            path.open(encoding="utf-8", newline="", errors="replace")))
    rows: list[dict[str, Any]] = []
    for split, subject, fh in sources:
        try:
            for i, rec in enumerate(csv.DictReader(fh)):
                item = {k: v for k, v in rec.items() if k}
                item.update({"__subject": subject, "__row": i, "__split": split})
                rows.append(item)
        except (csv.Error, UnicodeDecodeError):
            continue            # 单科读不动不拖垮整包：条数由最终清单兜底可见
        finally:
            fh.close()
    return rows


def _zip_jsonl_members(raw: Path, prefixes: tuple[str, ...]) -> list[dict[str, Any]]:
    """读 zip 里指定目录下的 jsonl，回 {task: 文件名, rows: [...]}（盘上已解压的也读）。"""
    out: list[dict[str, Any]] = []
    for path in sorted(raw.rglob("*.zip")):
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if not name.endswith(".jsonl") or not any(p in name for p in prefixes):
                    continue
                rows: list[dict[str, Any]] = []
                with io.TextIOWrapper(zf.open(name), encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict):
                            rows.append(obj)
                out.append({"task": Path(name).stem, "rows": rows})
    for path in sorted(raw.rglob("*.jsonl")):
        rel = str(path.relative_to(raw))
        if any(p in rel for p in prefixes):
            out.append({"task": path.stem, "rows": _read_jsonl(path)})
    return out


def _gold_coverage(records: list[Any]) -> dict[str, int]:
    """数一数这批副本里有多少条真带着人工答案（信封看份额合计，原始行看 answer/gold/label）。

    白话：登记一集"两百题"容易，容易被忽略的是里面有多少题其实没有答案——公开评测的 test 档
    常常把答案藏起来（CLUE 就是整档 -1）。没有答案的题只能喂着看，不能打分。所以把"带答案的
    条数"和"总条数"分开记：将来哪个下游拿这一集出分，先看这个数对不对得上，对不上就说明它在
    拿没答案的题算分，账本当场就能拦一道。
    """
    with_gold = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        sample = rec.get("sample")
        if isinstance(sample, dict):
            slots = (sample.get("targets") or {}).values()
            if any(any(float(v or 0.0) > 0.0 for v in t.values())
                   for t in slots if isinstance(t, dict)):
                with_gold += 1
            continue
        gold = rec.get("gold") if "gold" in rec else _first_present(rec.get("answer"),
                                                                    rec.get("label"),
                                                                    rec.get("expected"))
        if gold is not None and str(gold) not in ("", "-1") and gold is not False:
            with_gold += 1
    return {"with_gold": with_gold, "total": len(records)}


def _count_qtypes(records: list[Any]) -> dict[str, int]:
    """统计装配结果的逐题型条数（qtype 贯通的最小证据：choice/noul/score 各有几条）。"""
    counts = {q: 0 for q in QTYPES}
    other = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        q = rec.get("qtype") or rec.get("qtype_hint")
        if q in counts:
            counts[q] += 1
        else:
            other += 1
    counts["_other"] = other
    return counts


def _write_jsonl(rows: list[Any], dst: Path) -> int:
    """把装配结果写成 jsonl（一行一条），回写后的字节数；父目录自动建。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    dst.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    return dst.stat().st_size


def _write_subset(table: Any, rows: list[Any], images: list[bytes], out_dir: Path) -> dict[str, Any]:
    """含图集的落盘：子集 parquet + 索引 jsonl + images/ 目录，回字节与图数。"""
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_dir / "subset.parquet")
    index = [{"id": f"mmbench:{r.get('index', i)}", "question": r.get("question"),
              "options": {k: r.get(k) for k in ("A", "B", "C", "D") if r.get(k) is not None},
              "answer": r.get("answer"), "category": r.get("category"),
              "qtype": r.get("qtype") or r.get("qtype_hint"), "image": r.get("image")}
             for i, r in enumerate(rows)]
    idx_bytes = _write_jsonl(index, out_dir / "index.jsonl")
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)
    total_img = 0
    for pos, blob in enumerate(images):
        name = str(rows[pos].get("image") or f"{rows[pos].get('index', pos)}.png")
        (img_dir / name).write_bytes(blob)
        total_img += len(blob)
    return {"bytes": (out_dir / "subset.parquet").stat().st_size + idx_bytes + total_img,
            "images": len(images), "image_bytes": total_img,
            "sha256": _sha256(out_dir / "index.jsonl")}


def _store_assembled(pin: Pin, asm: Any, out_dir: Path) -> dict[str, Any]:
    """把装配产物落盘并回统计：普通集写 jsonl，含图集写 parquet 加索引加图。"""
    if isinstance(asm, dict) and "__subset_rows__" in asm:
        info = _write_subset(asm["__table__"], asm["__subset_rows__"], asm["__images__"], out_dir)
        sub_rows = asm["__subset_rows__"]
        return {"samples": len(sub_rows),
                "qtype_counts": _count_qtypes(sub_rows),
                "gold_coverage": _gold_coverage(sub_rows),
                "assembled": str(out_dir), "assembled_bytes": info["bytes"],
                "assembled_files": {"index": len(asm["__subset_rows__"]), "images": info["images"],
                                    "image_bytes": info["image_bytes"]},
                "assembled_sha256": info["sha256"], "envelope_ready": False}
    records = list(asm or [])
    for rec in records:   # 数据类记录显式带档位（D2：split 语义随信封走，登记侧是单一真源）
        if isinstance(rec, dict):
            rec["split"] = pin.split
    dst = out_dir / f"{pin.id}.jsonl"
    nbytes = _write_jsonl(records, dst)
    return {"samples": len(records), "qtype_counts": _count_qtypes(records),
            "gold_coverage": _gold_coverage(records),
            "assembled": str(out_dir), "assembled_bytes": nbytes,
            "assembled_sha256": _sha256(dst) if records else None,
            "envelope_ready": bool(records) and all(isinstance(r, dict) and "sample" in r
                                                   for r in records)}


# ---------------------------------------------------------------- fetch 主流程
def fetch(pids: tuple[str, ...] | list[str], *, data_dir: str | Path | None = None,
          force: bool = False, do_assemble: bool = True) -> dict[str, Any]:
    """按登记把点名的集拉到盘上并装配成信封；回 {ok, failed, manifest} 三段。

    每个集走下载 → 装配 → 记账三步：下载只认 pin 的那个版本；装配失败算这个集失败（不把
    半成品记成成功）；记账把 resolved 散列、真字节、文件数、样本数、逐题型条数与 sha256 写进
    `manifest.json`。已拉过且散列对得上就报 cached，装配也复用盘上副本，不重复走网络——除非 force。

    白话：一句话就是点名要的东西按版本号取回来，取回来还要拆成能直接用的样子，最后把花了
    多少流量、拿到多少条都写进账本。哪个集没到手就明明白白列在失败那一栏里，退出码也非零，
    绝不让少了一个集混成全都准备好了。
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    raw_root, asm_root = root / RAW_DIR_NAME, root / ASSEMBLED_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(root / MANIFEST_NAME)
    ok: dict[str, Any] = {}
    failed: dict[str, str] = {}
    for pid in pids:
        pin = REGISTRY.get(pid)
        if pin is None:
            failed[pid] = "未注册（registry 里没有这个 id）"
            continue
        dst = raw_root / pid
        try:
            prev = dict(manifest["sets"].get(pid) or {})
            got = FETCHERS[pin.kind](pin, dst)
            entry = prev
            this_run = int(got.get("bytes_this_run", got.get("bytes") or 0))
            # 流量累计取较大者：同一版本反复复跑不重复计费，首轮真下的量必须留在账上
            downloaded = max(int(prev.get("bytes_downloaded") or prev.get("bytes") or 0), this_run)
            entry.update({"status": "cached" if got.get("cached") else "fetched",  # 状态只认真走过网络没有
                          "resolved": got.get("resolved"), "bytes": downloaded,
                          "bytes_downloaded": downloaded, "bytes_this_run": this_run,
                          "bytes_on_disk": sum(_dir_snapshot(dst).values()) if dst.is_dir() else 0,
                          "files": got.get("files"), "endpoint": got.get("endpoint"),
                          "source_url": got.get("source"), "revision": pin.revision,
                          "revision_full": pin.full, "split": pin.split, "seed": pin.seed,
                          "sha256": got.get("sha256"), "raw_path": str(dst)})
            reuse = bool(got.get("cached")) and not force and prev.get("samples")  # --force 必重装配
            if do_assemble and pin.assembler and not reuse:
                entry.update(_store_assembled(pin, ASSEMBLERS[pin.assembler](dst), asm_root / pid))
            elif reuse:
                entry["assembled"] = prev.get("assembled")
            manifest["sets"][pid] = entry
            ok[pid] = entry
        except (FetchError, AssembleError, OSError, KeyError, ValueError) as exc:
            msg = f"{type(exc).__name__}: {exc}"
            failed[pid] = msg
            bad = dict(manifest["sets"].get(pid) or {})
            bad.update({"status": "failed", "error": msg, "revision": pin.revision,
                        "revision_full": pin.full, "split": pin.split, "seed": pin.seed})
            manifest["sets"][pid] = bad
    manifest["fetch"] = {"at": datetime.now().astimezone().isoformat(timespec="seconds"),
                         "requested": list(pids), "ok": sorted(ok), "failed": sorted(failed)}
    save_manifest(manifest, path=root / MANIFEST_NAME)
    return {"ok": ok, "failed": failed, "manifest": manifest}


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    """搭出注册表 CLI 的参数表（versions / fetch / list 三个子命令）。

    白话：把命令行能敲的选项摆清楚：想看清单就敲 versions，想拉数据就敲 fetch 并点名要哪几套
    （全要 --all，或 --typed/--intern/--jev/--cn/--mm/--long/--needle 单点）；还可以指定落盘
    位置、要不要重拉、装完要不要顺手拆成信封。一个开关都没按时默认只动三件套，省得空跑一趟网络。
    """
    parser = argparse.ArgumentParser(
        prog="python -m sys1.eval.registry",
        description="评测集注册表：版本清单导出与一键拉取装配（pin 单一真源，落 bench/eval_data/）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    ver = sub.add_parser("versions", help="导出 id/revision/split/seed 清单（含实测条数与字节）")
    ver.add_argument("--json", default=None, help="同时把清单写成 json 文件")
    ver.add_argument("--ids", default="", help="只导出这些 id（逗号分隔）")
    fet = sub.add_parser("fetch", help="按 pin 拉取并装配（幂等；失败集列名并以非零码退出）")
    for flag, doc in (("typed", "只拉 typed-decisions"),
                      ("train", "只拉 typed-decisions 的 train 档（训练登记，不入六轴评测表）"),
                      ("intern", "只拉 Intern-Decision"),
                      ("jev", "只拉 jevbench"), ("cn", "拉中文扩展子集（CMMLU+CLUE 题面原件）"),
                      ("zh", "装配中文决策信封（零流量，派生自 --cn 那份题面原件）"),
                      ("zh-train", "拉中文 train 语料（CLUE tnews/ocnli train 档 + 现推决策信封；"
                                   "只挂 train 轴，不入六轴评测表）"),
                      ("mm", "拉 MMBench-CN 子集"), ("long", "拉 LongBench-zh 子集"),
                      ("needle", "只生成合成针计划")):
        fet.add_argument(f"--{flag}", action="store_true", help=doc)
    fet.add_argument("--all", action="store_true", help="三件套 + 全部扩展集")
    fet.add_argument("--sets", default="", help="按注册表 id 点名（逗号分隔）")
    fet.add_argument("--data-dir", default=str(DATA_DIR), help="落盘根目录（gitignored）")
    fet.add_argument("--force", action="store_true", help="已拉过也重拉（默认幂等跳过）")
    fet.add_argument("--no-assemble", action="store_true", help="只下载，不拆成信封")
    lst = sub.add_parser("list", help="打印注册表（不联网）")
    lst.add_argument("--axis", default=None, help="只看某条轴的集")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：versions 打清单，fetch 拉数据并逐集记账，list 只读注册表。

    白话：敲 versions 就交一张这批分数用什么数据的表；敲 fetch 就按版本号把东西取回来、
    拆成能直接吃的样子、把流量与条数写进账本，最后报一句哪些成了、哪些没成——没成的会一个个
    念出名字并让退出码非零，脚本据此决定要不要继续；敲 list 就只把登记的东西念一遍，不联网。
    """
    args = build_parser().parse_args(argv)
    if args.cmd == "versions":
        ids = tuple(s.strip() for s in args.ids.split(",") if s.strip()) or None
        rows = export_versions(ids=ids)
        print(format_versions_table(rows))
        if args.json:
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[registry] versions -> {out}")
        return 0
    if args.cmd == "list":
        for row in export_versions():
            if args.axis and row["axis"] != args.axis:
                continue
            print(f"{row['id']:<20} {row['source']:<11} {str(row['revision']):<12} {row['split']:<5} "
                  f"seed={row['seed']} axis={row['axis']} sampler={row['sampler']} "
                  f"qtypes={row['qtypes']}")
        return 0
    # argparse 把 --zh-train 的 dest 写成 zh_train（连字符换下划线），而 FLAG_TO_IDS 认的是
    # 按钮本名；不按这个规矩换算，带连字符的分组按下去就是块死键（只在表里加键，没人理它）
    groups = tuple(flag for flag in FLAG_TO_IDS
                   if getattr(args, flag.replace("-", "_"), False))
    picked = registry_ids(groups=groups,
                          ids=tuple(s.strip() for s in args.sets.split(",") if s.strip()),
                          all_sets=args.all)
    if not picked:
        picked = THREE_SHEET_IDS
    print(f"[registry] 点名 {len(picked)} 集：{', '.join(picked)}")
    result = fetch(picked, data_dir=args.data_dir, force=args.force,
                   do_assemble=not args.no_assemble)
    for pid, entry in result["ok"].items():
        print(f"[fetch] OK   {pid:<20} resolved={str(entry.get('resolved'))[:16]:<16} "
              f"bytes={entry.get('bytes')} files={entry.get('files')} samples={entry.get('samples')} "
              f"qtypes={entry.get('qtype_counts')} status={entry.get('status')}")
    for pid, err in result["failed"].items():
        print(f"[fetch] FAIL {pid:<20} {err}", file=sys.stderr)
    if result["failed"]:
        print(f"[fetch] 失败集：{', '.join(sorted(result['failed']))}", file=sys.stderr)
        return 1
    print(f"[fetch] 全部就绪（{len(result['ok'])} 集） -> {args.data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
