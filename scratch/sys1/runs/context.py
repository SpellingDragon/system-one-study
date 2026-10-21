"""runs 上下文：给每次实验开一个可溯源的记录本，守住"曲线不许重写、结论必须留下"两条规矩。

【做什么】跑实验（训练/评测/消融）之前先调用 new_run()，它会在 runs/ 下开一个专属文件夹，
里面固定放四份记录：用了什么参数、跑在哪台机器上、一步一步的曲线数据、以及"假设—观察—结论"三行笔记。
事后报告里每个数字都能指回这个文件夹，指不出去的数字按手册算捏造。

【怎么做】文件夹名字（run_id）= 月日 + 实验名短码 + 参数散列的前 4 位；参数写进 config.yaml 时
自动补上代码版本（git commit）、数据版本占位与本机硬件；log_metrics 每次追加一行曲线点，
但同一个 step 的同一个指标名只许写一次，重复即抛 MetricsConflictError；finish() 收尾前检查笔记里
是否已有真正的结论行，没有就抛 MissingConclusionError。

【为什么】防重排在方便前面：曲线若允许同 step 覆盖，"为凑一张好图反复重跑"就会把证据洗白，
而负结果恰恰是评审要看的。曾考虑用启动时间戳当 run_id —— 否，同一秒并发起两个同名 run 会互相
覆盖同一个目录，记录直接丢失；换成参数散列后，同秒但参数不同两个 run 必然分家，真撞上再补递增后缀。
也考虑过把记录全交给托管实验平台 —— 否，外网一断或账号一变证据链就断，本地文件必须是权威真源。
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

__all__ = ["MetricsConflictError", "MissingConclusionError", "RunContext", "RunNotebookError", "new_run"]

# ── 约定常数（改这里等于改契约，须同步 tests/test_runs.py 与 runs/README.md）────────────
CONCLUSION_PREFIX = "结论"          # 笔记里"结论行"的起头字（全角/半角冒号都认）
PLACEHOLDER = "待填写"              # 骨架里的占位文本，等于没写，不算结论
NOTES_SKELETON_KEYS = ("假设", "观察", "结论")  # 三行骨架的顺序被单测锁定
RESERVED_METRIC_KEYS = ("step", "ts")  # 曲线行的这两个字段由工具占用，调用方不许当指标名
HASH_LEN = 4                          # run_id 尾部散列长度
SLUG_MAX = 40                         # 实验名短码最长长度，防目录名爆炸
UNPINNED = "UNPINNED"                 # 数据版本占位：一阶段先用，二阶段接 registry 后填真值

# 本文件所在包往上两级 = 仓库根；commit 与默认 runs/ 落点都以它为准，与当前工作目录无关
_REPO_ROOT = Path(__file__).resolve().parents[2]


class RunNotebookError(RuntimeError):
    """本模块所有可预期错误的共同父类，方便调用方一把兜住。"""


class MetricsConflictError(RunNotebookError):
    """同一个 step 的同一个指标名被写第二次时抛出（曲线唯一真源，拒绝被重写）。"""


class MissingConclusionError(RunNotebookError):
    """收尾时笔记里还没有真正的结论行就抛出——失败的实验也必须交代结论。"""


class RunContext:
    """一个 run 目录的句柄：四件套的路径 + 记曲线 + 写结论 + 收尾校验。

    由 new_run() 返回。四件套路径做成普通属性而不是 property：它们是纯数据、建目录时就定死，
    做成方法反而每个都要写一遍说明文档，读代码的人还以为是会重算的东西。
    """

    def __init__(self, run_id: str, path: str | Path) -> None:
        self.run_id = run_id
        self.path = Path(path)
        self.config_file = self.path / "config.yaml"
        self.system_file = self.path / "system.json"
        self.metrics_file = self.path / "metrics.jsonl"
        self.notes_file = self.path / "notes.md"

    def __repr__(self) -> str:  # 出问题时第一手线索就是"哪个 run、在哪儿"
        return f"RunContext(run_id={self.run_id!r}, path={str(self.path)!r})"

    # ── A2: 曲线只追加、不许回头改 ──────────────────────────────────────────────

    def log_metrics(self, step: int, **kv: Any) -> dict[str, Any]:
        """追加一行曲线点（step + 墙钟时间 + 若干指标），返回真正写进去的那行。

        同一个 step 的同一个指标名只许出现一次，冲突时抛 MetricsConflictError 且完全不写盘——
        宁可让训练脚本当场炸，也不能让两条互相矛盾的数字同时留在一个 run 里，事后画图谁也
        说不清该信哪条。换别的指标名（同一步的学习率、显存峰值）不受影响，照常追加。

        白话：账本只能一页一页往后写，写完就不许回头涂改。要是有人想给第 100 步的损失再记一遍
        新数值（比如把 3.2 改成 3.1 好让曲线好看些），这里直接拒绝并告诉他已经记过了；
        但第 100 步想再记学习率、记耗时，那是另一件事，照写不误。
        """
        if not kv:
            raise ValueError("log_metrics 至少要有一个指标，空记录没有意义")
        reserved = sorted(set(kv) & set(RESERVED_METRIC_KEYS))
        if reserved:  # 保留字段被占用会让"哪个是指标"变得说不清，宁可在写侧就挡住
            raise ValueError(f"{'、'.join(reserved)} 是保留字段名，不能当指标名")

        point = int(step)
        dup = sorted(self._logged_names(point) & set(kv))
        if dup:
            raise MetricsConflictError(
                f"run {self.run_id} 的 step={point} 已记录过 {dup}；metrics.jsonl 是曲线唯一真源，禁止重写"
            )
        row: dict[str, Any] = {"step": point, "ts": _wall_clock()}
        row.update({key: _jsonable(val) for key, val in kv.items()})
        _append_line(self.metrics_file, json.dumps(row, ensure_ascii=False))
        return row

    def _logged_names(self, step: int) -> set[str]:
        """扫已有流水，取出该 step 写过的指标名集合（残行按 _iter_metric_names 的规则丢掉）。"""
        return {name for row_step, name in _iter_metric_names(self.metrics_file) if row_step == step}

    # ── A3: 结论行强制 ─────────────────────────────────────────────────────────

    def conclude(self, text: str) -> None:
        """往 notes.md 追加一行"结论：…"——这是 finish() 放行的唯一凭据。

        空白内容直接拒收：留下一个空结论行等于骗过检查，比不写更糟。

        白话：一句话交代这次到底看出了什么、下一步还做不做。哪怕实验跑砸了也要写，
        砸在哪个环节同样值钱——写下来才叫"负结果入库"，不然这一趟等于白跑。
        """
        body = str(text).strip()
        if not body:
            raise ValueError("结论内容不能为空")
        _append_line(self.notes_file, f"{CONCLUSION_PREFIX}：{body}")

    def finish(self) -> Path:
        """收尾校验：笔记里没有真结论行就抛 MissingConclusionError，通过则返回 run 目录。

        骨架里那句"结论：待填写"不算数——占位文本是人还没写就自动存在的，把它当凭据等于
        这道门形同虚设。

        白话：散场前先翻到笔记页看一眼，那行结论还挂着"待填写"三个字，就说明这次实验
        没交代结果，此时直接报错，逼着人把话说完再收工；失败的实验也一样要把结论补上。
        """
        text = self.notes_file.read_text(encoding="utf-8") if self.notes_file.exists() else ""
        if not _has_conclusion(text):
            raise MissingConclusionError(
                f"run {self.run_id} 的 notes.md 还没有结论行：先 conclude(...) 再 finish()（失败实验也必须记结论）"
            )
        return self.path


def new_run(
    name: str,
    config: dict[str, Any] | None = None,
    *,
    root: str | Path | None = None,
) -> RunContext:
    """开一个新 run：建目录、落四件套、返回 RunContext。

    name 是实验名（进 run_id 的短码部分），config 是本次超参字典（原样落盘并自动补溯源字段），
    root 是记录本根目录，默认仓库的 runs/；测试与临时实验用 root 指到别处，别污染真记录。

    白话：先给这次实验取个门牌号，由"几月几日 + 名字 + 一长串参数算出来的四个字符"拼成；
    然后照着门牌号开一间屋子，屋里摆四样东西——一张参数单、一张机器身份卡、一本只许往后翻
    不许回头改的流水账、还有一页"我猜什么、我看到什么、我下什么结论"的笔记。
    """
    cfg: dict[str, Any] = dict(config or {})  # 拷一份，绝不就地改调用方的字典
    runs_root = Path(root) if root is not None else default_runs_root()
    run_id = _make_run_id(runs_root, name, cfg)
    path = runs_root / run_id
    path.mkdir(parents=True)  # 名字已查重，parents=True 只为首次创建 runs/ 本身

    # 溯源三要素：代码版本 + 参数 + 机器。缺一条，事后就无法复跑，报告里的数字立刻失去支撑
    record: dict[str, Any] = {"run_id": run_id}
    record.update(cfg)
    record["commit"] = _git_commit()          # 强制覆盖：代码版本只认 git 说了算的那个
    record.setdefault("data_revision", UNPINNED)  # 占位；接上 eval/registry 后填真实 pin
    record["hardware"] = _hardware()
    _write_yaml(path / "config.yaml", record)
    _write_json(path / "system.json", {"run_id": run_id, **_system_info()})
    (path / "metrics.jsonl").touch()  # 流水账先开空本：后续只追加，读侧按行解析
    _write_text(path / "notes.md", _notes_skeleton(run_id))
    return RunContext(run_id, path)


# ── run_id：目录名即身份（与 .gitignore 的 runs/**/config.yaml 等白名单对齐）───────────

def _slug(name: str) -> str:
    """把实验名压成目录友好的一串短码：小写、非字母数字一律变减号。

    只留 ASCII 是刻意的：中文名会被丢掉，换来的是 shell 补全、git 与各平台都不折腾；
    若要留中文，改成 unicodedata 转写即可，但那会让 run_id 正则锁不住（已在【为什么】权衡）。
    """
    s = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
    s = s[:SLUG_MAX].strip("-")
    return s or "run"  # 全被过滤光（如纯中文）时兜底，保证 run_id 不出现空洞段


def _config_hash(config: dict[str, Any]) -> str:
    """对参数算 4 位短散列：同样名字、不同参数的两次实验因此分家。

    键排序后转 JSON 再散列，避免"字典顺序不同→散列不同"的假差异；default=str 兜住
    路径、集合这类非 JSON 原生值，宁可散列粗糙也不在开 run 这一步崩掉。
    """
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:HASH_LEN]


def _make_run_id(runs_root: Path, name: str, config: dict[str, Any]) -> str:
    """生成 run_id = <MMDD>-<slug>-<hash4>；同秒撞名再往后缀 -2、-3 递增。"""
    stamp = time.strftime("%m%d")
    base = f"{stamp}-{_slug(name)}-{_config_hash(config)}"
    candidate, n = base, 1
    while (runs_root / candidate).exists():
        n += 1
        candidate = f"{base}-{n}"  # 设计里的回退路径：散列仍撞就加序号，绝不静默复用别人的目录
    return candidate


# ── 笔记骨架（假设→观察→结论）与结论判定 ─────────────────────────────────────────────

def _notes_skeleton(run_id: str) -> str:
    """三行骨架的正文：机器生成，人只负责把"待填写"换掉。"""
    lines = [f"# run: {run_id}", ""]
    for key in NOTES_SKELETON_KEYS:
        lines.append(f"- {key}：{PLACEHOLDER}")
    lines += [
        "",
        "> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。",
        "> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。",
        "",
    ]
    return "\n".join(lines)


def _has_conclusion(text: str) -> bool:
    """判断笔记里是否存在"写实的结论行"——骨架里那句待填写不算。"""
    for line in text.splitlines():
        m = re.match(r"^\s*(?:[-*]\s*)?结论\s*[:：]\s*(\S.*)$", line)
        if m and m.group(1).strip() != PLACEHOLDER:
            return True
    return False


# ── 曲线行的读写辅助 ───────────────────────────────────────────────────────────────

def _wall_clock() -> str:
    """墙钟时间戳（本地时区，ISO 形）：事后能判断两个点隔了多久、是不是夜里跑的。"""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _jsonable(value: Any) -> Any:
    """把指标值收敛成 JSON 原生类型：数值型（含 numpy/torch 的标量）先转 float，怪对象转字符串。

    训练管线递过来的常是那些库自己的数字对象，不先收一道的话整行写不进流水账；
    转 float 保住了可画性，兜底成字符串至少保住"这个数被记下来了"。
    """
    if value is None or isinstance(value, (bool, int, float, str, list, dict, tuple)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _iter_metric_names(path: Path) -> Iterator[tuple[int, str]]:
    """逐行读流水账，产出 (step, 指标名)；解析不了的残行跳过。

    白话：上一次进程被砍断时，最后一行可能只写了一半。读的时候遇到这种半截行就丢掉，
    前面完好的记录照常能用——不能因为一行坏了就让整个 run 的曲线全都读不出来。
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # 残行：进程被 kill 的产物，剔除但不报错（design 风险项）
        if not isinstance(row, dict) or not isinstance(row.get("step"), int):
            continue
        for key in row:
            if key not in RESERVED_METRIC_KEYS:
                yield row["step"], key


# ── 环境与版本采集（全部 stdlib；装了 torch/tilelang 就顺手记一笔，没装记 null）───────

def _pkg_version(pkg: str) -> str | None:
    """读包版本号，但不 import 它。

    白话：想知道机器上装了哪个版本的某个大件，直接翻它的包装标签就够了，
    不必真把大件搬起来跑一遍——搬一次可能要几秒，而这个函数在建目录时就会被调用。
    """
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _run_cli(args: list[str]) -> str | None:
    """跑一条外部命令取 stdout；命令不存在、超时、非零退出统一返回 None。

    宁可记 null 也不让建目录失败：换到没有 git 的机器（如容器里 .git 未挂载）时，
    缺代码版本是"记录不全"，抛异常则是"实验跑不了"，后者代价大得多。
    """
    exe = shutil.which(args[0])
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, *args[1:]], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stdout.strip()
    return text if out.returncode == 0 and text else None


def _git_commit() -> str | None:
    """取当前代码版本（git rev-parse HEAD），失败返回 None 并如实留在参数单里。"""
    sha = _run_cli(["git", "rev-parse", "HEAD"])
    # 只认 40 位十六进制：短哈希、报错文本、子模块前缀都可能混进来，宁缺毋滥
    return sha if sha and re.fullmatch(r"[0-9a-f]{40}", sha) else None


def _chip() -> str:
    """取芯片型号：macOS 走 sysctl（能报出 Apple M3 Pro 这类名字），其余退回 platform。"""
    if sys.platform == "darwin":
        brand = _run_cli(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            return brand
    return _cpu_fallback()


def _cpu_fallback() -> str:
    """拿不到具体芯片时给出的最粗粒度机器描述。

    白话：有的系统愿意告诉我们是哪颗芯，有的只肯说某厂商的某种架构，
    那就把能拿到的两段拼起来写，反正比空着强，事后至少能筛出是哪一类机器跑的实验。
    """
    proc = platform.processor() or ""
    mach = platform.machine() or ""
    return f"{proc} {mach}".strip() or "unknown"


def _hardware() -> dict[str, Any]:
    """config.yaml 里的 hardware 段：够回答"这数字是在什么机器上跑出来的"。"""
    return {
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "chip": _chip(),
    }


def _system_info() -> dict[str, Any]:
    """system.json 的正文：软件版本为主，硬件与主机名辅助定位"是不是那台机器"。"""
    return {
        "python": sys.version.split()[0],
        "torch": _pkg_version("torch"),
        "tilelang": _pkg_version("tilelang"),
        "hostname": _hostname(),
        **_hardware(),
    }


def _hostname() -> str | None:
    """主机名取 stdlib 的 socket.gethostname()；拿不到就 None，不影响其余字段。"""
    try:
        return socket.gethostname() or None
    except OSError:
        return None


# ── 落盘小工具（统一 utf-8 + 末尾换行，防 yaml/json 与后续追加互相咬住行）──────────────

def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    _write_text(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    """参数单落盘：sort_keys=False 保住"先身份、再参数、后溯源字段"的可读顺序。

    白话：这张单子给人看，也给人抄——把重要的几行放在最上面、其余按原始次序排，
    比按字母打散更容易一眼扫到要核对的那一项；真要改值时也不会改错行。
    """
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
    _write_text(path, text)


def default_runs_root() -> Path:
    """默认记录本根目录：仓库下的 runs/，可用环境变量 DMLAYA_RUNS_DIR 整体挪走。

    白话：平时东西都堆在自家仓库的 runs 文件夹里；要是某天磁盘不够、或者想在同一份代码上
    跑两拨互不干扰的实验，设一个环境变量就能把整本记录本搬到别处去，代码一行不用改。
    """
    env = os.environ.get("DMLAYA_RUNS_DIR")
    return Path(env).expanduser().resolve() if env else _REPO_ROOT / "runs"


def _append_line(path: Path, line: str) -> None:
    """追加一行并立刻 flush：进程被 kill 时最多丢掉这一行，不会连着带走前面的内容。"""
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
        f.flush()
