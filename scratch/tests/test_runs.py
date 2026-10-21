"""sys1/runs 的单测：run 目录四件套（A1/B1）、曲线防重（A2）、结论强制（A3）。

对应 spec：openspec/changes/teacher-p1-scratch-mps/changes/p1-02-run-notebook/specs/run-notebook/spec.md
全部用例把 root 指到 pytest 的 tmp_path，绝不往仓库真 runs/ 里写东西（保持仓库干净）。
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from sys1.runs import MetricsConflictError, MissingConclusionError, RunContext, new_run

# run_id 合法式：四位日期 + 小写短码段 + 四位十六进制散列（B1 锁定的格式契约）
RUN_ID_RE = re.compile(r"^\d{4}-[a-z0-9](?:[a-z0-9-]*[a-z0-9])?-[0-9a-f]{4}$")


def _git_head() -> str:
    """仓库当前 HEAD，用来核对 config.yaml 里的 commit 字段是真取自 git 而非写死。"""
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _rows(run: RunContext) -> list[dict]:
    """把 metrics.jsonl 读回成字典列表，跳过空行。"""
    text = run.metrics_file.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ── A1: new_run 建目录 + 四件套落盘 ────────────────────────────────────────────────

def test_new_run_creates_four_files(tmp_path):
    """spec 场景「创建 run」：目录含四个文件，RunContext 暴露 run_id。"""
    run = new_run("s1-pretrain", {"lr": 0.001, "bs": 8}, root=tmp_path)

    assert isinstance(run, RunContext)
    assert run.path == tmp_path / run.run_id, "目录名必须就是 run_id（.gitignore 白名单按它匹配）"
    names = sorted(p.name for p in run.path.iterdir())
    assert names == ["config.yaml", "metrics.jsonl", "notes.md", "system.json"], names

    assert run.metrics_file.read_text(encoding="utf-8") == "", "新建 run 的曲线流水必须是空本"


def test_new_run_config_carries_commit_and_provenance(tmp_path):
    """溯源三要素：commit 取自 git HEAD，超参原样保留，data_revision 有占位，硬件有记录。"""
    cfg = {"lr": 0.001, "nested": {"depth": 2}}
    run = new_run("s2-sft", cfg, root=tmp_path)

    saved = yaml.safe_load(run.config_file.read_text(encoding="utf-8"))
    assert saved["commit"] == _git_head()
    assert saved["lr"] == 0.001 and saved["nested"]["depth"] == 2
    assert saved["data_revision"] == "UNPINNED"
    assert saved["hardware"]["chip"], "hardware 段要能回答「这数字是什么机器跑出来的」"
    assert saved["run_id"] == run.run_id

    assert cfg == {"lr": 0.001, "nested": {"depth": 2}}, "不得就地修改调用方传入的字典"


def test_new_run_config_commit_overrides_caller_value(tmp_path):
    """代码版本只认 git：调用方自带 commit 字段也一律被真实 HEAD 覆盖，防手写假版本。"""
    run = new_run("commit-override", {"commit": "deadbeef"}, root=tmp_path)

    saved = yaml.safe_load(run.config_file.read_text(encoding="utf-8"))
    assert saved["commit"] == _git_head()


def test_new_run_system_json_records_software_versions(tmp_path):
    """system.json 记 torch/tilelang/OS/芯片；未安装的包记 null 而不是崩掉建目录。"""
    run = new_run("sys-check", {}, root=tmp_path)

    info = json.loads(run.system_file.read_text(encoding="utf-8"))
    for key in ("python", "torch", "tilelang", "os", "machine", "chip"):
        assert key in info, f"system.json 缺 {key}"
    assert info["run_id"] == run.run_id


def test_new_run_default_root_is_repo_runs(tmp_path, monkeypatch):
    """不传 root 时默认落仓库 runs/，且可用 DMLAYA_RUNS_DIR 整体挪走（测试用后者避免污染）。"""
    monkeypatch.setenv("DMLAYA_RUNS_DIR", str(tmp_path / "moved"))
    run = new_run("default-root", {"a": 1})
    assert run.path.parent == (tmp_path / "moved").resolve()


# ── A2: metrics 追加与防重（spec 场景「定时记录与防重」）────────────────────────────

def test_conflict_rejects_same_step_same_metric(tmp_path):
    """先记 loss=3.2 再想改成 3.1 → 第二次抛 MetricsConflictError，step=100 仍只有一行。"""
    run = new_run("conflict-loss", {}, root=tmp_path)
    run.log_metrics(100, loss=3.2)

    with pytest.raises(MetricsConflictError):
        run.log_metrics(100, loss=3.1)

    at_100 = [r for r in _rows(run) if r["step"] == 100]
    assert len(at_100) == 1, f"被拒的第二次不该留下新行，实得 {len(at_100)} 行"
    assert at_100[0]["loss"] == 3.2, "先写者为准：被拒的写入不得改动已有数值"


def test_conflict_allows_other_metric_same_step(tmp_path):
    """同一步换指标名不算冲突：step 与墙钟时间随每行落盘。"""
    run = new_run("conflict-multi", {}, root=tmp_path)
    run.log_metrics(100, loss=3.2)
    run.log_metrics(100, lr=0.001, ece=0.12)  # 同 step 不同指标，允许

    rows = _rows(run)
    assert len(rows) == 2
    assert rows[1]["step"] == 100 and rows[1]["lr"] == 0.001 and rows[1]["ece"] == 0.12
    assert rows[0]["ts"] and rows[1]["ts"], "每行必须带墙钟时间"


def test_conflict_allows_same_metric_on_other_step(tmp_path):
    """防重的边界是「同一 step」：loss 在 100/200 各记一次完全合法（曲线本就该逐 step 长）。"""
    run = new_run("conflict-steps", {}, root=tmp_path)
    run.log_metrics(100, loss=3.2)
    run.log_metrics(200, loss=2.8)

    assert [r["loss"] for r in _rows(run)] == [3.2, 2.8]


def test_conflict_free_append_records_multiple_keys_at_once(tmp_path):
    """一次调用可递多个指标，整行原子追加；数值型对象先收敛成 JSON 原生类型。"""
    run = new_run("conflict-batch", {}, root=tmp_path)
    row = run.log_metrics(7, loss=1.5, acc=0.42, note="smoke")

    assert row["step"] == 7 and row["loss"] == 1.5 and row["note"] == "smoke"
    assert len(_rows(run)) == 1
    with pytest.raises(ValueError):
        run.log_metrics(8)  # 空指标拒绝：写一行只有时间戳的记录没有意义


def test_conflict_scan_tolerates_broken_tail_line(tmp_path):
    """进程被砍断留下的半截行：读侧剔除它，同 step 因此可以补记（design 风险项）。"""
    run = new_run("conflict-tail", {}, root=tmp_path)
    run.log_metrics(50, loss=2.0)
    with run.metrics_file.open("a", encoding="utf-8") as f:
        f.write('{"step": 51, "loss": ')  # 半截 JSON，模拟写一半就断了

    run.log_metrics(51, loss=1.9)  # 残行没写成，51 步仍可记
    with pytest.raises(MetricsConflictError):
        run.log_metrics(50, loss=1.0)  # 完好行仍受防重保护


def test_conflict_rejects_reserved_field_names(tmp_path):
    """step / ts 由工具占用，不能被当指标名——否则一行里出现两个 step，事后无法判断哪个是真。"""
    run = new_run("conflict-reserved", {}, root=tmp_path)

    with pytest.raises(ValueError):
        run.log_metrics(10, ts="2026-01-01")


# ── A3: 结论行强制（spec 场景「无结论收尾被拦」）───────────────────────────────────

def test_conclude_missing_blocks_finish(tmp_path):
    """从没 conclude 过就 finish → MissingConclusionError（失败实验也必须交代结论）。"""
    run = new_run("no-conclusion", {}, root=tmp_path)

    with pytest.raises(MissingConclusionError):
        run.finish()


def test_conclude_then_finish_passes_and_appends_line(tmp_path):
    """conclude 追加"结论：…"行，finish 放行并返回 run 目录。"""
    run = new_run("has-conclusion", {}, root=tmp_path)
    run.conclude("温度 1.5 之后 ECE 从 0.11 降到 0.06，采纳该值")

    assert run.finish() == run.path
    last = run.notes_file.read_text(encoding="utf-8").rstrip().splitlines()[-1]
    assert last.startswith("结论："), f"结论行须以「结论：」起头，实得 {last!r}"
    assert "ECE" in last


def test_conclude_skeleton_placeholder_is_not_a_real_conclusion(tmp_path):
    """骨架自带的「结论：待填写」不算凭据：光填观察不填结论，finish 仍被拦。"""
    run = new_run("placeholder-only", {}, root=tmp_path)
    text = run.notes_file.read_text(encoding="utf-8")
    run.notes_file.write_text(text.replace("- 观察：待填写", "- 观察：loss 第 100 步 3.2 后不再下降"), encoding="utf-8")

    with pytest.raises(MissingConclusionError):
        run.finish()


def test_conclude_rejects_blank_text(tmp_path):
    """空结论拒收：留个空结论行等于骗过检查，比不写更糟。"""
    run = new_run("blank-conclusion", {}, root=tmp_path)

    with pytest.raises(ValueError):
        run.conclude("   ")
    with pytest.raises(MissingConclusionError):
        run.finish()


def test_conclude_can_be_called_twice_latest_wins_in_report(tmp_path):
    """结论可多次追加（复盘时补一句），finish 只要求至少有一条写实结论。"""
    run = new_run("two-conclusions", {}, root=tmp_path)
    run.conclude("首轮：seed3 明显优于 seed0")
    run.conclude("复核：差距在 3 seeds 下仍稳定，结论保留")

    assert run.finish() == run.path
    lines = [ln for ln in run.notes_file.read_text(encoding="utf-8").splitlines() if ln.startswith("结论：")]
    assert len(lines) == 2, "conclude 是追加而非覆盖，两次都要留下"


# ── B1: run_id 规则与 notes 三行骨架 ──────────────────────────────────────────────

def test_run_id_format_is_date_slug_hash(tmp_path):
    """run_id = <MMDD>-<slug>-<hash4>：只有最后一段是散列，中间整段都是实验名短码。"""
    run = new_run("S1 Pretrain!! v2", {"lr": 0.1}, root=tmp_path)

    assert RUN_ID_RE.match(run.run_id), f"run_id 不合格式: {run.run_id}"
    date, _, rest = run.run_id.partition("-")
    slug, _, hash4 = rest.rpartition("-")  # exp-id 自己就带减号，故散列只能从尾部切一刀
    assert date == time.strftime("%m%d"), "日期段必须跟当天走（MMDD）"
    assert slug == "s1-pretrain-v2", f"slug 应只留小写字母数字并以减号分隔，实得 {slug}"
    assert re.fullmatch(r"[0-9a-f]{4}", hash4), f"hash4 应是四位十六进制，实得 {hash4}"
    assert run.path.name == run.run_id, "目录名即 run_id"


def test_run_id_hash_tracks_config_and_splits_same_name(tmp_path):
    """同名不同参数 → 不同 hash4 → 两个独立目录（防同秒互相覆盖）。"""
    a = new_run("same-name", {"seed": 1}, root=tmp_path)
    b = new_run("same-name", {"seed": 2}, root=tmp_path)

    assert a.run_id != b.run_id
    assert a.run_id.rsplit("-", 1)[1] != b.run_id.rsplit("-", 1)[1], "尾段 hash4 应随参数变化"
    assert a.path.is_dir() and b.path.is_dir()


def test_run_id_suffix_increments_on_hash_collision(tmp_path, monkeypatch):
    """散列真撞车时递增后缀，绝不复用别人的目录（design 风险项的回退路径）。"""
    import sys1.runs.context as ctx

    monkeypatch.setattr(ctx, "_config_hash", lambda cfg: "dead")
    first = new_run("clash", {"seed": 1}, root=tmp_path)
    second = new_run("clash", {"seed": 2}, root=tmp_path)

    assert first.run_id.endswith("-dead")
    assert second.run_id == f"{first.run_id}-2", second.run_id
    assert second.path.is_dir() and second.path != first.path


def test_run_id_notes_skeleton_has_three_locked_lines(tmp_path):
    """notes.md 三行骨架：假设 → 观察 → 结论，顺序与占位文本被单测锁死。"""
    run = new_run("notes-skeleton", {}, root=tmp_path)

    text = run.notes_file.read_text(encoding="utf-8")
    order = [line for line in text.splitlines() if "：" in line and line.lstrip().startswith("-")]
    assert [ln.split("：")[0].strip("- ").strip() for ln in order] == ["假设", "观察", "结论"], order
    assert run.run_id in text.splitlines()[0], "首行应标明这是哪个 run 的笔记"
