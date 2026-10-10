"""E_probe_edit_triage.py — 一次性订正工具（agent E，缺陷三）：把 `docs/ci_baseline_triage.md`
里被 D 的 6-run 表与本次实测**证伪**的 sliding 结论（§0 速览行 / §3 全节 / §5 两处）按事实改写为
"判据缺陷，已修"，并链到 `D_determinism.md` 与 `E_RESULT.md`。**只动这三处**，别人章节结论不碰。

定点替换用 `assert old in s`，改后 `git diff -- docs/ci_baseline_triage.md` 只报这三处。

用法（release 目录下）：
    .venv/bin/python ascend/kernels/reconcile/E_probe_edit_triage.py --check
    .venv/bin/python ascend/kernels/reconcile/E_probe_edit_triage.py --apply
"""
from __future__ import annotations

import pathlib
import sys

DOC = pathlib.Path(__file__).resolve().parents[3].parent / "docs" / "ci_baseline_triage.md"

# 计数占位：--apply 前先用 host 全量实测数填（见 E_RESULT.md §5）
HOST_TOTAL = "340"

REPLACEMENTS: list[tuple[str, str]] = [
    # ---------------------------------------------------------------- §0 速览行（§3 的摘要，跟着 §3 一起订正）
    (
        ('| test_longctx sliding 逐位相等 | 1 | 环境伪影（KNOWN-ENV，数值平台敏感） | '
         '**不 skip**，注释挂账 + 本档 §3 | 仍红 ×1（待编排者裁决） |'),
        ('| test_longctx sliding 逐位相等 | 1 | **判据缺陷**（原记"环境伪影"已被证伪，见 §3 订正） | '
         '**判据重建**（结构判据）+ 顺带修掉产品出口形状缺陷；不 skip、不 xfail、不放宽容差 | '
         '不再贡献红（该用例转绿，同码同色） |'),
    ),
    # ---------------------------------------------------------------- §3 全节
    (
        '''## 3. KNOWN-ENV ×1：`test_longctx.py::test_sliding_routes_agree_with_mask_reference`

**判定：纯计算、零资产依赖，不得 skip 掩盖。** `compare_routes(200,32,chunk=48)`
（`sys1/layers/attention.py:394-432`）固定 seed 合成张量，输入与磁盘无关。

红点唯一落在 `kernel_vs_masked_max_abs == 0.0` 的**逐位相等**断言：掩码路物化 200×200
分数矩阵、带状路按 79 宽带分块，两路 fp32 matmul/softmax 的**归约顺序**随 BLAS 实现与
指令集而变——host（arm64, Accelerate/vecLib 轮）位等成立绿；ubuntu（x86 CPU wheel,
OpenBLAS）位等被打破红。数值仍在 allclose 容差内（同报告里 `kernel_allclose=True`
的断言先于位等断言执行），故属"验收口径平台敏感"，非路由回归。

**处置**：不 skip、不放宽（放宽容差/改 xfail 都动 sys1 侧验收口径，越白名单）。用例上方挂
KNOWN-ENV 注释指回本档；在编排者裁决前，下一次 ubuntu run 预期保留这一条红
（`continue-on-error` 尚在，不阻塞合入）。裁决选项：A) 容差改 `<=1e-6` 并同步 spec 口径；
B) `@xfail(strict=True, reason=KNOWN-ENV)` 显式挂账；C) 上 mac runner 设备矩阵后自然消解。''',
        '''## 3. 判据缺陷 ×1（**已修**）：`test_longctx.py::test_sliding_routes_agree_with_mask_reference`

> **订正声明（2026-10-11，执行代理 E）**：本节原文把该红判为"环境伪影（KNOWN-ENV）→ 预期保留红"，
> **该结论已被证伪**。事实是**测试判据缺陷**（并顺带坐实一个**产品形状缺陷**），两者都已修。
> 证据链：`release/ascend/kernels/reconcile/D_determinism.md`（6-run 表 + 三道闸读码 + 宿主 28 次
> pytest/38 次进程内判定）与 `release/ascend/kernels/reconcile/E_RESULT.md`（修复、前后对照、自证跑表）。
> 纪律：未放宽容差当止血、未加 skip、未写 xfail、未改 `sys1/kernels/**`。

**原判定仍成立的部分**：纯计算、零资产依赖，不得 skip 掩盖。[文档]

**证伪一：所谓"内核带状路 vs 掩码路"这个对子根本不存在。** `route="kernel"` 在 CPU 上**三道闸必回落**
`attn_sw_kernel._eager()`：① `backends.active_backend(cpu)` 只放 mps 设备过关（`sys1/kernels/backends.py:104`
"只允许 mps 设备走方言"）；② `attn_sw_mps.plan` 第一门槛要 fp16，而 `compare_routes` 造的是 fp32；
③ `band_len=chunk+window-1=79` 不被 `BLOCK_Q=16` 整除。复跑：
`cd release && .venv/bin/python ascend/kernels/reconcile/D_probe_routes.py route`
→ `[gate1] ... False (actual='torch_eager')`、`[gate2] plan(...) -> None`。
⇒ 断言比的是**两份 eager 实现、不同归约长度（200 列 vs 79 列）之间的位等**；原 §3"host 走内核、
ubuntu 走 OpenBLAS"的图景两侧都不成立。[实测]

**证伪二：红来自判据，不来自环境。** 6 个 ubuntu 数据点 2 红 4 绿；run1↔run2、run4↔run5 在
`git diff --name-only f96f9b3..d57ef8a -- release/sys1` **为空**（被测体一字节未变）的前提下红↔绿翻转
⇒ 位等口径对仓库外的 BLAS/ISA 分派没有免疫力，而"位等"从来不是产品的承诺。红值
`1.7881393432617188e-07` = fp32 3 ULP 且同一报告 `kernel_allclose=True`；宿主同机只换形状，
**姊妹路（只差一个 M 形状）10/20 组出末位差**（D_determinism §4.8）——同一用例对一路要位等、
对另一路给 `1e-5`，判据自相矛盾且在构造上不可保证（带状路的全部意义就是把归约长度从 T 改成
chunk+window）。[实测]

**证伪三：位等口径无 spec 授权。** 该 Scenario 的验收原文只有"选项 argmax 一致 + 长序列显存峰值
低于全注意力"；仓内既有数值契约是 `fp16 err≤2e-2、argmax 一致` 的容差制。"必须逐位相等"的出处
仅测试自身与 `attention.py` 模块 docstring 的自我陈述，属实现自设口径。[文档]

**处置（已落盘）**：

1. **判据重建**（`release/tests/test_longctx.py` C1 段）：主判据换成平台无关的**结构判据**三条——
   A1 非零可见位集合逐格相等（`v=I` 抽权重，含未来位严格 0、对角自见）、A2 argmax 一致
   （带自证门槛：top1-top2 最小间距必须比本平台实测跨路末位差大两个数量级，防"近并列"这一
   残留平台敏感面）、A3 同进程自比（每条路与**自身重跑**位等——同形同分派，位等有构造保证，
   不成立即真非确定）；`*_vs_masked_max_abs <= 1e-5` 降为**附列**，不再当主判据。
   误导措辞按实际订正（"内核带状路"→ eager 带状路），并加一条**措辞事实钉**
   （`backends.active_backend("cpu") == TORCH_EAGER`）防这笔账再回来。
   负例自证（R14）：`test_sliding_visible_set_criterion_detects_reversed_window`——A1 对
   "方向抄反/窗宽 off-by-one"必须判否。
2. **顺带坐实并修掉的产品形状缺陷**（`release/sys1/layers/attention.py`）：`masked_fill` 拿 (1,1,T,T)
   掩码套 (H,T,T) 分数 ⇒ 广播升维（3 维入 → 4 维出）；出口 `reshape(lead_shape(q) + (dim,))` 丢了
   seq 维 ⇒ 4 维入当场 `RuntimeError: shape '[1, 2, 16]' is invalid for input of size 6400`。
   修为同秩喂掩码（`window_band(...)[0]` = (1,T,T)）+ `reshape(lead_shape(q) + (seq, dim))`，
   `attention_band` 出口同族同修。数学未变：3 维出口展平字节哈希改前后同为 `5a7d9530e2ae0a32`，
   `compare_routes` 差值原样（kernel `0.0`、local `1.7881393432617188e-07`）。
   回归钉：`release/tests/test_attention_shapes.py`（7 用例，2/3/4 维全档 + 批 vs 逐样本）。
   复跑：`cd release && .venv/bin/python ascend/kernels/reconcile/E_probe_shapes.py matrix`、
   `.venv/bin/python ascend/kernels/reconcile/D_probe_routes.py shapebug`。
3. 本档 **§0 速览行**与 **§5** 的"预期稳态 `1 failed`"同步订正（同一次改动，口径一致）。

**新预期**：该用例 host/ubuntu **同码同色（绿）**，不再贡献伪红。`continue-on-error` 的撤销时机由
编排者按 §5 观察 2–3 个自然 push 后定夺——本波执行代理**未碰** `.github/workflows/ci.yml`。''',
    ),
    # ---------------------------------------------------------------- §5 条件 2
    (
        '2. 满足其一：`0 failed`；或唯一红仅剩 §3 的 KNOWN-ENV 一条**且**已按 §3 选项 A/B 裁决；',
        ('2. 满足其一：`0 failed`；或唯一红是**新的、已按 R-P1-4 先证后判**的条目（原写法"唯一红仅剩 '
         '§3 的 KNOWN-ENV 一条"作废——§3 已按判据缺陷修复，不再预期任何保留红）；'),
    ),
    # ---------------------------------------------------------------- §5 预期稳态（line ~115）
    (
        '''预期下一次 ubuntu run 稳态：`1 failed (仅 §3 KNOWN-ENV), 290 passed, 38 skipped, 3 deselected` —— 10 条红转 SKIP 只增 skipped（28→38），passed 数不变（红项本就未计入 passed）；host 全量为 `332 passed, 3 skipped, 3 deselected`（收集数与 ubuntu 差在可选依赖的收集期条件，属既有现象非本轮引入）。''',
        '''预期下一次 ubuntu run 稳态（**2026-10-11 订正**）：`0 failed`，且 §3 那条滑窗用例计入 passed。
原文"`1 failed (仅 §3 KNOWN-ENV), 290 passed, 38 skipped, 3 deselected`"这条**预期稳态已被证伪**：
D 的 6-run 表里该用例 **4 绿 2 红**（green：`38052628797` `38061623756` `38068744512` `38070929040`，
其中 `38070929040` 汇总行 `297 passed, 38 skipped, 3 deselected` 已是 `0 failed`），且 run1↔run2、
run4↔run5 之间 `release/sys1` **一字节未变**（`git diff --name-only f96f9b3..d57ef8a -- release/sys1` 为空）
却红↔绿互换——"预期保留一条红"从来没有成立过，它靠的是判据缺陷而非环境事实
（数据点与取数命令见 `release/ascend/kernels/reconcile/D_determinism.md §2.1`）。
本波（缺陷一+二修复后）新增 8 条用例（`tests/test_attention_shapes.py` 7 + `test_longctx` 负例 1），
host 全量基线由 `332 passed, 3 skipped, 3 deselected` 变为 `''' + HOST_TOTAL + ''' passed, 3 skipped,
3 deselected`（`cd release && .venv/bin/python -m pytest tests -q -m "not integration"`）；ubuntu 侧
绝对计数**不作预测**（收集期可选依赖差异导致 passed/skipped 数与 host 不同，属既有现象非本轮引入）。''',
    ),
]


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    s = DOC.read_text()
    bad = 0
    for i, (old, _) in enumerate(REPLACEMENTS, 1):
        n = s.count(old)
        print(f"[locate {i}/{len(REPLACEMENTS)}] 命中 {n} 处  ← {old.splitlines()[0][:60]}…")
        bad += n != 1
    if bad:
        print(f"[abort] {bad} 个位点不唯一——不动文件")
        return 1
    if mode == "--apply":
        for old, new in REPLACEMENTS:
            s = s.replace(old, new)
        DOC.write_text(s)
        print(f"[apply] 已订正 {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
