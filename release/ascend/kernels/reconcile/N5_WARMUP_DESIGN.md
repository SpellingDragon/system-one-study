# N5_WARMUP_DESIGN.md — 回退 M4 ＋ 落 A2「每键首发射预热丢弃重跑」

> 执行代理：**N5**　|　变更：p2-13　|　日期：2026-10-12　|　**不占卡**（host 双证；真机判决留窗八）
> 一句话：**回退窗七证伪的 M4（入口预清零），改落 A2（每键首发射预热）**——在**内核 run 包装层**
> 用「键首次真发射前置一次同参丢弃发射」规避 fresh `dbeta` 错。**机理仍未定位（窗七：低于生成码
> 层），这是证据驱动的规避，不是语义修复**（同族于 M2 的 bk 避让）。

---

## ① 状态锚点（可复跑）

| 项 | 值 | 复跑命令 |
|---|---|---|
| `gdn_asc.py` 回退前 md5（M4 版） | `f3419883b2206667ed893f89d9276311` | `git show HEAD:release/ascend/kernels/gdn_asc.py \| md5` |
| `gdn_asc.py` 回退目标 md5（a882463 版） | `676c07936547e9d897d3b2366a0317e0` | `git show a882463:release/ascend/kernels/gdn_asc.py \| md5` |
| `gdn_asc.py` 现值（回退 + A2） | `eebdd15e61e267b14da5cee0338b23be` | `md5 ascend/kernels/gdn_asc.py` |
| `gdn_conv_asc.py` 现值（仅 A2） | `e05c48e039005bdaea935c5e575f3883` | `md5 ascend/kernels/gdn_conv_asc.py` |
| `M4_UBPREZERO.md` 现值（头部横幅） | `0de0fced59d38d0bc456e269af4d9636` | `md5 ascend/kernels/reconcile/M4_UBPREZERO.md` |
| 回退净效果 | 删除 `gdn_asc.py` 的 M4 块（+18 行 @ 旧 L386-403），**只回退这一个文件** | `git diff a882463 HEAD -- ascend/kernels/gdn_asc.py` → 见 ② |

---

## ② 回退 M4（窗七证伪且有害）

窗七真机判决（`runs/1012-p2-13-oncard-window4-pin/notes.md` §⑩）：
- fresh `ITEM dbeta rel=8.002e-01` 与窗四**逐字同** ⇒ M4 的入口清零**未生效**；
- zero-prefill 档 `1.018e-07 → 4.606e-01` **恶化**（`dirty==zero` 收敛同值） ⇒ M4 在 ub 设备
  路径**改变数值（非中性）**；
- 复盘：M4 的 host 验证只覆盖 `pure`（cpu-alias 走 pure），**ub 体从未被数值验证过** ⇒ 一次
  「未验证路径上的猜测性修复」，被窗七当场抓回。

处置：`git show a882463:release/ascend/kernels/gdn_asc.py > ascend/kernels/gdn_asc.py`。
逐字验证：
- md5 复归 `676c07936547e9d897d3b2366a0317e0`（= 回退目标）；
- `wc -l` 776；`git diff a882463 -- ascend/kernels/gdn_asc.py` **空**（除下述 A2 新增外）；
- a882463..HEAD 在 `gdn_asc.py` 上**有且仅有** M4 这一处改动 ⇒ 回退即精确撤销 M4、不误伤他物；
- `gdn_conv_asc.py` 未被 M4 触碰（`git diff a882463 HEAD -- ascend/kernels/gdn_conv_asc.py` 空）。

M4 的探针（`M4_probe_ubaudit.py`）/留档文档保留作**负结果**，`M4_UBPREZERO.md` 头部已加证伪横幅。

---

## ③ A2 形态（最小面、四处复用）

**为什么在内核层而非 autograd 层**：窗四证据是「**同键先前发射**」，键的粒度在**内核 run**；
且 conv（`gdn_conv_asc`）与 delta 同族 —— 一处模式、四处复用。

### 3.1 证据链
- 窗四原始数据（未被动摇）：`gdnb` **无 M4** 时 —— fresh（首次发射）`dbeta 0.80` 错；
  **任何先前发射（dirty 或 zero prefill）后 `1.018e-07` 全对**。
  ⇒ **每键首次发射丢弃重跑**即应为修复。
- K4（`K4_RESULT.md`）：`ub` 的 fresh 指纹 `e609a6ab` 恰等于 `pure` 指纹 ⇒ `ub` 在 UB 未初始化
  （fresh）时与 `pure` 同错，`prefill=zero` 后才对。
- 窗七 §⑩：M4 证伪后 A2 = 正解候选（本件落地）。

### 3.2 代码形态（`gdn_asc.py`，conv 复用）
```python
ENV_DISABLE_WARMUP = "SYS1_DISABLE_WARMUP"     # 逃生口（负例 / 窗八 A0 对照臂）
_WARMED: set[str] = set()                       # 模块级：已预热过的编译键

def _on_npu(args):                              # 真设备判据（见 3.3）
    for t in args:
        if getattr(getattr(t, "device", None), "type", "") == "npu":
            return True
    return False

def _launch_dev(kernel, args, key, target):
    if target == ascend_env.TARGET_ASCEND and _on_npu(args) and _warmup_should_fire(key):
        print(f"WARMUP {key}", flush=True)
        kernel(*args)   # 预热发射：同参、丢弃结果
        _sync_device()
        kernel(*args)   # 真发射：取结果
        _sync_device()
    else:
        kernel(*args)
```
- 四处调用点：`gdn_asc.run` / `gdn_asc.run_backward`（前向重算 + 伴随各一处）/ `gdn_conv_asc.run` /
  `gdn_conv_asc.run_backward`。
- `_sync_device()` 用 `torch.npu.synchronize`（无 npu 设备时空操作，口径同 `bench_card._sync_fn`）。
- **计数语义不受影响**：预热发射用**同一个编译键**（`spec["key"]`），不新增键。
- 可观测：预热触发时打一行 `WARMUP <key>`（进日志，供窗八确认「确实预热了」）。

### 3.3 ⚠️ 关键实现点：判据必须是「真实设备」而非「spec target 名」
初版按 `spec["target"] == ascend` 判，**当场把 host 腿打红**：`run_delta_card --target cpu` 是
**cpu-alias** —— 产品件内部口径仍是 `target="ascend"`（`run_delta_card.py:243`），但张量在 CPU、
编译后端是 `c`。于是预热在 host 上误触发，`run_backward` 前向重算件被跑两遍，`--bwd` 判决行
从 PASS 翻 FAIL（dq `7.53e-08→2.77e-01`…）。
修正：按**张量设备类型** `device.type == "npu"` 判（口径同 `train_step_real.py:477/768/808`）。
短路顺序 `target==ascend and _on_npu(args) and _warmup_should_fire(key)` 保证 cpu-alias 连
`_warmup_should_fire` 都不触发 ⇒ `_WARMED` 不被 host 污染。此坑已由 host 腿复跑回归抓出。

---

## ④ 窗八三臂判读表（真机）

| 臂 | 配置 | 期望（若 A2 为真） | 判据行 |
|---|---|---|---|
| **A0**（基线/对照） | `SYS1_DISABLE_WARMUP=1` | 复现窗四 fresh 错：`ITEM dbeta rel≈8.002e-01` | `ITEM dbeta … verdict=FAIL` |
| **A1**（预清零，已否） | M4 在码内 | 已证伪且有害（本件已回退，不再跑） | —— |
| **A2**（默认=预热） | 默认（预热开） | fresh `ITEM dbeta rel < 2e-4` 六项全 PASS；日志见 `WARMUP gdnb[…]` | `ITEM dbeta … verdict=PASS` |

辅助观察（窗八）：
- `D3_probe_dbeta_order.py --dev npu`：顺序探针 `none` 指纹应**收敛到** `zero`（无 M4 时窗四的
  `e609a6ab` 世界）；**不应**出现窗七带 M4 的 `4b017bea`。
- `gradcheck_card.py --target ascend`：`gdn_delta` 期望转绿；**`gdn_conv` 仍 FAIL**（F4 预测：
  UB 措施修不了 conv，其 `dw_ub` 本就清零）——A2 若把 conv 也带绿，则该预测被推翻，需重审。

---

## ⑤ 双证（host，本次已跑）

| 证 | 结果 | 命令 |
|---|---|---|
| 回退 md5 | `676c0793…`（回退后、加 A2 前） | 见 ① |
| host 四腿判决行**逐字不变** | delta_bwd / gradcheck / train_step / train_step_real --dry **IDENTICAL** | 见下 |
| `pytest` | **340 passed, 6 skipped**（改动前/后同） | `.venv/bin/python -m pytest -q tests/` |
| A2 单元自证（含负例） | `N5-WARMUP-DONE pass=8 fail=0` | `…reconcile/N5_probe_warmup.py` |
| 容器键形（host 替身） | `gdnb[ascend]\|h2k8v8\|c2ubz` PASS | `…reconcile/N5_probe_keyshape.py` |
| 负例（逃生口） | `SYS1_DISABLE_WARMUP=1` ⇒ 单次发射、零 WARMUP | `N5_probe_warmup.py` 场景 ④⑤ |

host 四腿复跑（`release/`）：
```bash
.venv/bin/python ascend/port910b/run_delta_card.py --target cpu --bwd
.venv/bin/python ascend/gradcheck_card.py --target cpu
.venv/bin/python ascend/train_step.py --target cpu
.venv/bin/python ascend/train_step_real.py --target cpu --dry
```
判决行 diff（改动前 vs 改动后）逐字为空（`ITEM …`/`NUM-…`/`GRADCHECK-…`/`P2-STEP…`/`P2-REAL-…`/
`KERNEL-USED…`/`DTYPE-TRACE…`/`COMPILE-KEY…`）。`run_delta_card --bwd` 仍
`NUM-delta_bwd PASS … dbeta=1.01e-07 … rel_max=1.01e-07`。

---

## ⑥ 未决 / 偏离

- **容器 `--no-run`（`c2ubz`）本次未在真容器跑**：本沙箱到昇腾实例 ssh **不可达**（TCP 通、SSH
  banner 超时，`Connection timed out during banner exchange`）。改用 host 直接构造 `plan_backward`
  的 spec 键作**替身**（`N5_probe_keyshape.py` → `c2ubz`）；且 A2 未触碰 `plan`/`plan_backward`
  的**函数体**（`git diff` 仅覆盖 helper 区与四处调用点）⇒ 键形不变是**结构性保证**。
  **建议**：容器得连时补跑 `run_delta_card.py --target ascend --no-run` 逐字核 `c2ubz`。
- **机理未定位**：窗七已把读数疑点下移到「低于生成码层」（硬件首发射态/同步语义层）。A2 是规避，
  **不解释**为何首发射读未初始化 UB。
- **A2 附带预热了 `run_backward` 的前向重算件**（`fwd["key"]` 同键首次也预热）：属「每键首次」的
  自然外延；若窗八测得前向重算件本无 fresh 病，可再收窄到仅伴随件（改动面不变，仍在此四处）。
