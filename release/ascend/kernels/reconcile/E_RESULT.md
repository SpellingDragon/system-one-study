# E_RESULT — p2-13/audit 交界 执行代理 E（缺陷一 / 二 / 三）

派单：修 D 代理坐实的两处真缺陷（产品形状违约 + 环境相关非确定判据），并订正被证伪的 CI 台账。
完成度：**3/3 缺陷处置落盘**；host 全量 `340 passed, 3 skipped, 3 deselected`（基线 332 + 新增 8）。

**三态标注约定**：[实测]=本文附可复跑命令的真实输出；[文档]=读码/规范原文；[推断]=推理，未独立验证。

**未触碰清单（派单禁改）**：`.github/workflows/ci.yml`、`scratch/**`、`release/ascend/**` 产品件、
`sys1/kernels/**`、`production/**`、trunk、他人 reconcile 产物、`tasks.md`；无 git 写操作。
复跑：`cd /Users/pengweiye/Documents/codes/system-one && git status --porcelain -- .github/workflows/ci.yml scratch release/sys1/kernels release/production openspec/changes`
→ [实测] 空输出（未触碰）。

---

## 1. 缺陷一：`attention_masked` 出口形状违约（产品缺陷，已修）

### 1.1 现契约与消费方预期秩 [文档]

| 依据 | 原文要点 |
|---|---|
| `sys1/layers/attention.py:303-306`（函数签名） | `attention_masked(q, k, v, *, window=None, scale=None) -> AttnOut`，q/k/v 支持 2/3/4 维开本 |
| `attention.py:292-295` `AttnOut` 定义 | `out: torch.Tensor  # (heads, seq, dim)`（原文行内注释）——出口秩契约的**出处** |
| `_flatten_heads` / `lead_shape` | `_flatten_heads` 显式接受 2/3/4 维；`lead_shape` 的用途是"把出口 lead 维还原成入口 lead 维" ⇒ **入出同秩**是该模块自设契约 |
| `compare_routes`（`attention.py:394+`） | 消费方，喂 3 维 |
| `release/tests/test_longctx.py` C1 段 | 消费方，喂 3 维 |

**调用方盘点（全仓 grep）** [实测]：
`grep -rn "attention_masked" release/ --include="*.py"` → 命中的**产品侧调用方只有 `compare_routes` 与
`test_longctx.py`**，均喂 3 维；**没有任何 4 维消费者**。
⇒ 结论：4 维这条契约"声明支持、一调即崩"，属**未被告知的缺陷**而非未覆盖的假设 [推断（基于 grep 事实）]。

### 1.2 根因：两处，不是一处 [文档 + 实测]

1. **`masked_fill` 广播升维**（原 `attention.py:320`）：`window_band(seq, window)` 返回 `(1,1,T,T)`，
   分数是 `(H,T,T)` ⇒ torch 广播出 `(1,H,T,T)`，**3 维入 → 4 维出**。
2. **出口 reshape 丢 seq 维**（原 `attention.py:322` 与 `:391`）：
   `out.reshape(lead_shape(q) + (dim,))` 对 4 维入只补回 `(B,H)` + `dim`，**漏掉 seq**，元素数不匹配当场崩。

⇒ 只修第 1 处仍会在第 2 处崩；`attention_band` 吃的是**同一个**出口 reshape 缺陷（4 维崩、2 维升维）。
这是本次把 band 出口一并同修的根据（详见 §1.3 与 §6 自行决策申报）。

### 1.3 最小改动（`git diff -- release/sys1/layers/attention.py` 原文）[实测]

```diff
@@ -317,9 +317,10 @@ def attention_masked(
     heads, seq, dim = q3.shape
     scale = dim ** -0.5 if scale is None else float(scale)
     score = q3.to(torch.float32) @ k3.to(torch.float32).transpose(-1, -2) * scale
-    score = score.masked_fill(~window_band(seq, window, device=q.device), NEG_INF)
+    band = window_band(seq, window, device=q.device)[0]        # (1,T,T)：与 score 同秩，masked_fill 不升维
+    score = score.masked_fill(~band, NEG_INF)
     out = torch.softmax(score, dim=-1) @ v3.to(torch.float32)
-    return AttnOut(out.reshape(lead_shape(q) + (dim,)) if q.dim() > 3 else out, {
+    return AttnOut(out.reshape(lead_shape(q) + (seq, dim)), {
         "route": "masked", "window": window, "seq": seq, "heads": heads,
@@ -388,7 +389,7 @@ def attention_band(
-    return AttnOut(out.reshape(lead_shape(q) + (dim,)) if q.dim() > 3 else out, stats)
+    return AttnOut(out.reshape(lead_shape(q) + (seq, dim)), stats)
```

要点：
- **`window_band` 本体不动**（对外仍 `(1,1,T,T)`；`sliding_visible`/主干按此秩消费，改它会外溢）。
  只在喂 `masked_fill` 处取 `[0]` ⇒ `(1,T,T)` 与 `(H,T,T)` 同秩，广播不再升维。
- `lead_shape(q) + (seq, dim)` 统一覆盖 2/3/4 维（3 维时 reshape 是 no-op view）。
- 数学语义零改动：可见集合、scale、softmax、matmul 次序一字未动（证据见 §1.5）。

### 1.4 前后对照（派单要求）

**改前**（HEAD 实现；现态可用进程内打桩复现，不必回滚文件）：
`cd release && .venv/bin/python ascend/kernels/reconcile/E_probe_negative.py` [实测]

```
[RESPONDED] test_attention_routes_preserve_rank[2]: AssertionError: masked: 2 维入 (24, 8) → 出 (1, 1, 24, 8)
[RESPONDED] test_attention_routes_preserve_rank[3]: AssertionError: masked: 3 维入 (2, 24, 8) → 出 (1, 2, 24, 8)
[RESPONDED] test_attention_routes_preserve_rank[4]: RuntimeError: shape '[3, 2, 8]' is invalid for input of size 1152
[RESPONDED] test_batched_4d_matches_per_sample_loop[masked|band_kernel|band_local]: RuntimeError: shape '[3, 2, 8]' is invalid for input of size 1152
```
改前派单原始签名（D 探针，本代理在改前工作副本上实跑过）[实测（改前态）]：
```
cd release && .venv/bin/python ascend/kernels/reconcile/D_probe_routes.py shapebug
  attention_masked.out   = (1, 2, 200, 16)   # 3 维入 → 4 维出（升维）
  4 维入参             = RuntimeError: shape '[1, 2, 16]' is invalid for input of size 6400
```
改前 `E_probe_shapes.py matrix` 全档表（本代理改前采集）[实测（改前态）]：
```
masked  rank=2: (32, 4) -> (1, 1, 32, 4)   rank=3: (2, 32, 4) -> (1, 2, 32, 4)   rank=4: CRASH shape '[1, 2, 4]' is invalid for input of size 256
band    rank=2: (32, 4) -> (1, 32, 4)      rank=3: OK                             rank=4: CRASH
```

**改后现态** [实测]：
```
$ .venv/bin/python ascend/kernels/reconcile/D_probe_routes.py shapebug
  attention_masked.out   = (2, 200, 16)  # AttnOut 声明 (heads,seq,dim)
  attention_band.out     = (2, 200, 16)
  4 维入参             = (1, 2, 200, 16)（期望 (1,2,200,16)）

$ .venv/bin/python ascend/kernels/reconcile/E_probe_shapes.py matrix
  masked window=   8  rank=2/3/4: OK  (32,4)->(32,4) / (2,32,4)->(2,32,4) / (1,2,32,4)->(1,2,32,4)
  masked window=None  rank=2/3/4: OK  （同上）
  band kernel          rank=2/3/4: OK  （同上）
  band local           rank=2/3/4: OK  （同上）
  batched vs per-sample loop: shape_eq=True bitwise_eq=True max_abs=0.0   （三条路各自成立）
```
⇒ **12/12 档位秩保持；批 = 逐样本位等**。

### 1.5 数学未变（同一性证据）[实测]

| 指标 | 改前 | 改后 |
|---|---|---|
| 3 维出口展平字节 `sha1_16`（固定 seed） | `5a7d9530e2ae0a32` | `5a7d9530e2ae0a32` |
| `compare_routes(200,32,chunk=48)` kernel_vs_masked | `0.0` | `0.0` |
| 同 local_vs_masked | `1.7881393432617188e-07` | `1.7881393432617188e-07` |

复跑：`cd release && .venv/bin/python ascend/kernels/reconcile/E_probe_shapes.py compare`
（哈希取 `out.reshape(-1)` 的字节流 ⇒ 只改形状不改数值时才不变，正是本次的改动形态。）

### 1.6 回归钉：`release/tests/test_attention_shapes.py`（新建，7 用例）[实测]

| 用例 | 钉的内容 |
|---|---|
| `test_attention_routes_preserve_rank[2/3/4]` | masked / band_kernel / band_local 三条路，**入出同秩**（形状逐档相等 + 报错文案带回归形态） |
| `test_batched_4d_matches_per_sample_loop[masked/band_kernel/band_local]` | 4 维批入出同秩 **且与逐样本循环一致**（`allclose(atol=1e-6, rtol=1e-5)` + 同进程同形两次 `torch.equal`） |
| `test_masked_fill_band_is_same_rank_as_scores` | 根因级说明钉：同一 `(1,1,T,T)` 掩码直喂 ⇒ 升维、取 `[0]` 喂 ⇒ 同秩（把"为什么是 `[0]`"钉在测试里） |

```
$ .venv/bin/python -m pytest tests/test_attention_shapes.py -q
7 passed
```

模块 docstring 内嵌改前秩矩阵表与两处根因，并区分 [文档]（契约出处）与 [实测]（消费方预期秩来自 grep）。

### 1.7 `scratch/sys1/layers/attention.py`（冻结第一幕）：**只记不改** [实测]

- `ls scratch/sys1/layers/` → 仅 `README.md`，**无 attention.py**。
- `grep -rn "attention_masked\|window_band\|lead_shape" scratch --include="*.py" --exclude-dir=.venv` → **0 命中**。

⇒ **登记**：冻结第一幕不存在同缺陷件，无需勘误；本项**未做任何修改**。若后续归档件另有副本，
按派单口径走"勘误只登记不改动"。

---

## 2. 缺陷二：环境相关非确定的判据（测试缺陷，已修）

### 2.1 三道闸：所谓"内核带状路"在 CPU 上不存在 [文档 + 实测]

| 闸 | 判据 | 结果 |
|---|---|---|
| ① `backends.active_backend("cpu")` | `sys1/kernels/backends.py:104` 明写"只允许 mps 设备走方言" [文档] | → `torch_eager`（`D_probe_routes.py route` 打 `[gate1] ... False (actual='torch_eager')`）[实测] |
| ② `attn_sw_mps.plan` | 第一门槛要 fp16；`compare_routes` 造 fp32 [文档] | → `[gate2] plan(...) -> None` [实测] |
| ③ `BLOCK_Q=16` 整除 | `band_len = chunk+window-1 = 48+32-1 = 79`，`79 % 16 != 0` [实测] | 不进内核 |

⇒ 原断言比的是**两份 eager 实现、不同归约长度（200 列 vs 79 列）之间的位等**；对两条 eager 路要
`== 0.0` 逐位相等，而同文件姊妹路给 `1e-5` —— 判据自相矛盾且"位等"无 spec 授权（详见台账 §3 证伪三）。

### 2.2 派单"二选一"：选**措辞订正 + 措辞事实钉**，不补"真进内核的路"（理由与代价如实记下）

- [文档] `backends.py:104` ⇒ **CPU 永不进该内核**；要真进内核只能是 MPS 形态。
- CI 的 release-gates job 跑在 `ubuntu-latest`（`ci.yml` release-gates 段）[文档] ⇒ MPS 内核路在 CI **永不执行**；
  mac-gates 只跑 `scratch/tools/ci.sh --fast`，不跑 release 全量 [文档]。
- ⇒ 若补一条 `@pytest.mark.mps` 的内核路，结果就是**新增一条 CI 上恒 skip 的守卫**，
  恰好再留一笔"看起来有、实际不响"的账，与派单"别留名为内核实为 eager 的账"的精神冲突 [推断]。
- **处置**：措辞按实际订正（"内核带状路"→ `带状 route="kernel" 路（CPU 上执行体是 attn_sw_kernel._eager）`），
  并加一条**措辞事实钉**（`assert backends.active_backend("cpu") == backends.TORCH_EAGER`）——
  若将来有人真让 CPU 走内核而改了判据措辞，这条钉会先响。**代价**：内核实现体本身仍无 CPU 可达守卫，
  该守卫属 `sys1/kernels/**` 的 MPS 设备门工作（本波白名单外），已在 §6 登记为遗留。

### 2.3 新判据（`release/tests/test_longctx.py` C1 段，`:634-759`）[文档（落盘原文）]

主判据 = **平台无关结构判据**三条，量级容差降为附列：

| 编号 | 判据 | 平台无关性来源 |
|---|---|---|
| A1 | **非零可见位集合逐格相等**：`torch.equal(w != 0, truth.expand_as(w))`，含"未来位严格 0""对角自见" | 只比 0/非 0 的布尔格，不比浮点值 |
| A2 | **argmax 一致** + 自证门槛（见 §2.4） | 排名而非数值，末位差只有在"近并列"时才能翻转 ⇒ 门槛自证不是近并列 |
| A3 | **同进程自比**：每条路与**自身重跑**位等（重跑前做一次无关张量分配扰动分派器） | 同形同分派 ⇒ 位等有构造保证；不成立即真非确定 |
| 附列 | `compare_routes(200,32,chunk=48)` 的 `*_vs_masked_max_abs <= ROUTE_TOL(1e-5)` | **不再当主判据**（原 `== 0.0` 位等口径废除） |

权重抽取技巧：探针用 `v = I`（`dim == seq`）⇒ 出口最后一维**逐位就是**该路权重 P，零侵入（不改产品件）。
探针形状 `PROBE_SEQ/WIN/CHUNK/HEADS = 48/12/16/2`，`PROBE_SEED = 20261011`。

新增负例发生器 `test_sliding_visible_set_criterion_detects_reversed_window`：A1 必须对
"方向抄反（`delta<=0 & delta>-W`）"与"窗宽 off-by-one（`W+1`）"判否 —— 防 A1 沦为摆设（R14）。

**用例函数名保持不变** `test_sliding_routes_agree_with_mask_reference`：`ci.yml:35-38` 的注释按该名引用，
而 ci.yml 是禁改区 ⇒ 名字改不动，改用**内部措辞 + 事实钉**消账（此为本代理自行决策，见 §6）。

### 2.4 处方残留的平台敏感面（如实上报，非静默照做）

派单给的"结构判据"本身仍有两个平台敏感面，本代理在实现里补了自证门槛并改了判据形式：

1. **"非零"依赖 softmax 结果不落到 fp32 下溢**：若可见位权重极小，`!=0` 集合可能平台抖动。
   ⇒ 加门槛 `assert min_visible > 1e-6`（实测 `1.106e-03`，余量 3 个数量级）。
2. **argmax 在"近并列"时可被 3 ULP 末位差翻转** ⇒ 加自证门槛 `margin > gap_floor`，
   其中 `margin = min(top1 - top2)`（实测 `1.656e-03`），
   `gap_floor = 100 * max(probe_diff, 10*eps)`，`probe_diff` = **本平台本次实测**的跨路最大末位差（实测 `0.000e+00`）。
   ⇒ 门槛跟着本平台实测走：若哪天出现 `1e-7` 级末位差，门槛升到 `1e-5` 量级仍能守住两个数量级余量，
   而不是写死绝对值把自家平台打红（本代理第一版就写了绝对门槛 `100*ROUTE_TOL=1e-3`，被自家实测
   `9.220e-04` 打红，见 §6 错误记录）。
   **这是"若处方本身有问题则上报"的落点**：绝对门槛在跨平台（x86 OpenBLAS 有末位差、arm Accelerate 无）
   下不可保证，改成相对本平台实测的门槛后，判据仍是"结构为主、量级为附列"，未违反"放宽容差替代修判据"
   的红线（放宽容差=放宽量级口径；这里放宽的是**门槛的自适应方式**，主判据换成结构判据）。

### 2.5 自证跑表（派单要求逐条列，不得只报"N 次通过"）

**A. 连跑 20 次**：`cd release && .venv/bin/python -m pytest "tests/test_longctx.py::test_sliding_routes_agree_with_mask_reference" -q -p no:cacheprovider`

```
run01: 1 passed in 0.96s   run09: 1 passed in 0.93s   run17: 1 passed in 0.91s
run02: 1 passed in 0.90s   run10: 1 passed in 0.91s   run18: 1 passed in 0.93s
run03: 1 passed in 0.91s   run11: 1 passed in 0.90s   run19: 1 passed in 0.91s
run04: 1 passed in 0.91s   run12: 1 passed in 0.90s   run20: 1 passed in 0.91s
run05: 1 passed in 0.92s   run13: 1 passed in 0.91s   ── 合计 20 passed / 0 failed
run06: 1 passed in 0.94s   run14: 1 passed in 0.91s
run07: 1 passed in 0.92s   run15: 1 passed in 0.91s
run08: 1 passed in 0.93s   run16: 1 passed in 0.91s
```

**B. `OMP_NUM_THREADS` 三档（各 5 次，同法 + `-s` 抓实测行）**：
`OMP_NUM_THREADS=$t MKL_NUM_THREADS=$t .venv/bin/python -m pytest "tests/test_longctx.py::test_sliding_routes_agree_with_mask_reference" -q -s`

```
OMP=1 run1: 1 passed in 0.91s | min_visible=1.106e-03 argmax_margin=1.656e-03 probe_diff=0.000e+00 gap_floor=1.192e-04
OMP=1 run2: 1 passed in 0.90s | min_visible=1.106e-03 argmax_margin=1.656e-03 probe_diff=0.000e+00 gap_floor=1.192e-04
OMP=1 run3: 1 passed in 0.91s | min_visible=1.106e-03 argmax_margin=1.656e-03 probe_diff=0.000e+00 gap_floor=1.192e-04
OMP=1 run4: 1 passed in 0.91s | min_visible=1.106e-03 argmax_margin=1.656e-03 probe_diff=0.000e+00 gap_floor=1.192e-04
OMP=1 run5: 1 passed in 0.91s | min_visible=1.106e-03 argmax_margin=1.656e-03 probe_diff=0.000e+00 gap_floor=1.192e-04
OMP=4 run1: 1 passed in 0.91s | min_visible=1.106e-03 argmax_margin=1.656e-03 probe_diff=0.000e+00 gap_floor=1.192e-04
OMP=4 run2: 1 passed in 0.90s | min_visible=1.106e-03 argmax_margin=1.656e-03 probe_diff=0.000e+00 gap_floor=1.192e-04
OMP=4 run3: 1 passed in 0.91s | min_visible=1.106e-03 argmax_margin=1.656e-03 probe_diff=0.000e+00 gap_floor=1.192e-04
OMP=4 run4: 1 passed in 0.90s | min_visible=1.106e-03 argmax_margin=1.656e-03 probe_diff=0.000e+00 gap_floor=1.192e-04
OMP=4 run5: 1 passed in 0.91s | min_visible=1.106e-03 argmax_margin=1.656e-03 probe_diff=0.000e+00 gap_floor=1.192e-04
OMP=8 run1: 1 passed in 0.90s | min_visible=1.106e-03 argmax_margin=1.656e-03 probe_diff=0.000e+00 gap_floor=1.192e-04
OMP=8 run2: 1 passed in 0.91s | min_visible=1.106e-03 argmax_margin=1.656e-03 probe_diff=0.000e+00 gap_floor=1.192e-04
OMP=8 run3: 1 passed in 0.91s | min_visible=1.106e-03 argmax_margin=1.656e-03 probe_diff=0.000e+00 gap_floor=1.192e-04
OMP=8 run4: 1 passed in 0.90s | min_visible=1.106e-03 argmax_margin=1.656e-03 probe_diff=0.000e+00 gap_floor=1.192e-04
OMP=8 run5: 1 passed in 0.90s | min_visible=1.106e-03 argmax_margin=1.656e-03 probe_diff=0.000e+00 gap_floor=1.192e-04
```
⇒ 15/15 绿，且四项实测数值**跨线程数逐字相同**（本档位不改变分派）[实测]。
**注意**：`probe_diff=0.000e+00` 是本机（arm64, torch 2.14.1 CPU）事实，**不代表 ubuntu/OpenBLAS**；
新判据不再依赖该值为 0（`gap_floor` 自适应 + 主判据是结构判据），这正是本次重建的意义。

**C. 全量无回归**：`cd release && .venv/bin/python -m pytest tests -q -m "not integration"` [实测]
```
340 passed, 3 skipped, 3 deselected, 2 warnings in 35.87s
```
基线 `332 passed, 3 skipped, 3 deselected` + 新增 8（`test_attention_shapes.py` 7 + `test_longctx` 负例 1）
= 340，**零回归、零新增 skip**。

---

## 3. 缺陷三：台账订正（`docs/ci_baseline_triage.md`）

订正工具（定点替换、位点唯一性断言）：`release/ascend/kernels/reconcile/E_probe_edit_triage.py`
`--check` → [实测] 4 个位点各"命中 1 处"；`--apply` → `已订正 .../docs/ci_baseline_triage.md`。

| 位点 | 原文（被证伪结论） | 现文 |
|---|---|---|
| §0 速览行 | 归因"环境伪影（KNOWN-ENV，数值平台敏感）"、预期"仍红 ×1" | 归因"**判据缺陷**（原记环境伪影已被证伪，见 §3 订正）"、预期"不再贡献红" |
| §3 全节 | "KNOWN-ENV ×1 … 预期保留这一条红" | "判据缺陷 ×1（**已修**）" + 订正声明 + 证伪一/二/三 + 处置清单 + 新预期 |
| §5 条件 2 | "唯一红仅剩 §3 的 KNOWN-ENV 一条" | 作废；改为"唯一红须是新的、已按 R-P1-4 先证后判的条目" |
| §5 line ~115 | 预期稳态 `1 failed (仅 §3 KNOWN-ENV), 290 passed, 38 skipped` | `0 failed` + D 的 6-run 表事实（4 绿 2 红；`38070929040` 汇总行已 `0 failed`；run1↔run2、run4↔run5 之间 `release/sys1` 一字节未变却红绿互换）+ host 基线 `332 → 340` + **ubuntu 绝对计数不作预测** |

链出的证据件：`D_determinism.md`（6-run 表、三道闸读码、§4.8 姊妹路 10/20 出末位差）、本件 `E_RESULT.md`。
`diff --stat`：68 insertions / 18 deletions（**只动上述四处**；§1/§2/§4/§6 未改，§6 遗留见 §4）。

---

## 4. 遗留与待裁决（不在白名单内，未动）

1. **`docs/ci_baseline_triage.md` §6 两处数字/措辞随本次改动变陈旧** [文档]：
   - `:169` "`test_longctx.py`（门+KNOWN-ENV 注释）" ⇒ 现该文件内已**无** KNOWN-ENV 字样
     （`grep -rn "KNOWN-ENV" release/tests release/sys1` → [实测] 无残留）；
   - `:174` "全量自证 → `332 passed, 3 skipped, 3 deselected`" ⇒ 现为 `340`。
   派单只授权 §3/§5 ⇒ **未改，登记待裁决**（一句话可订正）。
2. **`.github/workflows/ci.yml:35-40` 注释仍写着被证伪的 KNOWN-ENV 归因与翻转条件** [文档]。
   禁改区 ⇒ **待编排者翻门禁时一并订正**；本代理未触碰。
3. **MPS 带状内核体在 CPU 无可达守卫**（见 §2.2 代价）。
4. **既有 ruff 告警，非本次引入**（同一 `--stdin-filename` 路径上下文对比 HEAD 版）[实测]：

   | 位置 | 规则 | HEAD 版是否同样存在 |
   |---|---|---|
   | `sys1/layers/attention.py:176` | TRY004 | 是（`:176`） |
   | `sys1/layers/attention.py:285` | I001 | 是（`:285`） |
   | `sys1/layers/attention.py:361` | I001 | 是（`:360`，本代理 +1 行位移） |
   | `tests/test_longctx.py:308` | F841 `r1` | 是（`:308`） |

   复跑：`cd release && git show HEAD:release/sys1/layers/attention.py > /tmp/base_attn.py && cat /tmp/base_attn.py | uvx ruff@latest check --output-format concise --stdin-filename sys1/layers/attention.py -`
   本代理新建/重写的文件（含两个 splice 工具）`uvx ruff@latest check` → **All checks passed**（`ISC004` ×3 已括号化，
   并用 `REPLACEMENTS` 8 个字符串 sha1 前后逐字相同验证格式修复未改替换语义）。

---

## 5. 自行决策 / 偏离（派单未逐项指明，但为完成任务所必需）

| # | 决策 | 依据与代价 |
|---|---|---|
| 1 | `attention_band` 出口**一并**同修（派单只点名 `attention_masked`） | 同一行缺陷文本（`lead_shape(q)+(dim,)`）；不修则 band 路 4 维仍崩、2 维仍升维，回归钉的 `[band_*]` 参数档无法成立。改动同为"补 seq 维"，数学不变（§1.5 compare_routes 差值原样） |
| 2 | 不改 `window_band` 本体，改喂法取 `[0]` | `window_band` 对外 `(1,1,T,T)` 是既有契约，`sliding_visible`/主干按此消费；改本体属外溢 |
| 3 | 用例函数名**不**订正，只订正措辞并加事实钉 | `ci.yml` 注释按该名引用且属禁改区 |
| 4 | "补真进内核的路"→ 选"措辞订正" | 见 §2.2 三条理由与代价 |
| 5 | A2 argmax 门槛写成**相对本平台实测**（`gap_floor`）而非绝对值 | 见 §2.4；绝对门槛会被自家数据打红（已发生一次） |
| 6 | 批 vs 逐样本主判据用 `allclose(atol=1e-6, rtol=1e-5)`，位等只用于同进程同形两次 | 跨形状（3 维循环 vs 4 维批）归约长度可不同 ⇒ 位等不是产品承诺（同一 D 代理结论的应用） |
| 7 | 台账 §0 与 §5 条件 2 与 line ~115 **一并**订正 | 三者共享同一被证伪前提；只改 §3 会让本档自相矛盾 |
| 8 | 新建两个一次性 splice 工具 `E_probe_edit_c1.py` / `E_probe_edit_triage.py` | 保证"只动这一块"（`assert old in s` + 位点唯一性）；同时是判据原文/台账原文的可复跑留档。`triage` 工具已 apply ⇒ 重跑 `--check` 会报 0 命中（幂等保护，防止二次改写） |
| 9 | 探针落 `ascend/kernels/reconcile/E_probe_*.py`（只读，不写产品件） | 白名单允许新建 `reconcile/E_probe_*.py`；`E_probe_negative.py` 用进程内打桩复现改前形态，**不需要**回滚文件即可复核 |

---

## 6. 错误与失败尝试（含已解决）

1. `zsh` 不认 `--include=*.py`（未加引号）→ `no matches found` ⇒ 改 `--include="*.py"` / 用 Grep 工具。
2. **自家 argmax 绝对门槛打红自家**：
   `AssertionError: argmax 判据前提破了：top1-top2 间距 9.220e-04 与量级容差 1e-05 不成量级差`（门槛 `100*ROUTE_TOL=1e-3`）
   → 扫 15 组 `(seed, seq, win, chunk, sharpness)`：锐化分数会压低 `min_visible` 反而制造下溢敏感面 ⇒ 弃用 sharpen；
   选定探针 `(T=48, W=12, chunk=16)` + 相对门槛 ⇒ 复跑 9 passed。
3. `test_attention_shapes.py::_qkv` `IndexError: base[2] is out of bounds for dimension 0 with size 2`
   （误按 head 索引复用 q 的切片）→ 改为同一 generator 直接取三份同形张量。
4. `E_probe_negative.py` 自造 lambda 打桩行导致语法错（`Try 语句必须至少有一个 except/finally`）
   → 删除该行改为 `[N/A]` 说明；后续重构为两阶段打桩时批量替换误删 `seq` 变量 → 补 `seq, dim = q3.size(1), q3.size(-1)`。
5. **docstring 里未经核实的断言**：初稿写"`sys1/model.py` 侧 `forward_with_windows` 走 4 维批"
   → grep 复核后（4 维无消费者）**改写为诚实口径**，并把它转成"契约声明支持却一调即崩"的证据。
6. **台账草稿事实错误**：初稿写"四个 ubuntu run 实测都是 0 failed"
   → 依 D 的 6-run 表更正为"4 绿 2 红，其中 `38070929040` 汇总行已是 `0 failed`"（apply 前修正，未入库）。
7. `release/.venv` 无 ruff、PATH 无 ruff → 改用 `uvx ruff@latest`（网络取包，一次性）。

---

## 7. 产物清单与复跑命令索引

**修改**
- `release/sys1/layers/attention.py`（形状修复；`numstat` = `4 3`）
- `release/tests/test_longctx.py`（C1 段判据重建 `:634-759` + 新增负例用例；`git diff --numstat` = `121 7`）
- `docs/ci_baseline_triage.md`（§0 一行 / §3 全节 / §5 两处；`numstat` = `68 18`）

**新增**
- `release/tests/test_attention_shapes.py`（7 用例，缺陷一回归钉）
- `release/ascend/kernels/reconcile/E_probe_shapes.py`（秩矩阵 + 数值指纹，只读探针：`matrix` / `compare`）
- `release/ascend/kernels/reconcile/E_probe_negative.py`（R14 负例自证，进程内复刻改前实现）
- `release/ascend/kernels/reconcile/E_probe_edit_c1.py`（C1 段定点 splice 工具，内嵌判据原文）
- `release/ascend/kernels/reconcile/E_probe_edit_triage.py`（台账定点 splice 工具，内嵌台账原文）
- `release/ascend/kernels/reconcile/E_RESULT.md`（本件）

**复跑索引**
```bash
cd /Users/pengweiye/Documents/codes/system-one/release
.venv/bin/python ascend/kernels/reconcile/D_probe_routes.py shapebug      # 现态秩；改前签名见本文 §1.4
.venv/bin/python ascend/kernels/reconcile/E_probe_shapes.py matrix        # 12 档秩 + 批/逐样本
.venv/bin/python ascend/kernels/reconcile/E_probe_shapes.py compare       # sha1_16 指纹 + compare_routes 差值
.venv/bin/python ascend/kernels/reconcile/E_probe_negative.py             # 改前形态复现 + 判据是否真响（R14）
.venv/bin/python -m pytest tests/test_attention_shapes.py -q              # 7 passed
.venv/bin/python -m pytest tests/test_longctx.py -k sliding -q            # 9 passed, 1 skipped
.venv/bin/python -m pytest tests -q -m "not integration"                  # 340 passed, 3 skipped, 3 deselected
git status --porcelain -- .github/workflows/ci.yml scratch                # 空 = 禁改区未触碰
```
