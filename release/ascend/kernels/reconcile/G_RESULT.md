# G_RESULT — p2-13 G：把 F2 的 fp16 档接进卡窗 [4] 段（同窗两趟，不多开一窗）

执行代理：G（极小接线活）。日期：2026-10-11。
**只改一处**：`release/ascend/port910b/card_run.sh` 的 [4] 段（外加"汇总相关行"与钩子文档各一处）。
**新建**：本文。禁改件一个没碰，未执行任何 git 写操作。

- 改前 md5 `cf0c2bdf7daa2564cca024af2c8ceffe`（= B_RESULT §4 记录的 B 写面正身）
- 改后 md5 `5ef1e13a1546318a7c519156d4c7dacc`，`bash -n` OK，`git diff --stat` = **+43/-3**
- 判决行形状**零改动**（F2 §7 承诺"判决行格式已兼容"，本文逐条核）

---

## 1 改了什么（四处锚点，全部可 `git diff` 复核）

| # | 位置 | 改动 | 性质 |
|---|---|---|---|
| ① | 钩子清单（`SYS1_BENCH` 之后） | 新增 3 行文档：`SYS1_SKIP_FP16_LEG=1` 的语义与"不许静默消失"承诺 | 文档，零行为 |
| ② | `N_LEGS_4` 上方 | `FP16_PLANNED`（开关关上才计）+ `N_LEGS_4=$(( 2 + BENCH_PLANNED + FP16_PLANNED ))` | **汇总相关行**：计划腿数含新腿，否则 `STOP_AFTER=2/3` 分支会把"少跑一条腿"报成"一条没少" |
| ③ | `runner_done` 的 `LOGS …` 提示串 | 补看正则加 `\|^DTYPE-TRACE` | **汇总相关行**：纯字符串，改的是"窗内怎么读"的说明书，不是任何被执行的 grep |
| ④ | [4] 段 | fp32 腿**逐字不动**；其后加 fp16 腿（`--dtype fp16`，标签 `trainstep-fp16`）+ 显式砍腿分支 | 主改点 |

fp32 腿的**标签/超时/三个正则/命令**与改前逐字节相同（凭据见 §5.1 的 `diff`），
新腿挂在它后面，**不改顺序、不改 fp32 判决行、不碰 [3]/[4b]/[4c]/[5]**。

---

## 2 两趟的 grep 与退码语义（任务书点名的"判决行都要被现有 grep 抓到"）

### 2.1 两条腿的参数对照（`run_leg` 六参签名 = B_RESULT §3.1，一个槽位都没新造）

| 槽位 | fp32 腿（未动） | fp16 腿（新） |
|---|---|---|
| 标签 | `trainstep` | `trainstep-fp16`（日志 `$RUNLOG/trainstep-fp16.log`，tally 独立一行） |
| 超时 | 900s | 900s（腿级独立，两腿不共享额度） |
| disp_re | `^P2-STEP\|^KERNEL-USED\|^NPU-REG\|已回退 torch eager` | 同左 **+ `^DTYPE-TRACE`** |
| fail_re | `^P2-STEP-.*FAIL\|^KERNEL-USED-FAIL\|NPU-REG fail` | 同左 **+ `^DTYPE-TRACE-FAIL` + `^DTYPE-TRACE fp16 .*fp32_accum=false`** |
| pass_re | `^KERNEL-USED-PASS` | 同左（不加新形状：B 的"两样都要"口径原样继承） |
| deg_re | `-` | `-`（train_step **没有** `note=无设备⇒只判编译` 那种 PASS 形状的降级，那是 run_numerics 面；给它配 deg_re 就是臆造） |
| 命令 | `env DMLAYA_ASCEND_TARGET=ascend python3 ascend/train_step.py --target ascend` | 同左 **+ `--dtype fp16`** |
| track | gating | **gating**（产品面，严禁 advisory——见 §2.3） |

`DTYPE-TRACE` 走的是**现有两趟 grep 的第一趟**（`run_leg` 的 `grep -E "$disp_re" | head -30`），
`P2-STEP-ascend` / `KERNEL-USED-*` / `NPU-REG` 同处一列，三行一起上屏；第二趟错误 grep 的匹配词
`error:|FATAL|Traceback|RuntimeError|AssertionError` **一个字没动**，
**尤其没有把 `aicore` 加回来**（B 的 H2 病灶：它命中 bisheng 命令行 `-cce-aicore-*`，每件失败件白吃 3 行，
两件同灭时把判决行完全挤掉）。

### 2.2 退码 → 判决 的映射（`train_step` 优先级 2 > 3 > 6 > 5，全部落在 B 的执行器语义里）

| train_step 退码 | 含义 | fp16 腿的 verdict | 顶窗？ | 凭据 |
|---|---|---|---|---|
| 0 + `KERNEL-USED-PASS` | 真过了 | `PASS` | 否 | 矩阵 C1 |
| **6** | 位宽纪律钉违规（`fp32_accum=false`） | **`FAIL`**（fail_re 的 dtype 判据命中） | **是，fail++** | 矩阵 C2 |
| **6**（但判决行全绿） | 只改退码不改行 | **`FAIL-EXIT6`**（B 的 `rc≠0` 校验兜底） | **是** | 矩阵 C3 |
| 4 | 设备/注册不可用（无卡） | `FAIL`（`NPU-REG fail` + `P2-STEP-ascend FAIL`） | 是 | 容器干跑 + 矩阵 C5 |
| 2 | grad 缺失 | `FAIL`（`^P2-STEP-.*FAIL` 命中） | 是 | 容器干跑（见 §5.5 的意外实测） |
| 3 | loss 未下降 | `FAIL` | 是 | 同上判据 |
| 5 | 内核门禁（strict） | `FAIL`（`^KERNEL-USED-FAIL`） | 是 | F2 §1.2"三证同账本同退码" |
| 0 但零判决行 | 静默假绿 | `FAIL-NO-VERDICT` | 是 | 矩阵 C4 |

### 2.3 为什么 fp16 腿**必须** gating（任务书红线）

`run_leg` 的分账逻辑（B 写面）是 `LEG_TRACK=advisory ⇒ _bump_fail 进 ADV_N、不顶退码`。
advisory 的既有口径只覆盖 `NUM-*` / `PERF-*` 对照轨（Y：不可当正确性判决引用）。fp16 档是
**产品面**（真发 fp16 模具、真落位宽账），所以沿用 [4] 段既有的 `LEG_TRACK=gating` 上下文，
新腿**不重新赋值 track**。
这条不是装饰——变异体 **M3**（把新腿改成 advisory）实测：**11 条判据当场 MISS，整窗从 exit 1
退化成 exit 0**，正是任务书警告的"退码 6 被当成 advisory ⇒ 静默放行"。详见 §5.3。

### 2.4 为什么**没有**扩 [5] 的 H3 自扫正则（申报为自行决策，见 §6-③）

H3 的既有前缀族里**本来就没有 `^P2-STEP`**（B 写面如此），`DTYPE-TRACE` 也一样不归它管：
H3 的用途是"把腿内被 `head` 截断的行补回来"，而 train_step 一趟总共只出 4–5 行判决，
第一趟额度是 30，截断不了。反过来若把 `DTYPE-TRACE` 塞进 H3，**fp32 腿的日志也会多出两行**
（F2 无条件打 `DTYPE-TRACE fp32 …`），改前/改后对照就不再是"只多一条腿"了——那才叫越权。

---

## 3 fp16 腿在无卡时判什么（不静默 PASS，按 B 的既有语义）

容器（`cann910b-h`，有 bisheng、无设备）实测两趟都出判决行：
```
CASE-trainstep      VERDICT-FAIL rc=4 track=gating tally(pass=1 fail=5 adv=5) log=…/trainstep.log
CASE-trainstep-fp16 VERDICT-FAIL rc=4 track=gating tally(pass=1 fail=6 adv=5) log=…/trainstep-fp16.log
```
屏上判决行原文（两腿同形，fp16 没有多一条红、也没有少一条红）：
```
CASE-trainstep-fp16 track=gating tmo=900s
P2-STEP-ascend FAIL: NPU-REG-FAIL 设备名不可用 RuntimeError: Expected one of cpu, cuda, … npu（import torch_npu -> ModuleNotFoundError: No module named 'torch_npu'）
NPU-REG fail none target=ascend npu_avail=False import torch_npu -> ModuleNotFoundError: No module named 'torch_npu'
CASE-trainstep-fp16 VERDICT-FAIL rc=4 track=gating
```
**判 DEGRADED 还是 FAIL**：判 **FAIL**。任务书写"DEGRADED/FAIL 按 B 的既有语义"——B 语义里
DEGRADED 专指 `note=无设备⇒只判编译` 那种"PASS 形状的降级"，而 train_step 压根不出那种行
（它在 NPU 注册门就退 4，连 `KERNEL-USED-PASS` 都凑不出来）。给它配 deg_re 属于我臆造，故 `deg_re=-`，
与 fp32 腿逐字同。**要点是两种结果都不许静默绿**，实测确实不绿。

---

## 4 预算复算（A3 §4 的 20 分钟窗）——**默认跑两趟，不砍腿**

### 4.1 一手实测（本代理自己跑的，不是引用）
| 项 | 命令 | 读数 |
|---|---|---|
| fp16 档全图冷编（13 键） | 容器 `TILELANG_CACHE_DIR=/dev/shm/g_fp16cache timeout 900 python3 ascend/kernels/reconcile/F2_probe_card_fp16_step.py` | **wall=20s rc=0，F2-CARD-SUMMARY 4/4，冷编产物 13 枚** |
| fp32 档同机（A3 §3 原记录） | `A3-STEP wall=20.0s compile_count=13 real_compiles=13 sum_sec=19.1` | 20s / 13 键 |
| 整窗干跑 改前 | 容器 `bash card_run.sh`（before） | **WALL=45s** |
| 整窗干跑 改后（两趟） | 容器 `bash card_run.sh`（after） | **WALL=45s**（无卡时 fp16 腿在设备门前就退，≈0 成本） |
| 整窗干跑 显式砍腿 | 同上 + `SYS1_SKIP_FP16_LEG=1` | WALL=44s |

⇒ **fp16 档的编译面与 fp32 档同量级（20s/13 键，同机对同机）**，不是"更多"的一档。
键含位宽段（实测原文：`gemm_dw…dfloat16`、`gemm…dfloat16`、`rope…qfloat16afloat32`）
⇒ **与 fp32 不共享缓存，两趟各付一次冷编**，这正是任务书担心的点，量级已测：+20s。

### 4.2 换算到卡上（沿 A3 §4 的口径，只加"新增项"）
A3 原账：净链 2.5–6 分钟（npu 初始化 15 进程×3–15s ≈45–225s + 编译面 ≈33 件 56–80s + 真发射对拍 <30s）。
G 之后：多 **1 个进程**（+3–15s）、多 **13 件冷编**（+20–60s，卡上 CPU 若弱于容器按 3× 计）、
多 **1 次真发射对拍**（+<30s）⇒ **净链 3–7.5 分钟**，A3 建议的 **20 分钟窗（= 净链 6 + fixup 补跑 6 + 冷启/读日志 8）仍够**，
腿级 900s 超时对新腿的富余 ≈ 15×（按容器 20s 冷编 + 卡上发射余量估）。

### 4.3 兜底开关（不许静默砍腿）
`SYS1_SKIP_FP16_LEG=1` ⇒ 屏上打
`CASE-trainstep-fp16 SKIP reason=SYS1_SKIP_FP16_LEG=1 显式砍腿 ⇒ fp16 档本窗未验`，
`skipped_legs` +1（实测 1→2），`N_LEGS_4` 由 3 回落到 2（`FP16_PLANNED=0`，**不与 [4] 自己记的那条 SKIP 重复计账**），
其余计数与改前逐字同（实测 fail=5 与 before 一致）。**默认关闭**：不设该变量就一定跑两趟。

---

## 5 自证记录（全部可复跑，命令见 §8）

### 5.1 容器端到端干跑三态（真件、真脚本、零伪造）
```
BEFORE（git HEAD 的 card_run.sh，单趟）
  RUNNER-DONE SUMMARY pass=2 fail=5 skipped_legs=1 degraded=3 advisory_fail=5 exit=1   WALL=45s
AFTER （G 版，fp32+fp16 两趟）
  RUNNER-DONE SUMMARY pass=2 fail=6 skipped_legs=1 degraded=3 advisory_fail=5 exit=1   WALL=45s
SKIP  （G 版 + SYS1_SKIP_FP16_LEG=1）
  RUNNER-DONE SUMMARY pass=2 fail=5 skipped_legs=2 degraded=3 advisory_fail=5 exit=1   WALL=44s
```
**差异逐条**（这就是任务书纪律 4 要的 before/after SUMMARY 对照）：
- `fail` 5→6：**只多了 `CASE-trainstep-fp16` 一条腿**；`pass/skipped/degraded/advisory_fail/exit` 全不变。
- `tally.txt` 12 行 → 13 行，**新增且仅新增** `CASE-trainstep-fp16 VERDICT-FAIL rc=4 track=gating`；
  其余 12 行**逐字同**（`cat -n` 两侧对照已核）。
- fp32 腿原文日志 `diff /tmp/g_legs_before/trainstep.log /tmp/g_legs_after/trainstep.log` ⇒ **无输出（IDENTICAL）**。
- `BEFORE` 那一行与 **B_RESULT §5 记录的容器干跑结果逐字一致**（`pass=2 fail=5 skipped_legs=1 degraded=3 advisory_fail=5 exit=1`）
  ⇒ 本沙盒就是 B 的那套复现环境，不是新造的乐观环境。
- `[4c] KERNELS-TGZ` 三态都 `entries=224 kernels=32`——**注意口径**：无卡时两趟 train_step 都在设备门前退，
  压根没编东西，所以这组数字**不能**拿来宣称"卡上缓存不变"。卡上真实态是 fp32+fp16 **两套模具键**，
  `entries/kernels` 会比单档多（预计 kernels +13 量级），这是**卡上待验**，我不在此声称。

### 5.2 腿级判决矩阵（`B_probe_cardrun_verdict_matrix.py` 同款技法：原样摘录执行器 + [4] 段，喂真形状假载荷）
**22 条判据（6+5+3+2+3+3）/ 0 MISS / `G-LEG-DONE PASS`**。载荷里的每一行都**不是我手写的**：
- rc=6 的两行（`DTYPE-TRACE fp16 params=16 fp16=16 fp32_accum=false fallback=1`、
  `DTYPE-TRACE-FAIL: fp16 档归约违规 ['sum(fp16 直接归约: shape=(16, 32) dtype=None)'] ⇒ exit=6`）
  = 用 **F2 的 guard 注入法**（覆写 `LAST_AUDIT` 违规账）从**真 `train_step.py`** 跑出来的原文，
  真链路返回值 `HARNESS_RC=6`；
- rc=4 那两行 = 真件 `--target ascend --dtype fp16` 在本机的原文；
- `KERNEL-USED-PASS kernels=13 blockers=0 fallbacks=0 target=ascend` = **B_RESULT §6 钉死的"卡上应当逐字出现的形"**
  （这是**预期形状**，本地不可验，如实标注）。

| 例 | 场景 | 期望→实测 |
|---|---|---|
| C1 | fp32=PASS fp16=PASS | `pass=2 fail=0 … exit=0`，两腿标签/日志分离，`DTYPE-TRACE fp16` 上屏，`N_LEGS_4=3` |
| C2 | **fp16 腿 rc=6（真注入原文）** | `VERDICT-FAIL rc=6 track=gating`、`advisory_fail=0`、**整窗 exit=1** |
| C3 | fp16 腿行全绿但 rc=6 | `VERDICT-FAIL-EXIT6` → 仍顶窗（B 的 rc 校验兜底 = 第二重保险） |
| C4 | fp16 腿 rc=0 零判决行 | `VERDICT-FAIL-NO-VERDICT` → 顶窗（静默假绿不放过） |
| C5 | fp16 腿=真件无卡原文（rc=4） | `VERDICT-FAIL rc=4 track=gating` → 顶窗 |
| C6 | `SYS1_SKIP_FP16_LEG=1` | `SKIP reason=…` 上屏 + `skipped_legs=1` + `pass=1 fail=0 exit=0` + `N_LEGS_4=2` |

### 5.3 变异体自证（`G_CARDRUN_SRC` 指向被剪过的副本，判据一改探针必须响）
**3/3 抓住**（`G-SABOTAGE-SUMMARY 3/3`）：
- M1 抽掉 disp_re 的 `^DTYPE-TRACE` ⇒ 2 条 MISS（fp16 判决行与 `DTYPE-TRACE-FAIL` 都不上屏）
- M2 抽掉 fail_re 的两条 dtype 判据 ⇒ 1 条 MISS（`fp16 腿判 FAIL` 退化成只靠 rc）
- M3 把新腿改成 `advisory` ⇒ **11 条 MISS，含"整窗 exit=1"与"advisory_fail=0"**（正是任务书禁止的形态）

### 5.4 回归（B/F2 写面零破坏）
| 项 | 结果 | 基线 |
|---|---|---|
| `bash -n card_run.sh` | OK（改后 + 三个变异体都过） | — |
| `B_probe_cardrun_verdict_matrix.py`（**默认读我改后的正身**） | `B-CR-DONE PASS` | F2 §4 表 33 ok/0 MISS 同判 |
| `B_probe_h1_npu_reg.py` / `B_probe_kernel_usage_negative.py` | `B-H1-DONE PASS` / `B-KU-DONE PASS` | 一致 |
| `F2_probe_dtype_guard.py` | `F2-GUARD-SUMMARY 8/8` | 一致 |
| host `pytest tests -q -m "not integration"` | **340 passed, 3 skipped, 3 deselected, 2 warnings in 40.60s** | B/F2 基线逐字一致 |
| 真件 fp16 档 host 直跑（对照 F2 §3.1） | `loss0=0.910987 loss1=0.823871 reduced=True params_grad_ok=True`，`DTYPE-TRACE fp16 params=16 fp16=16 fp32_accum=true fallback=1`，exit 0 | F2 §3.1 表逐字同 |

### 5.5 跑过程中真踩的坑（不是设计验证，是工装 bug，如实登记）
第一版干跑脚本写成了 `VAR=1 VAR2=2 S=$(date +%s); bash card_run.sh`——**赋值串没有命令词，
只成为当前 shell 变量、不进环境**，于是 `SYS1_DIR`/`CARDRUN_SKIP_CLONE` 全部没生效，
card_run.sh 用默认 `SYS1=/tmp/sys1` **真 clone 了一次 origin/main**（60M，容器盘未受影响，df 仍 5.0–5.1G avail/87%），
拿到的是 **B 之后 F2 之前的 `train_step.py`（382 行，无 `--dtype`）**，结果：
```
CASE-trainstep-fp16 VERDICT-FAIL-NO-VERDICT rc=2 track=gating   （argparse usage 报错，零判决行）
RUNNER-DONE SUMMARY pass=2 fail=6 … exit=1
```
这次"事故"反而白送一条证据：**接错版本（产品件不认 `--dtype`）时，新腿判 `FAIL-NO-VERDICT` 并顶窗，
不会静默绿**——正是 B 的"没有判决行一律 FAIL"口径在新腿上的体现。已按 `env K=V … bash` 修正并重跑（§5.1）。
教训：**给卡窗写工装时，凡给子进程传变量一律 `env`，别信裸赋值**（容器里 `timeout` 同理不可缺，darwin 无 `timeout(1)`）。

---

## 6 我没做什么、为什么（含自行决策申报）

**③ 疑惑点与自行决策**
1. **没扩 [5] 段 H3 自扫正则**（`DTYPE-TRACE` 不在其中）：H3 本来也不含 `^P2-STEP` 族；扩它会让
   fp32 腿日志多出 `DTYPE-TRACE fp32` 行，破坏"改前/改后只差一条腿"的对照。判决行由**现有**两趟 grep
   的第一趟捕获（矩阵 C1 实测上屏）。若编排者要求 H3 也收 `DTYPE-TRACE`，是一行改动，但请一并把
   `^P2-STEP` 收进去，否则口径仍然不齐。
2. **新腿 `deg_re=-`（不配 DEGRADED 判据）**：train_step 无"note=无设备⇒只判编译"那种形状；
   任务书写的"DEGRADED/FAIL 按 B 的既有语义"我理解为"二选一，不许静默 PASS"，实测走 FAIL(rc=4)。
3. **pass_re 不加 `^DTYPE-TRACE`**：`rc=0` 已蕴含无违规（F2 的退码链在 `violations` 非空时必返 6），
   再叠一条同义判据只增加"两条腿形状不一致"的读者负担。真违规有 **两条独立红路**（fail_re 判决级 + rc 校验），
   C2/C3 分别证到。
4. **`SYS1_SKIP_FP16_LEG` 默认关**：预算复算结论是"够用"（§4），开关只作兜底；
   加它的同时把它接进 `N_LEGS_4`，是为了 `STOP_AFTER=2/3` 分支的 `skipped_legs` 仍诚实。
5. **工装落在 `.tmp_a1/`（gitignored）与容器 `/tmp`**，没在仓库里新增脚本——白名单只允许
   `card_run.sh` + 本文；`G_*` 探针是"跑得出凭据"的必需品，故放既有沙盒惯例目录（A1/B/F2 同款）。
6. **卡上真值仍待下窗**：本文所有"卡上应当出现的形"（`KERNEL-USED-PASS …`、fp16 档 loss 轨迹、
   `[4c] entries` 增长）都是**预期**，本机与容器都无 npu，我不声称已验。

**④ 偏离记录**
1. 任务书写"F2 实测 fp16 档 host 13 枚模具编译 **19.1s**"：`F2_RESULT.md` 全文**无任何秒数**（`grep -nE "sec|秒|[0-9]+\.[0-9]s"` 命中 0）。
   `19.1s` 的真实出处是 **`A3_PREFLIGHT.md §3`：容器冷缓存下 **fp32** 档 `13 次真编译合计 19.1s / 整步墙钟 20.0s`**。
   我没有沿用这个被转错的归属，改为**自己实测 fp16 档同机读数 `wall=20s / 13 键`**（§4.1），
   结论不变（两档同量级），但账是真的。**fp32 与 fp16 的键不同 ⇒ 缓存不共享 ⇒ 两趟各付 20s**，这一点任务书说得对。
2. 除 [4] 段外还动了 ② `N_LEGS_4`（汇总相关，任务书纪律 1 明确允许）与 ③ `LOGS` 提示串（同属汇总相关，
   纯字符串零行为）+ ① 钩子文档 3 行（新功能登记，零行为）。
   若判定"文档行也算越权"，删掉①即可，不影响任何判决与计数。

**没做的事（明确不扩权）**
- 没改 `train_step.py`/`autograd_asc.py`/`gemm_kernel.py`/任何 `*_asc.py`/`compile_all_asc.py`/
  `run_numerics.py`/`bench_card.py`/trunk/`F2_*`/`B_RESULT.md`/**任何 tasks.md**；没执行 git 写操作。
  （`git status` 与开工时一致：`M autograd_asc.py / M kernels/gemm_kernel.py / M train_step.py` + F2 未跟踪件 + 本次的 `M card_run.sh`。）
- 没加"卡上 fp16 数值断言"（loss 阈值、fp16 与 fp32 差值判据）——那是编排者/数值面的活，
  接线活加数值判据就是把一次开窗变成一次实验，且阈值无出处。
- 没为 fp16 腿单开一窗、没挪 [3] 任何腿、没调任何 timeout 额度（新腿沿用与 fp32 腿同的 900s）。
- 没顺手把 `SYS1_KERNEL_USAGE_STRICT=1` 塞进 fp16 腿（F2 §7 把它列为"卡窗应归零"的**观察项**，
  不是本任务授权；一旦设了它，cpu 轨已知回落会顶红，属越权改门）。
- 没把 fp16 腿改成 advisory，也没把 `aicore` 加回任何 grep。

---

## 7 窗内怎么读（下窗照这五行判）

```bash
# ① 两趟都在、各自判决（标签必须能区分）
grep -E '^CASE-trainstep( |-fp16)' $RUNLOG/tally.txt
#   CASE-trainstep      VERDICT-<…> rc=<…> track=gating
#   CASE-trainstep-fp16 VERDICT-<…> rc=<…> track=gating     ← 少了这行 = fp16 档根本没跑
# ② fp16 档三行齐（缺 DTYPE-TRACE 就是跑的不是 fp16 腿）
grep -E '^P2-STEP|^KERNEL-USED|^DTYPE-TRACE' $RUNLOG/trainstep-fp16.log
#   DTYPE-TRACE fp16 params=16 fp16=16 fp32_accum=true fallback=0   ← fp32_accum=false ⇒ rc=6 ⇒ 腿级 FAIL
# ③ 位宽纪律钉有没有咬人
grep 'DTYPE-TRACE-FAIL' $RUNLOG/trainstep-fp16.log      # 出现 ⇒ 该腿 VERDICT-FAIL，整窗 exit 1
# ④ 是不是被我显式砍腿了（不许静默消失）
grep 'CASE-trainstep-fp16 SKIP' <窗日志>                 # 出现 ⇒ 本窗 fp16 档未验，别当成过了
# ⑤ 总账
grep 'RUNNER-DONE' <窗日志>                              # fail>0 ⇒ exit 1（含 fp16 腿）
```

卡上预期（**待逐字对照，本文不声称已验**，口径同 B_RESULT §6）：

| 位置 | 卡上应当出现的形 | 若不符说明什么 |
|---|---|---|
| [4] fp32 腿 | `KERNEL-USED-PASS kernels=13 blockers=0 fallbacks=0 target=ascend` | `kernels<13` ⇒ 有件回落（与 B §6 同判） |
| [4] fp16 腿 | 同上行 + `DTYPE-TRACE fp16 params=16 fp16=16 fp32_accum=true` | 出 `fp32_accum=false`/`DTYPE-TRACE-FAIL` ⇒ **位宽纪律钉抓到现场**（F2 §3.5 那类组装面 fp16 内积回落）；`fallback>0` ⇒ fp16 档有件落了 eager |
| [4] fp16 腿 | `P2-STEP-ascend loss0=… loss1=… reduced=True` | `reduced=False` ⇒ rc=3；`params_grad_ok=False` ⇒ rc=2（优先级都在 6 之前） |
| [4c] | `KERNELS-TGZ … entries/kernels` 比单档多（两套模具键） | 与 fp32 档数字一致 ⇒ **fp16 腿根本没编译**（多半在设备门前就退了） |
| [5] | `RUNNER-DONE SUMMARY pass=… fail=0 … exit=0` | 任何 `fail>0 ⇒ exit 1`，含 fp16 腿的 rc=6 |

---

## 8 复跑配方

```bash
SRC=$REPO_ROOT
# ① 语法与改动
cd $SRC && bash -n release/ascend/port910b/card_run.sh && git diff release/ascend/port910b/card_run.sh
# ② 真件取证：干净 fp16（exit 0）+ F2 注入法真 rc=6
python3 $SRC/.tmp_a1/g_probe_rc6_real.py                  # G-RC6-SUMMARY 5/5
# ③ 腿级判决矩阵（22 判据，6+5+3+2+3+3）+ 变异体自证（3 例必须响）
python3 $SRC/.tmp_a1/g_probe_leg_matrix.py                # G-LEG-DONE PASS
python3 $SRC/.tmp_a1/g_probe_sabotage.py                  # G-SABOTAGE-SUMMARY 3/3
# ④ 回归（B/F2 写面 + 单测）
cd $SRC/release && .venv/bin/python ascend/kernels/reconcile/B_probe_cardrun_verdict_matrix.py  # B-CR-DONE PASS
.venv/bin/python -m pytest tests -q -m "not integration"    # 340 passed
# ⑤ 容器端到端三态干跑（§5.1 的出处；先跑 .tmp_a1/g_sync_card.sh 送件）
bash $SRC/.tmp_a1/g_dryrun_card.sh
# ⑥ fp16 档编译预算实测（§4.1 的出处）
bash $SRC/.tmp_a1/g_measure_fp16_compile.sh                # G-FP16COMPILE wall=20s，F2-CARD-SUMMARY 4/4
```
