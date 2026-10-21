# N_traceability · README 门面数字逐条溯源表（代理 N，2026-10-10）

> 口径：写进 `README.md` 的**每一个数字/状态断言**都要能落到 run-id、文件行或一条可复跑命令。
> 复核环境：本机 macOS（Apple Silicon），HEAD = `365ccc1`（工作区含同波并行代理 O/P 的**未入库**改动，凡受影响处均已注明）。
> 复跑命令全量见文末附录；表中"实测"= 本次会话真实输出。

## 一、第一幕（scratch/）

| # | README 写入值 | 来源（run-id / 文件行 / 命令） | 实测输出摘要 |
|---|---|---|---|
| 1 | 参数**总 54.2M（54,213,632）** | `scratch/learning/configs/mps_main.yaml:21-30`（d=512/L=12/heads=8/vocab=16000/ffn_mult=4）实例化 `sys1.model.Decoder` 逐参数求和；与审计 §2.3.2 手工核算同值 | `total 54213632 54.2M` |
| 2 | 参数**非嵌入 37.8M** | 同上，剔除 `tok_emb`+`lm_head`（各 8,192,000） | `non-emb 37829632 37.8M`（嵌入合计 16,384,000） |
| 3 | 注意力是 **MHA、无分组 kv 头**（旧门面那个架构名词删词） | `scratch/sys1/model.py:205` `self.qkv = nn.Linear(config.d, 3 * config.d)`；`grep -rn "GQA\|kv_head\|n_kv" scratch/sys1/model.py` → **0 命中** | 行号 205 原文在场，grep 无命中 |
| 4 | **A–Z=38..63**（连续）、a–z=70..95，52 枚字母各单 token | `scratch/runs/1003-s0-bpe-16k-realedu-zh-en/tokenizer/vocab.json`（自训 BPE 产物） | `A-Z ids: 38 .. 63 连续= True` / `a-z ids: 70 .. 95` |
| 5 | 第二幕 Qwen3.5 分词器字母段 **32–57**（归属标注，不与 #4 互抄） | `release/tests/test_assets.py:192` `assert ids == tuple(range(32, 58))`；run `1005-p202-teacher-adapters-b3-realpath-b570/notes.md:5`（"首末号 [32, 57]"） | 断言行在场；notes 原文命中 |
| 6 | choice **+31.67pp**（0.6500 vs 随机 1/3，k=3 子桶 **n=40**，全轴 n=100）/ noul **+25.00pp**（0.7500 vs 0.5000，n=60） | run `1006-eval-report-aa36/report.md:7-8`；`metrics.jsonl:1`（`acc_choice_k3=0.65, base=0.333333, acc_noul_k2=0.75, base=0.5`） | `choice 3 40 0.6500 0.3333 +31.67pp` / `noul 2 60 0.7500 0.5000 +25.00pp` |
| 7 | 参考线 = 自定常数 **10.0pp**（非竞品基线） | `scratch/sys1/eval/quality.py:39` `REFERENCE_MARGIN_PP = 10.0` | 行号 39 命中 |
| 8 | **parity 20/20**（20 样本；口径为手工装配前向） | `1006-eval-report-aa36/report.md:20` `parity argmax 一致 20/20 (100.0%)`；口径出处 `scratch/sys1/eval/parity.py:74 class KernelForward` | 报告行原文命中 |
| 9 | **P50 5.8ms** | `1006-eval-report-aa36/metrics.jsonl:1` `e2e_p50_ms: 5.817`（MPS eager 路径） | 字段值 5.817 |
| 10 | 内核链经 `Decoder.forward` **走不到**（掩码门） | `model.py:386`（无条件 `_visible_mask(...)`）+ `model.py:211`（`if ... visible is not None ...: return False`） | 两处行号原文命中 |
| 11 | TileLang Metal 热态**慢 torch-MPS 2–6×** | run `1004-bench-tilelang-vs-torch-mps-1a18` notes 结论行 | "tilelang 0.1.15 Metal 后端在 M3 Pro 上全面慢于 torch-MPS 2–6×" |
| 12 | 可微链训练端到端 **−8.7%**（3,048 vs 3,315 tok/s）；推理端到端 **−12%**（0.88×） | run `1004-bench-kernel-autocograd-30step-28ba` notes 结论行；run `1004-bench-infer-e2e-mps-ce64` notes 结论行 | "端到端 -8.7%（3,048 vs 3,315 tok/s）"；"TileLang 主干前向 0.88×（11,497 vs 13,017 tok/s）" |
| 13 | `repro_p1.sh` 裸克隆 **exit 2 诚实拒跑**（需外部三集转写；`bench/` 不在库） | `.gitignore:10` `bench/`；审计 §2.2.1（A 级实测 REPRO_RC=2）；提示语在脚本阶段 0 | .gitignore 第 10 行命中 |
| 14 | **exit 0 不含质量阈值** | `scratch/sys1/eval/report.py:99-109` verdict `passed` 判据 + note "质量/把握为呈现值，不参与退出码判定" | 代码块原文命中 |

## 二、第二幕（release/）

| # | README 写入值 | 来源 | 实测输出摘要 |
|---|---|---|---|
| 15 | 文本教师实名 **`StartLuxAI/StartLux-Decision-4B#scaffold-cpu`**、**36.6M** 随机权重替身 | `release/production/sft.py:151`（`SCAFFOLD_SUFFIX`）、`:1348`（`scaffold_teacher`）；run `1005-p202-teacher-adapters-b3-realpath-b570/b3_payload.json:47` `scaffold_params_m: 36.6`；run config 例：`1005-p2-05-dev-gc-sft-qwen3-0-6b-torch-25d7/config.yaml:51` | 字段值 36.6；teacher_model_id 带 `#scaffold-cpu` 后缀 |
| 16 | 真 4B **已下载未载入**：**9,344,023,187 字节**；内存守卫（**4395.1MB / 3009.7MB** 可回收不足） | run `1005-p202-teacher-adapters-b3-realpath-b570/notes.md:5,15` 与 `-2/notes.md:15`（b3_payload 的 real_4b_load 段） | notes 原文两处数字命中 |
| 17 | GLM 首波 **14 次调用 / 2 行落包 / 颜色题 0/10**；次波 **11 次 / 10 行** | 同上两 run 的 notes.md:8 与 :15 | "实际发出 14 次，产标 2 条…颜色题答对 0/10"；"真调用 11 次…产标 10 行" |
| 18 | SFT **7 条 dev run 全为 qwen3-0.6b 替身 / CPU / tiny** | `ls -d release/runs/1005-p2-05-dev*` → 7 目录（sft×5 + gc-sft + rec-sft）；config `stage: p2-05-prod-sft/tiny-cpu`、`device: cpu`、`max_steps: 100`；notes 结论行自述"CPU 替身口径…910B 正式档待 C5" | 7 个目录；`TRAINABLE_NOTE`（`sft.py:166`）原文 |
| 19 | 蒸馏栈 **零训练 run**、p2-06 勾 **3/5** | `ls -d release/runs/*p2-06* release/runs/*opd*` → **no matches**；`changes/p2-06-opd-gkd/tasks.md` 顶层计数 3/5；散度口径 `production/opd.py:521-546`（选项级 forward KL） | 通配无匹配 |
| 20 | **RLCD 尚未落地：p2-11 勾 0/6**，HEAD 无 `rl_rlcd.py`/`train.py`/`tests/test_rl.py` | `ls release/production/rl_rlcd.py release/production/train.py` → No such file；`changes/p2-11-rl-rlcd/tasks.md` 顶层 `- [x]` 计数 = **0/6** | `0/6`；文件缺失报错原文 |
| 21 | 九件 `*_asc.py` **target=ascend 编译 PASS（9/9）**，含 `asc_fill_l1`→`set_l1_2d` | `release/ascend/kernels/reconcile/J_RESULT.md:3`（`ASC-COMPILE 9/9 PASS + 静态口径 3/3 PASS`）；`ls release/ascend/kernels/*_asc.py \| wc -l` = 9；p2-13 `tasks.md:21`（P1-4b 已闭，编排者容器复验 9/9）+ 提交 `9687df7` | `ASC-COMPILE 9/9 PASS`；件数 9 |
| 22 | GDN 反向 **kernelized** | `J_RESULT.md:5,41`；`release/ascend/kernels/gdn_asc.py:43` `BWD_STATUS = "kernelized"`；attempts/F 六梯接入（p2-13 `tasks.md:19`、`attempts/F/RESULT.md`） | `BWD_STATUS = "kernelized"` 命中 |
| 23 | train_step host 绿 **loss 0.911→0.824、15 枚梯度全非 None**；`test_ascend_train_step.py` **8 passed**、gradcheck 基线 **20 passed** | p2-13 `tasks.md:23`（代理 M，编排者 host 复验）+ 提交 `dba5dc0` 信息体 | 原文数字命中（本波未在 Mac 重跑该 host 复验，账面来源为编排者复验记录） |
| 24 | **数值/单位 V1–V6 待卡** | p2-13 `tasks.md:20`（dW 数值属 V1/V2/V3 卡待域）、`:21`（"上卡待证（登记 V4/V5/V6）"） | 两处行号原文 |
| 25 | cube 面真机曾 **aicore exception 507015**；修复在途、**待 ≤8min 卡窗**复验 | run `1010-p2-13-oncard-wave1-cc3d/metrics.jsonl`（`grep -c 507015` = 1）；run `1010-p2-13-p11d-cube-fix-wave-0c72/notes.md` 结论行（"位序修复为高置信…数值真机复验…为唯一欠账，待一次 ≤8min 卡窗"） | grep 命中 1 处；notes 原文 |
| 26 | kernel 开关眼下**只接管 LoRA 旁路**，backbone 前向仍 HF | `release/production/sft.py:33-38`（被否方案四原文） | 行号原文命中 |
| 27 | 温度标定**恶化**：choice ECE **0.1175→0.2267**（Δ+0.1092）、noul 0.1562→0.1682、标定温度顶格 **5.0** | run `1006-eval-report-aa36/report.md:12` + `metrics.jsonl:1`（`ece_before/after_choice`）；上限出处 `scratch/sys1/calibrate.py:35 GRID_HI = 5.0`（另 `sys1/decision/temperature.py:40 TEMP_MAX = 5.0`）；过程 run `1006-s3-calibrate-962f`（0.1709→0.2129，improved=False） | report.md:12 行原文"否" |
| 28 | 零样本 32K 召回 **0.25 = 4 题中 1 题**；128K 档 1 题 | run `1006-p2-07-a2-needle-curve-v3-1035/metrics.jsonl`：bucket 32768 四行 `ok` = true/false/false/false（1/4）；bucket 131072 一行 `ok=false`（0/1）；8192 档 1/2 | 逐行 ok 字段清点 |
| 29 | 1M 三件套落地约 **1/3**（仅滑窗成件），1M 档零实测 | `changes/p2-07-long-context/tasks.md` 顶层 7/8，唯一未勾为 C2"1M 云端配置（measured/config 双列）"；线性记忆适配器无训练 run、门控短路零实现（审计 §3.2 B 级判定） | 计数 7/8 |

## 三、门面账（编排/测试）

| # | README 写入值 | 来源 | 实测输出摘要 |
|---|---|---|---|
| 30 | **81 条孙任务 / 57 勾（2026-10-10）**，13 子域 | 对 `openspec/changes/teacher-p2-production-full/changes/p2-*/tasks.md` 计数**顶层** `- [ ]` / `- [x]` 行（口径同 design D3；含嵌套子行为 83/58，已在下表注明两种口径） | `TOTAL(顶层): 57/81`；子域目录 13 个 |
| 31 | "全勾对账"属**已归档第一幕**：**66/66 + 一级 14/14** | `openspec/changes/archive/2026-10-03-teacher-p1-scratch-mps/changes/*/tasks.md`（10 子域）与同目录 `tasks.md` | `孙: 66/66 子域数=10`、`一级: 14/14` |
| 32 | 未勾重头：p2-11 0/6、p2-12 0/4、p2-09 1/3、p2-05 5/8、p2-06 3/5 | 同 #30 的逐文件计数输出 | 逐目录 `d/t` 明细（本次会话打印） |
| 33 | **326 passed, 3 skipped, 3 deselected**（第二幕验收命令本机实跑，2026-10-10） | `cd release && .venv/bin/python -m pytest tests -q -m "not integration" -p no:cacheprovider` | 原样输出尾行 `326 passed, 3 skipped, 3 deselected, 1 warning in 37.25s` |
| 34 | 该命令**前置条件**：`test_backends.py` 双重 fixture 修复前收集期硬错误 | `git show HEAD:release/tests/test_backends.py` 在 `:26/:28` 出现两次 `@pytest.fixture(autouse=True)`（工作树已由同波代理 O 修好、尚未入库）；审计 §2.1.4（`ValueError: @pytest.fixture is being applied more than once`，RC=2）；跨环境参照：审计容器最佳 **276 passed** | HEAD 版双装饰器在场，工作树版已单装饰 |
| 35 | 训练口 `axis: train` + 守卫（本波自查修复中） | 工作树 `git diff release/production/configs/` 五处 `-axis: quality`→`+axis: train`（HEAD 版曾为 quality）；新增 `release/tests/test_train_axis_guard.py`，实测 `.venv/bin/python -m pytest tests/test_train_axis_guard.py -q` → **7 passed** | `7 passed in 1.42s` |
| 36 | **93 个 run 目录 / 77 个 notes 残留"待填写"** | `grep -rl 待填写 scratch/runs release/runs --include=notes.md \| wc -l` = 77；`ls -d scratch/runs/*/ release/runs/*/ \| wc -l` = 93（审计在 b4a399f 为 76/88，差值=本波新增 run） | `77` / `93` |
| 37 | 三个空跑壳（run 不删） | `release/runs/1006-p2-07-a2-needle-curve-{b46b,e6c8,v3-e4ba}/metrics.jsonl` 均 0 行 | `0 行metrics` ×3 |
| 38 | R 反模式编号**以 skill 为源、本页不写死区间**；AGENTS 旧区间待 P1 对齐 | `.agent/skills/openspec-multilevel-planning/SKILL.md:82`（"反模式（均真实出现并被修复）"），该 skill 内实存 R1–**R20**；`AGENTS.md:37` 仍写 `R1–R14`（归代理 Q 的 R-P1-2 处理，N 不越界改它） | skill 内 R 编号集合 R1..R20 |
| 39 | StartLux 权重许可 **CC BY-NC-4.0**（vs 学生基座 Apache 2.0）；GLM 官方 320B-A18B | 审计 §3.3（A 级，HF/智谱官方页一手核验）；README 保留原结论、仅把"权利推演未成文"与"ToS 未分析"降级为 P2 登记（§2.5.2/§2.5.3） | 判定"成立"，附加口径说明 |

## 四、口径声明与已知不确定性（供编排者四查）

- **HEAD 与工作树混用点**（凡涉并行代理产物均已注明，不是臆测）：#33/#34/#35 的"修复后"状态含**未入库**改动（O 的 fixture 修复、P 的 axis 切换与守卫）。README 相应文字写的是"以运行输出为准 + 修复见 R-P0-3/R-P0-2 + 跨环境数不可互抄"，没有把未入库成果说成 HEAD 既有。集成入库后这些句子仍然成立，无需二次改写。
- **#20 的在途冲突风险（主动申报）**：工作区存在**未跟踪**的 `release/runs/1010-p2-11-dev-rl-qwen3-0-6b-torch-bf64{,-2,-3}/`（含 reward −3.05→−4.41 的 dev RL 记录）。HEAD 上 p2-11 仍 0/6、`rl_rlcd.py` 不存在，故 README 按 HEAD 写"尚未落地"。若同波代理本波入库，README 该句需在集成时同步为"dev RL 有 3 条探索 run（发散/未收口），p2-11 仍非全勾"。**N 未擅自把它写成已落地，也未删它。**
- **计数口径两式**（#30）：顶层 `- [ ]` 行 = 81/57（与 design D3 一致，README 采此）；含嵌套行 = 83/58。审计 §2.1.3 报的是"80–82 条 / 56 勾（b4a399f）"，差值来自 `9687df7`（接口 home 9/9）与 `dba5dc0`（autograd+train_step 接线）两笔提交的新增勾项。
- **#23 未在 Mac 重跑**：train_step host 绿数取自 p2-13 tasks:23 的编排者 host 复验记录（B 级账码一致），README 未把它写成"本机实测"。
- **未改动的旧数字**：`+31.7pp/+25pp` 改写为报告原值 `+31.67pp/+25.00pp` 并附 n 与基线；`P50 5.8ms`、`parity 20/20` 原值保留但补了口径；`"几十元"`、`"320B"`、`"qwen3_5"`、`"Apache 2.0"` 三个外部引用项经审计判"成立"，仅补估算/许可属性说明，未动结论。

## 附录：本会话复跑命令（可整段重放）

```bash
# 参数双口径（#1/#2）
scratch/.venv/bin/python -c "import sys;sys.path.insert(0,'scratch');from sys1.model import ModelConfig,Decoder;\
m=Decoder(ModelConfig());t=sum(p.numel() for p in m.parameters());\
e=sum(p.numel() for n,p in m.named_parameters() if 'tok_emb' in n or 'lm_head' in n);print(t,e)"

# 自训 BPE 字母编号（#4）
scratch/.venv/bin/python -c "import json,string;v=json.load(open('scratch/runs/1003-s0-bpe-16k-realedu-zh-en/tokenizer/vocab.json'));\
u=[v[c] for c in string.ascii_uppercase];print(min(u),max(u),u==list(range(min(u),max(u)+1)))"

# MHA 无分组 kv 头（#3）
grep -rn "GQA\|kv_head\|n_kv" scratch/sys1/model.py   # → 0 命中

# 孙任务两种口径计数（#30/#31/#32）
cd openspec/changes/teacher-p2-production-full && for f in changes/p2-*/tasks.md; do \
  echo "$(dirname $f): $(grep -cE '^- \[x\]' $f)/$(grep -cE '^- \[[ x]\]' $f)"; done
for f in changes/p2-*/tasks.md; do t=$((t+$(grep -cE '^-\s\[[ x]\]' $f))); d=$((d+$(grep -cE '^-\s\[x\]' $f))); done; echo $d/$t

# 第二幕验收实跑（#33/#34/#35）
cd release && .venv/bin/python -m pytest tests -q -m "not integration" -p no:cacheprovider
.venv/bin/python -m pytest tests/test_train_axis_guard.py -q

# 教师实名与真 4B 未载入（#15/#16）
grep -rn "scaffold-cpu" release/production/sft.py release/runs/1005-p2-05-dev-gc-sft-qwen3-0-6b-torch-25d7/config.yaml
grep -n "scaffold_params_m" release/runs/1005-p202-teacher-adapters-b3-realpath-b570/b3_payload.json

# 昇腾三态（#21/#22/#23/#24/#25）
head -5 release/ascend/kernels/reconcile/J_RESULT.md
grep -n "BWD_STATUS" release/ascend/kernels/gdn_asc.py | head -2
grep -c 507015 release/runs/1010-p2-13-oncard-wave1-cc3d/metrics.jsonl
sed -n '1,8p' release/runs/1010-p2-13-p11d-cube-fix-wave-0c72/notes.md

# 负结果上门面（#27/#28）
sed -n '12p' scratch/runs/1006-eval-report-aa36/report.md
grep -o '"bucket": 32768.*"ok": [a-z]*' release/runs/1006-p2-07-a2-needle-curve-v3-1035/metrics.jsonl

# 门面卫生（#36/#37/#38）
grep -rl 待填写 scratch/runs release/runs --include=notes.md | wc -l
grep -rhoE "R[0-9]+" .agent/skills/openspec-multilevel-planning/*.md | sort -u -t R -k2 -n | tr '\n' ' '
```

## 五、门面 grep 自证（写后三连）

```bash
grep -cE "296 绿|72 孙任务全勾|GQA|32\.\.57|完全体|R1–R18|R1-R18" README.md   # → 0
grep -n  "scaffold" README.md                                                  # → :22 命中
grep -nE "38\.\.63|54\.2M|57 勾|kernelized|编译 PASS" README.md                 # → :13/:16/:24/:37 命中
wc -l README.md                                                                # → 59（原 52）
```
