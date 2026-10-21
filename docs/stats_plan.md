# 统计补强方案（audit-remediation-1010 · R-P2-1 入档，2026-10-11）

> **本文性质**：记录级入档。把评审 §2.4.2 指出的四处统计功效不足，补成"可执行的最小设计 + 报告口径改造"文本。
> **本文不排产**：不指定变更号、不指定开工时间、不承诺完成日期（依 `openspec/changes/audit-remediation-1010/design.md:5` D1
> "本变更不代 p2-11/p2-13/C5 排产"）。落点建议只在 §6 登记为待编排者裁决的选项。
> **本文零代码改动**：只描述未来插入点（文件:行），不触碰 `sys1/`、`release/production/`、任何 run 档。
>
> 三态标注（`.agent/rules/common.md:90` 图例，配合 `:115` "结论区分文档承诺/实测确认/推断"）：
> ✅ 实测确认＝本会话跑过命令并贴输出；⚠️ 文档承诺＝只读到文字账目、未复现；❓ 推断＝由公式或常识推算，非实测。
> 与评审转述不一致处一律以本会话实测为准并在 §2 表下注明。

## 0. 结论速览

| # | 被指问题（评审 §2.4.2） | 本会话复核值（✅） | 差多少才可信 | 补强落点 |
|---|---|---|---|---|
| 1 | 全配置单种子 | 5 份 yaml 全 `seed: 20261005`；全仓无区间/重采样/多种子代码（命中 0 行） | ≥3 种子成对 | §3.1 |
| 2 | 双基线 n=24 单次、Δ=+0.0000 | 两模型各 `n_samples: 24`、acc 均 0.583333（14/24）；**Wilson 95% 实算 ±18.3pp**（评审估"约 ±20pp"，同量级） | 分辨 5pp 需 **1501/臂**（独立样本口径） | §3.2 §3.3 |
| 3 | 头条 +31.7pp 出自 n=40、参照随机 1/3 | `choice 3 40 0.6500 0.3333 +31.67pp`；桶表**已印 n 与 base**，无区间；参考线常数在 `quality.py:39`（scratch 与 release **两份副本同行号**） | n=40 区间 ±14.2pp；超随机本身成立（精确二项 **p=4.17e-5**，评审报 p≈2e-4，同数量级） | §3.2 §4 |
| 4 | needle 每档 1–4 题 | 逐题实测 **8192:1/2、32768:1/4、131072:0/1**（7 行 metrics，无 256K 档）；现成信封库仅 **8 题**（2/4/1/1） | 单档 ≥97 题才有 ±10pp 精度；当前 0.25 的区间是 **[0.046, 0.699]** | §3.3 |

**一句话**：四处都不是造假（run 档逐字对得上），是**样本量与呈现口径**问题；补强不需要新数据源——扩样所需的题池**已在本地盘上**（§3.3 实测 2000/12447/400/231 题，status=cached）。

## 1. 前置声明：训测分离已完成，统计补强可以启动

评审 §2.4.1 的雷（训练口 `axis: quality` 打在评测 test 集上）是统计补强的**启动条件**：若训测同集未解，扩样只会把污染分数测得更精。现状：

- ✅ 五份配置训练口全为 train 轴：`grep -h "^axis:" release/production/configs/*.yaml` → 5 行 `axis: train`（gate08b:19 / scaling_06b:20 / tiny_cpu:20 / zh_mix_30:28 / zh_mix_50:28）。
- ✅ 默认值同步：`release/production/sft.py:1048` `"axis": "train"`。
- ✅ 机器守卫在场且绿：`cd release && .venv/bin/python -m pytest tests/test_train_axis_guard.py -q` → **7 passed**（含 R14 注错自证：注入 `axis: quality` 必变红，`tests/test_train_axis_guard.py:119-124`）。
- ✅ 前后台账：run `1010-audit-remediation-p0p1-0aba/metrics.jsonl` 的 before/after 行——before「训口 axis:quality×6」→ after「axis:train×5+守卫 7passed(注错变红自证)」；`notes.md` 结论行记「训测分离落地有守卫」。
- ⚠️ 历史 dev run 仍在 test 集上训练（评审 §2.4.1，7 条 run），**不回写**（§3.4 口径）；它们本来就只出 loss、未出评测分，不构成对外主张。

**因此**：任何"新 run"从现配置起跑即不再训测同集，§3 的设计可直接用于 C5/C6 正式档，不需要额外的"先修数据口"前置。

## 2. 现状底账（逐条自查，附可复跑命令）

```bash
# ① 单种子
grep -rn "^seed:" release/production/configs/*.yaml                # → 5 处，全 20261005
grep -rniE "wilson|bootstrap|binom|scipy|statsmodels" --include="*.py" \
  release/sys1 release/tests release/production scratch/sys1 scratch/tests | wc -l   # → 0
grep -n "seed" release/production/sft.py                            # 种子传导点：1046 默认 / 1211-1212 random+torch / 1236 桌序
# ② 双基线样本量
sed -n '10p;20,21p' release/production/baselines/compare_table.md   # acc 0.5833/0.5833 Δ+0.0000；两行样本数 24
grep -n "n_samples" release/sys1/eval/baselines/{laya,startlux}/smoke24.row.json
# ③ 头条数字与其参照线
sed -n '6,9p' scratch/runs/1006-eval-report-aa36/report.md          # 表头含 n/base/Δ/"超参考线(+10.0pp)"
grep -rn "REFERENCE_MARGIN_PP = " scratch/sys1/eval/quality.py release/sys1/eval/quality.py  # 两份副本均 :39
# ④ needle 逐题账
cat release/runs/1006-p2-07-a2-needle-curve-v3-1035/metrics.jsonl | \
  python3 -c "import sys,json,collections;c=collections.Counter();ok=collections.Counter();\
[ (c.update([r['bucket']]), ok.update([r['bucket']]) if r['ok'] else None) for r in map(json.loads,sys.stdin)];\
print({k:(ok[k],c[k]) for k in sorted(c)})"                          # → {8192:(1,2), 32768:(1,4), 131072:(0,1)}
```

**本会话复核中新增的两条事实（评审未列，须一并进方案）**：

1. ✅ **needle 是"两个并存口径"，不是账目错乱**（本会话逐文件实测，并据此更正本文早期措辞）：
   题面/针位规划集 `release/bench/eval_data/assembled/needle-synthetic/needle-synthetic.jsonl` 实测 **30 行**（按 ctx：8192:2 / 32768:4 / 131072:8 / 262144:16），与 `release/bench/eval_data/manifest.json:389` 的 `"samples": 30` **一致**；
   而**可评测的长文信封** `needle-envelopes.jsonl` 实测 **8 行**（`needle-envelopes.meta.json` 的 `ledger.probes`：8192:2 / 32768:4 / 131072:**1** / 262144:**1**），
   上文 run 只消费前三档 = **7 行 metrics**（8192:1/2、32768:1/4、131072:0/1；262144 档未跑）。
   → 三点后果：① 评审与本文的"每档 1–4 题"用的是 **probes（信封）口径**；若有人引 manifest 的"30 题"会把可用样本**高估近一个数量级**；
   ② 扩样真正的瓶颈在**信封生成**（长文拼装与显存，见 §3.3 的 18.46GB/118K 实测），不在题面数——`needle_plan` 只决定题面；
   ③ 记账字段易混：`manifest.json:373` `envelope_ready: false`、`:377` `gold_coverage.with_gold: 0` 与信封 meta 的 `envelope_ready: true` 并存，
   建议后续 run notes 显式标注"口径=probes/题面"（本文不改 manifest，遵守 §3.4 冻结口径）。规划式出处：`release/sys1/eval/registry.py:1242-1252`（`needles = 2 ** (i + 1)`）、`release/sys1/eval/longctx.py:393-429`（`limit_per_bucket`）。
2. ✅ **P1 幕的 100 题池与 P2 幕的 2000 题池不是一回事**：aa36 报告头 `n=100` 来自 P1 pin（`scratch/bench/p1-09/typed_decisions/test.jsonl` 实测 100 行），而 release registry 的 typed-decisions test 档实测 **2000 行**（`release/bench/eval_data/assembled/typed-decisions/typed-decisions.jsonl`）。→ 扩样在第一幕口径下要重下数据，在第二幕口径下**零下载**。

**两处与评审转述的差值（如实记，不沿用）**：Wilson 半宽本会话实算 **±18.3pp**（评审"约 ±20pp"）；26/40 对 1/3 的精确单侧二项 **p=4.17e-5**（评审"p≈2e-4"）。结论方向一致。

## 3. 最小可执行设计

### 3.1 种子集合

- **集合**：`S = {20261005, 20261006, 20261007}`（3 颗）。首颗保持现值，**不重排既有可比性**：
  `gate08b.yaml:15` 与 `scaling_06b.yaml:16` 的行尾注释写明"同种子 → 同桌序（两档只差批规模时才可逐点对比）"——
  所以多种子必须**跨档成对**跑（同一颗种子下的 gate08b 与 scaling_06b 才允许逐点比），不能各档各取随机种子。
- **传导面**（✅ 已核，改动极小）：种子只经 `sft.py:1046`（默认）、`:1211-1212`（`random.seed`/`torch.manual_seed`）、`:1236`（`bucket_batches` 桌序）生效，
  故"换种子＝改一个 cfg 键"，**无需新机制**；外层三次调用或一个 `seeds:` 列表循环即可。
- **报数口径**：3 种子 → 每桶报 mean ± 极差（n=3 不给标准差，评审式的"单值头条"作废）；若做 ≥5 种子再报 sd。
  ❓ 推断：3 种子只能把"种子方差是否显著"变成可判断，不能替代题量扩充——两者是正交的两条补强，必须都做。

### 3.2 n 下限与功效估算式（纯标准库，零新依赖）

`release/pyproject.toml:7-30` 无 scipy/statsmodels（✅ 实测），故以下三式全部可用 `statistics.NormalDist + math` 实现，不引新依赖：

```
Wilson 95% 区间（份额类，x 命中 / n 题，z=1.96）：
  center=(p + z²/2n)/(1 + z²/n)  halfwidth = z·√(p(1−p)/n + z²/4n²) / (1 + z²/n)

单样本 vs 固定参照线（H0:p=p0）所需题量：
  n = ( z_{0.975}·√(p0q0) + z_{0.80}·√(p1q1) )² / (p1−p0)²
两模型同题量对照（H0:p1=p2，各臂 n）：
  n = 2·(z_{0.975}+z_{0.80})²·p̄(1−p̄) / (p1−p2)²      （功效 80%、双侧 α=0.05）
```

本会话用该式实算的门槛表（可整段重放，脚本见 §7）：

| 目的 | 算式 | 需要的 n | 现状 |
|---|---|---|---|
| 单值精度 ±10pp（p≈.5） | `(z/E)²·pq` | **97** | 双基线 24／choice 桶 40 |
| 单值精度 ±5pp | 同上 | **385** | 全轴 100（P1 pin） |
| 与随机参照线分辨 +5pp | 单样本式，p0=1/3 | **711** | 40 |
| 两模型分辨 5pp（独立臂） | 两样本式，p̄≈0.605 | **1501/臂** | 24/臂 |
| 现 n=24 能分辨的最小差 | 反解 | **≈40pp** | 观测 Δ=0.0000 → 此表在此 n 下不携带信息 |
| 现 n=40 的头条区间 | Wilson | [0.4951, 0.7787]，±14.2pp | 报告只印 0.6500 |
| 现 n=4 的 needle 档 | Wilson | [0.0456, 0.6994]，±32.7pp | 报告印"0.25" |

- **建议下限（写入报告口径，不作 gate）**：桶级 `n ≥ 97` 才允许出"百分点差"表述；`n < 40` 的桶只允许出"命中 x/n"的原始计数，禁止换算成 pp 结论。❓ 推断（阈值选择是工程判断，非统计强制）。
- **配对设计留作更优选项**：同一套题跑两模型属配对样本，McNemar 精确检验在 5pp 差上通常远省于 1501/臂；本会话未算其功效（需先估计不一致对比例），只登记为"扩样前先评估配对口径"的设计选项。❓

### 3.3 扩样清单与成本量级（题池已在盘，成本以实测吞吐折算）

✅ 本地已在盘且 `status=cached`：`release/bench/eval_data/manifest.json` 记数与 `assembled/*/*.jsonl` 实数逐集核对一致（唯一分叉是 needle，见 §2 注 1）。

| 集（轴） | 在盘题量 | 可扩到哪 | 成本量级（按实测吞吐折算；❓＝折算，非实测整跑） |
|---|---|---|---|
| typed-decisions（quality，test 档） | **2000** | ±5pp 级：全 2000 题用满 | 学生侧 MPS readout P50 5.82ms（aa36 :17）→ 2000 题 ≈ 12 s 量级 |
| typed-decisions-train（train 轴） | **6000** | 训练侧不受"考卷"约束，可整档用 | — |
| intern-decision（quality） | **12447** | 与 typed 合并可破 1501/臂门槛 | laya CPU 3.060 题/秒 → 2000 题 ≈ **10.9 min**；startlux-0.8B CPU 0.1836 题/秒 → 2000 题 ≈ **3.0 h**（同 run 实测值，`smoke24.row.json`） |
| jevbench（calibration） | **231** | ±13pp 级 | — |
| cmmlu-decision / clue-decision（quality，中文） | **200 / 400** | 单集即达 ±10pp；两集并 600 → ±5.7pp | — |
| longbench-zh（longctx） | **200** | 长文题池替代 needle 扩题 | 峰值内存随档涨：8K→7.6GB、118K→**18.46GB**（needle v3 metrics 逐行 peak_gb）；MPS 在 ~118K 崩于 conv1d（同 run notes:10 更正行）→ 长档归云端 C 波 |
| needle-synthetic | 题面 **30** / 可评测信封 **8**（probes 2/4/1/1；见 §2 注 1） | 每档 ≥97 题需改 `needle_plan`（`registry.py:1249` 的 `2 ** (i + 1)`）或加 `probes` | 每根针=一题；生成侧成本是长文拼装与推理显存，不是下载 |

- ✅ **扩样不需新代码**：`release/sys1/eval/baselines/run_baseline.py:481` 的 `--limit` 默认 24、注释即写"0=全量，全量属 C6"；`SMOKE_LIMIT = 24`（:67）。把 split 从 `smoke24` 换成新全量切分名 + `--limit 0` 即得 n=2000。
- ⚠️ 全量档在项目账上**本来就是挂账项**，不是本文新增：`compare_table.md:23-31` 第三节整表写"待 full"，注 3 明示"全量 `full` 延后至云端 C6（NPU）"。→ 统计补强与既有 C6 计划同向，只补"出分必带区间"这一条口径。
- ❓ 云端成本参照：`openspec/changes/teacher-p2-production-full/design.md:174-184` D12（910B ¥20/h、C5=探针报价、熔断线 ¥600）。多种子 × 全量 = 段数翻倍，**须先按 D12 报价再定种子数**，本文不替 C5 报价。

### 3.4 冻结—重跑口径（历史不回写）

1. **历史 run 与历史产物一律不改写**，勘误只追加：依据 `design.md:28`（"保留旧值注释与 run 档原样（历史 run 不改写）；新 run 起用 train 轴"）、`.agent/rules/common.md:101`（"历史审计记录只附勘误不改写"），以及仓内既有先例——`release/runs/1006-p2-07-a2-needle-curve-v3-1035/notes.md:10` 的「更正（编排者补录）」就是追加式修正。
2. **新数字起新档**：多种子/扩样结果**新起 run-id 与新 split 名**（建议 `pwr100`（±10pp 档）/ `pwr5`（±5pp 档）/ `full`），旧 `smoke24` 与其 row.json、compare_table 留档不覆盖（`compare_table.md` 由 `baselines/table.py` 生成，改生成器只影响新档输出）。
3. **两套口径并存时以 run-id 定身份**：同一模型同一集在旧 `smoke24` 与新 `pwr*` 下并存，报告引用必须写 run-id，禁止跨档取平均或"取更好看的那个"（评审 §2.4.4 的选择性采摘教训）。
4. **§2 注 1 的 needle 账目分叉**：新 run 起用重算后的 `samples` 口径；旧 manifest 不动，差异以本文件为勘误登记处。

## 4. 报告口径改造：头条数字必须"带 n、带区间、带参照线身份"

现状态（✅）：**表里已有 n 与 base，缺区间；参照线身份在文字里已澄清，但表头没有身份列**。改造点收敛为三处（本文只登记，不动码）：

| 文件:行（未来插入点） | 现状 | 改法 |
|---|---|---|
| `release/sys1/eval/quality.py:89-99`（桶行 dict） | 有 `n/accuracy/hits/baseline/delta_pp/meets_reference`，无区间 | 增 `ci95_wilson=[lo,hi]`（§3.2 式，纯标准库）；`hits` 已在，无需改分母 |
| 同文件 `:127 _HEADER` / `:131 format_table` | 表头 `… n acc base Δ … 超参考线(+10.0pp)` | 加 `95%CI` 列；"base" 列改名并**标身份**：`base(随机)` / `base(对手亲跑)` / `参考线=自定常数` |
| `release/sys1/eval/baselines/table.py:36-42 METRIC_ROWS` | 只有 acc/ece/ms/qps 四把尺 | 增 `acc_ci` 行；`Δ` 列在无区间时印"不可判"而非 `+0.0000`（现 :10 印 `+0.0000` 会被读成"两模型等价"） |
| `release/sys1/eval/report.py:99-110`（verdict） | `passed` 明确不含质量阈值（:109 note 自述），评审 §2.4.3 已判"这是写明的设计选择" | **不动退出码语义**；只加呈现字段 `low_power_axes=[…]`（n<97 的桶点名），报告层打折、门禁层不变 |
| `README.md:14`（头条行，禁改区，由编排者定） | 已带 `n=40`、`全轴 n=100`、"参考线是项目自写常数…不是任何竞品基线"（R-P0-1 已补） | 唯一缺口是**区间**：`+31.67pp` 后补 `（95%CI ±14.2pp）`；同法处理 `20/20`、`0.5833` |

**参照线身份三分类**（报告/README 表头必须落到其中一类，不得只写"参考线"）：
① **随机基线**（1/k；`quality.py:87` 的 `base_val` 来源含 `sys1.calibrate.bucket_baselines`）；
② **对手亲跑**（同 harness 同 split，`compare_table.md:20-21` 两行 run-id 为凭）；
③ **自定常数**（`REFERENCE_MARGIN_PP = 10.0`，`quality.py:39`，scratch 与 release 两份副本同行号）——现头条属 ①+③，**不含任何 ②**。

## 5. 补强前只能作"方向性参考"的既有主张（逐条点名）

以下主张在 §3 落地前，引用时必须附"小样本、单种子、无区间"限定（评审 §6.2 对评估者的建议同向：不要引 README 数字，引 run 档）：

| 主张 | 出处（✅ 已核） | 缺什么 | 补强前允许的最强表述 |
|---|---|---|---|
| choice **+31.67pp** | `scratch/runs/1006-eval-report-aa36/report.md:7` | 无区间；n=40；参照=随机 1/3 | "26/40 命中，超随机 1/3，单侧 p≈4e-5；±14.2pp 内不作幅度结论" |
| noul **+25.00pp** | 同 :8（n=60，0.7500 vs 0.5000） | 无区间 | 同上（±10.7pp） |
| 双基线 **acc 同为 0.5833 / Δ=+0.0000** | `release/production/baselines/compare_table.md:10`、两侧 `smoke24.row.json` | n=24；无区间；单种子 | "24 题下不可判两模型高低"（不得写"打平"） |
| needle **32K 召回 0.25**（8192→0.5/32768→0.25/131072→0.0） | `release/runs/1006-p2-07-a2-needle-curve-v3-1035/notes.md:9` + `metrics.jsonl`（7 行） | 每档 1–4 题 | "1/4 命中"，禁止写成衰减曲线 |
| **ECE 0.1175→0.2267 恶化** | aa36 `report.md:12`（n=40，温度顶 clamp 5.0，"改善：否"） | 无区间；n 小 | "标定在本档未见改善（负结果，样本 n=40）" |
| **parity 20/20**、**P50 5.8ms** | aa36 `report.md:20`、`:17`（speed serial n=30） | 与统计功效无关但口径需标（手工装配前向 / MPS eager） | 保持 README:14 现措辞（已带口径） |

## 6. 排产边界与待裁决项（交编排者，不代排）

- 本文不新建变更、不勾他人任务。可选落点（各条前置不同，任选其一即可推进 §3.4 的"新档"）：
  ① C6 全量档（`compare_table.md:23` 已挂账"待 full"）顺带做 §4 区间口径；
  ② C5 正式训练前按 D12 报价决定种子数（§3.1 的 3 颗是否付得起）；
  ③ needle 扩题需先解 §2 注 1 的"30 vs 8"账目分叉 + 长档显存（18.46GB/118K 实测）→ 只能云端；
  ④ 纯零成本先行项：**只做 §4 的呈现层改造**（加区间列与参照线身份列，不动任何数据），不依赖算力。
- ❓ 待裁决：区间列是否升级为 gate（本文建议**不**升级为 gate——评审 §2.4.3 已认定"质量不参与退出码"是写明的设计选择，改 gate 属另一决策）。

## 7. 本会话复放凭据

Wilson 与样本量全部数字由以下命令产生（无外部依赖）：

```bash
cd $REPO_ROOT
release/.venv/bin/python -c "
from statistics import NormalDist as N;import math
z=N().inv_cdf(0.975);zb=N().inv_cdf(0.80)
def wil(x,n):
    p=x/n;A=z*z/(2*n);B=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));d=1+z*z/n;return((p+A-B)/d,(p+A+B)/d)
for x,n in [(14,24),(26,40),(45,60),(1,2),(1,4),(0,1)]:
    lo,hi=wil(x,n);print(x,n,round(x/n,4),[round(lo,4),round(hi,4)],'half=+-%.1fpp'%(100*(hi-lo)/2))
n1=lambda p0,p1:math.ceil(((z*math.sqrt(p0*(1-p0))+zb*math.sqrt(p1*(1-p1)))/(p1-p0))**2)
n2=lambda a,b:math.ceil(2*(z+zb)**2*((a+b)/2)*(1-(a+b)/2)/(a-b)**2)
print('vs 1/3 +5pp ->',n1(1/3,1/3+.05));print('两臂 5pp ->',n2(.58,.63));print('E=10pp ->',math.ceil((z/.10)**2*.25),'E=5pp ->',math.ceil((z/.05)**2*.25))
print('26/40 vs 1/3 单侧二项 p=',sum(math.comb(40,k)*(1/3)**k*(2/3)**(40-k) for k in range(26,41)))"
```

**未核验/未能核验（如实）**：① 未实跑任何扩样评测（本文只算功效，无新 run）；② 云端 910B 与 NPU 吞吐、C6 全量成本为 ⚠️ 文档承诺（D12 报价未生成，C 波未跑）；③ 4B 真教师的每题耗时**无实测**（本机载入被内存守卫拦下，见 run `1005-p202-teacher-adapters-b3-realpath-b570` notes），故 §3.3 的整跑折算只给了已实测的 0.8B/CPU 与 0.8B 学生侧 MPS 两条；④ McNemar 配对功效未计算（只登记为设计选项）。
