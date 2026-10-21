## ADDED Requirements

### Requirement: benchmark pin 注册
`sys1/eval/registry.py` SHALL 注册全部评测集并 pin：Intern-Decision@`2f81580`、jevbench@`7ce310c7`、typed-decisions@`f7a2487e`、CMMLU/CLUE 子集、MMBench-CN 子集、LongBench-zh、合成 needle（seed 固定）；提供 `export_versions()` 自动导出 revision 清单。

#### Scenario: 版本导出
- **WHEN** 调用 `export_versions()`
- **THEN** 输出含每个集的 id/revision/split/seed，与注册表一致

#### Scenario: 拉取脚本一键
- **WHEN** 运行 `sys1/eval/registry.py fetch --all`
- **THEN** 三件套与各子集落 `bench/`（幂等），失败集列名退出非零

#### Scenario: 幂等复跑不改底账
- **WHEN** 同一批集第二次执行 `fetch`（未加 `--force`）
- **THEN** 状态记 `cached`、不再走网络，且账本里的累计下载字节量不被清零或翻倍

#### Scenario: 无真值的题面不得充当考卷
- **WHEN** 某个公开集的指定分割不带人工答案（实测 CLUE test 档 `label` 整列为 `-1`）
- **THEN** 该集改钉在带答案的分割（validation），或丢弃无真值的行
- **AND** 账本以 `gold_coverage.with_gold / samples` 两个数并列记账，供下游核对分母

### Requirement: 训练数据消费面与 train/test 隔离（D 组 · p2-05 发现回写）
`sys1/eval/registry.py` SHALL 把 train 分割登记为独立集项（typed-decisions@`f7a2487e` train 档，axis=`train`，不挂任何一条评测轴），经既有 fetch 通道真拉并按集记账；`sys1/eval/run.py` SHALL 提供 `load_train_records()` 训练消费口，返回的每条记录 MUST 显式携带 `split=="train"`。上游（Intern-Decision pinned-sha 归档）若查无 train 分区，MUST 把探查结论与取证原样记入底账（`INTERN_TRAIN_PROBE`），禁止硬造分区凑数。

#### Scenario: train 分割真拉入底账
- **WHEN** `fetch --train`（或 `--all`）按 pin 拉取 typed-decisions train 档并装配
- **THEN** manifest 逐集记真实下载字节量、文件白名单、resolved 全长散列与逐题型条数（choice/noul/score 三口与评测信封同形状）
- **AND** run notes 追加 train 底账行：字节量/条数/通道，且幂等复跑记 `cached` 不翻倍

#### Scenario: 数据类记录显式 split 语义
- **WHEN** 任何数据类消费口（`load_train_records` / `load_axis_records`）返回记录
- **THEN** 每条记录带 `split` 字段且与注册表登记值一致（装配时由登记项盖戳，单一真源）
- **AND** `load_train_records` 收到缺 `split=="train"` 的记录当场报错，不静默并入训练分母

#### Scenario: train/test 隔离断言
- **WHEN** 运行 `pytest tests/test_registry.py -k split_isolation`
- **THEN** 任何 train 记录 id 不得出现在 quality 评测轴集内；`train` 轴的集项不得挂在六轴表任何一条轴名下（训练装配复用评测题面=训测同集假分，p2-05 教训）
- **AND** 结构层断言无盘也必须绿：train 登记项存在、轴归属正确、quality 轴零 train 档

### Requirement: Uniform 自检锚点
harness SHALL 以 Uniform 基线复现 typed-decisions 卡面 KL/TV/Brier = 0.444/0.381/0.238 作为正确性自检（容差 0.01）；复现不出即 harness 接错。

#### Scenario: 卡面锚点复现
- **WHEN** 以均匀分布预测跑 typed-decisions 打分
- **THEN** KL/TV/Brier 与卡面差 ≤0.01

#### Scenario: 熔断器自身可信
- **WHEN** 题面为空或人为把真值改成 one-hot 后跑锚点自检
- **THEN** 自检结果为"不通过"并说明原因，禁止返回 ok（防止分母为零伪装成全对）

### Requirement: 六轴评测入口
评测 SHALL 提供六轴：quality / calibration / longctx / multimodal / speed / parity；一阶段四轴接口原样保留，长文与多模态轴为新增。

#### Scenario: 六轴调度
- **WHEN** `python -m sys1.eval.run --axes all --model <dir>`
- **THEN** 输出六轴摘要表，缺数据的轴显式标 "n/a" 而非省略

#### Scenario: 轴与评测集的对应关系只有一个真源
- **WHEN** 需要知道某条轴挂着哪些评测集
- **THEN** 由注册表每项的 `axis` 字段推出，入口内不另存一份轴→集清单

#### Scenario: 一致轴的对象裁定（B3 · P1 悬空轴教训回写）
- **WHEN** 六轴表排到 parity 轴，而被测主体的前向只有一条实现路径（transformers 载入的 backbone，本机 CPU/MPS）
- **THEN** 该轴显式标 `n/a`，reason 取自 `run.PARITY_ADJUDICATION["na_note"]`，报告模板同句脚注
- **AND** 裁定内容：P2 的一致轴只测 P1 自训 decoder 的延续路径（自研算子路 vs 纯 torch 路的名次对拍，沿用一阶段 parity 口径）
- **AND** 明确不测且已转交他域：昇腾卡上的算子对拍归 p2-13（G9 梯度对拍验收）、远端服务与本地权重一致性归 p2-10 serving
- **AND** 禁止把"没有第二条路径可对拍"计入"对拍通过"——单一实现自比恒等于全对是一枚假绿

#### Scenario: 打分入口的后端参数（D6 v3 · NPU 预留）
- **WHEN** 调用方传 `--backend cpu|mps|npu`
- **THEN** cpu/mps 原样通过并回显；npu 在接入前 MUST 抛 `NotImplementedError` 并指明接入域（p2-13 `sys1.kernels.backends`），禁止静默回退成 cpu
- **AND** 云端换卡时调用方只改这一个参数，评测口径不动

### Requirement: 评测数据底账入 run 档案
评测入口 SHALL 支持把一次调度的数据底账记成一页 run 档案（前缀 `p2-03`）：pin 表、真实下载字节量、登记条数与带真值条数、六轴表、一致轴裁定一并入档。

#### Scenario: 六轴表入档
- **WHEN** `python -m sys1.eval.run --axes all --record`
- **THEN** 生成 `<prefix>-eval-registry-six-axes` 档案，config 里的 `data_revision` 不是 UNPINNED 而是"三件套@版本 + 实拉集数 + 总条数"的实测串
- **AND** notes 由报告模板渲染，其中字节量出自 `fetch` 的真实网络计数（命中缓存时报的是当初真下载的那份字节），非估算
