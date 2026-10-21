# run: 1006-p2-03-d3-eval-registry-six-axes-4f3a

- 假设：评测集按版本钉住、可复跑装配并统一登记后，六轴表能在不出模型分的前提下自证数据就绪度（缺数据的轴显式 n/a 而非省略）
- 观察：
- 实拉 13 集 / 登记 26,308 条 / 带人工答案 26,278 条
- 真实下载累计 248,991,254 字节（幂等复跑状态记 cached、字节不被清零或翻倍）
- 六轴表：
- 一致轴按 B3 裁定显式 n/a；npu 后端当场拒绝并指向 p2-13 接入点（不静默回退 CPU）
- train 底账（D1，训练侧消费面，不入六轴评测表）：3 集 / 10,000 条 / 真实下载 6,442,951 字节（通道 https://hf-mirror.com、-、local）；qtype 逐集 [('typed-decisions-train', {'_other': 0, 'choice': 1800, 'noul': 1800, 'score': 2400}), ('clue-train-subset', {'_other': 0, 'choice': 2000, 'noul': 0, 'score': 0}), ('clue-train-decision', {'_other': 0, 'choice': 992, 'noul': 1008, 'score': 0})]；Intern-Decision train 探查结论：上游无 train 分区：pin 2f81580 归档内只有 test 档数据，不登记 Intern train 项（只交 typed train）（取证见 registry.INTERN_TRAIN_PROBE）——查无即如实记，不硬造分区凑数
- 中文 train 语料底账（D3，C5 前置闸之二）：[('clue-train-subset', 'cached', 2000), ('clue-train-decision', 'cached', 2000)] / 真实下载 5,840,335 字节；中文档 id 与考卷 id 逐条互斥（档位段 train vs validation/test 写在 id 里，断言见 tests/test_registry.py -k split_isolation）；语料缺口上报：CMMLU 上游无 train 分割：modelscope/cmmlu 仓只有 README.md/cmmlu.py/cmmlu_v1_0_1.zip 三件，归档内部只有 dev/(67 个 csv) 与 test/(67 个 csv)，`train` 路径 0 个——故不登记 CMMLU train 项；中文 train 语料只交 CLUE(tnews/ocnli) train 档（已登记 clue-train-subset / clue-train-decision）（取证见 registry.ZH_TRAIN_PROBE）——查无即如实记，不拿 dev 档改名凑 train

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。

## 数据底账（真实下载量，非估算）

```
id                   source      revision     split seed       axis          samples       bytes status
-------------------------------------------------------------------------------------------------------
typed-decisions      hf          f7a2487e     test  -          quality          2000     824,881 cached
typed-decisions-train hf          f7a2487e     train -          train            6000     602,616 cached
intern-decision      archive     2f81580      test  -          quality         12447  24,426,697 cached
jevbench             archive     7ce310c7     test  -          calibration       231   4,640,431 cached
cmmlu-subset         modelscope  master       test  20261005   quality           200   1,078,656 cached
clue-subset          modelscope  master       validation 20261005   quality           400     944,526 cached
cmmlu-decision       derived     zh_decision_v1 test  20261005   quality           200           0 cached
clue-decision        derived     zh_decision_v1 validation 20261005   quality           400           0 cached
clue-train-subset    modelscope  master       train 20261005   train            2000   5,840,335 cached
clue-train-decision  derived     zh_decision_v1 train 20261005   train            2000           0 cached
mmbench-cn-subset    modelscope  master       dev   20261005   multimodal        200  96,700,059 cached
longbench-zh         modelscope  master       test  20261005   longctx           200 113,932,529 cached
needle-synthetic     synthetic   20261005     n/a   20261005   longctx            30         524 fetched
```

- 数据版本串：`eval-registry[typed-decisions@f7a2487e,intern-decision@2f81580,jevbench@7ce310c7] sets=13 samples=26308`
- 累计下载字节：248,991,254（出自 fetch 的真实网络计数；命中缓存时报的是当初真下载的那份）
- 登记条数合计：26,308（其中带人工答案 26,278）
- train 底账（训练侧消费面，不入六轴评测表）：3 集 / 10,000 条 / 真实下载 6,442,951 字节（通道 https://hf-mirror.com、-、local）；Intern-Decision train 探查：上游无 train 分区：pin 2f81580 归档内只有 test 档数据，不登记 Intern train 项（只交 typed train）
- 中文 train 语料（D3 · C5 前置闸之二）：2 集 / 4,000 条 / 真实下载 5,840,335 字节（通道 -、local），逐集 [('clue-train-subset', 'cached', 2000), ('clue-train-decision', 'cached', 2000)]——条数按集累计，训练口只吃决策信封（原件档 envelope_ready=False，不入分母）；CMMLU train 探查：CMMLU 上游无 train 分割：modelscope/cmmlu 仓只有 README.md/cmmlu.py/cmmlu_v1_0_1.zip 三件，归档内部只有 dev/(67 个 csv) 与 test/(67 个 csv)，`train` 路径 0 个——故不登记 CMMLU train 项；中文 train 语料只交 CLUE(tnews/ocnli) train 档（已登记 clue-train-subset / clue-train-decision）
- 执行后端：`None`；npu 为云端占位，接入前必被拒绝（见 p2-13）

## 六轴表

```
axis         status      sets(present/registered)     samples  w/gold reason
----------------------------------------------------------------------------------------------------------------------
quality      未点名         -                                  -       - 这次 --axes 没包含它
calibration  未点名         -                                  -       - 这次 --axes 没包含它
longctx      未点名         -                                  -       - 这次 --axes 没包含它
multimodal   未点名         -                                  -       - 这次 --axes 没包含它
speed        未点名         -                                  -       - 这次 --axes 没包含它
parity       未点名         -                                  -       - 这次 --axes 没包含它
----------------------------------------------------------------------------------------------------------------------
backend=None  （npu 尚未接入：p2-13 sys1.kernels.backends）
parity 裁定：未裁定
```

## 一致轴裁定（B3 · P1 悬空轴教训回写）

- 裁定：P2 一致轴只测 P1 自训 decoder 的延续路径；HF 主干显式 n/a
- 测什么：同一份输入下，自研算子路与纯 torch 路的名次是否换人（沿用一阶段 parity 口径）
- 不测：HF 主干（经 transformers 载入的那条前向）——它只有一条实现路径，自己跟自己比恒等于全对
- 不测：昇腾卡上的算子对拍——归 p2-13 的梯度对拍验收（G9），不在本域
- 不测：远端服务与本地权重的一致性——归 p2-10 serving

- 报告脚注（parity 为 n/a 时原样带上）：parity=n/a：backbone 前向是单一实现路径（本机 CPU/MPS；云端换卡由 p2-13 承接），无双路可对拍；自研件的 parity 验收见 p2-13 G9

## 读表须知

- `n/a` 不等于没跑：要么是没得测（单一实现路径），要么是数据还没拉，reason 列写清是哪种。
- `data-ready` 只保证题面与真值在盘上、来源可回溯；长文与多模态的判分件分属 p2-07 / p2-08。
- 本域不出模型分：给了 `--model` 或 `--endpoint` 才走一阶段预测出口，口径复用不改写。
结论：六轴调度入口就绪；显式 n/a 的轴：无；一致轴按 B3 裁定降级。
