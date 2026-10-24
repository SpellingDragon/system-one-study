# 第二批：P1 的闭环成立到哪里

## 结论

P1 已经形成数据、分词、模型、训练、决策读出与评测的真实工程链。可保留为教学基线。但**归档通过、后续正式档质量、生产路径内核化是三种不同结论**，不能合并成“完整加速系统已验收”。本轮不修改冻结代码或归档勾选。

来源编号见 [证据索引](evidence.md)；十域覆盖见 [第一批](01-global-plan.md)。

## 1. 原始承诺与当前入口

```text
原始数据 → sys1/data/schema、transcribe → 训练／保留数据
文本 → learning/s0_tokenizer → tokenizer
编号流 → learning/s1_pretrain_gpt → Decoder checkpoint
决策样本＋checkpoint → learning/s2_decision_sft → 新 checkpoint
独立校准样本 → learning/s3_calibrate → temperature
评测样本 → sys1/eval/predict → decision readout → report
```

`Decoder.forward` 交出 `[B,T,d]` hidden；`lm_head` 独立，决策程序只取选项字母对应的行，不进行生成循环。P1 自训词表与 P2 现成词表的字母 ID 不相同，契约在“单 token 与映射”而非具体整数。[P101、G01]

本轮比较 HEAD 中两份 `decision/` Git tree，均为 `d199c4c88f54bc7069c572b7a440970c7ee099d6`，说明**已跟踪内容相同**。初次目录直接比较只发现 `__pycache__` 二进制不同，不能据此声称源代码契约改变。工作区无相关源文件改动。[P106、V01]

## 2. P1-F1：独立 parity 不等于生产 forward 已内核化

- **场景**：报告头 `backend=tilelang`，parity 20/20；容易据此认为质量、耗时也都走同一套自研核。
- **机制**：`scratch/sys1/model.py:181-187` 即使无 padding 也构造 causal visible 张量；`:211` 在 `visible is not None` 时拒绝 attention 内核路径；`:386-389` 公开 forward 始终传入该张量。融合 add_ln 的分支也依赖同一门。
- **消费者**：MLP 投影有独立门控，不能将该问题扩大为所有内核都不可达。独立 `eval/parity.py` 的 `KernelForward` 另行手工装配，不受相同公开入口路径支配。
- **更直接的差别**：`eval/predict.py:400` 以默认 `kernel_backend='off'` 加载 Decoder；`:478` 的 backend 描述来自环境探测，并不记录实际派发轨迹。
- **影响**：parity 能支持独立链等价，不能证明正式预测使用了这些核。后续性能研究需要生产调用计数和 fallback 原因；本轮不为让测试变绿而删 mask。[P101]

## 3. P1-F2：流程 gate 不检查质量和校准目标

`eval/report.py:99-109` 明确把溯源、非空轴、parity 名次作为 verdict；质量与把握是呈现值，不参与退出码。因此“exit 0”与“ECE 改善”不是同一判断。

历史正式档 `1006-eval-report-aa36`：

| 项 | 原始读数 | 能说明什么 |
|---|---|---|
| choice | 26/40=0.6500；三选一随机基线 1/3 | 该小样本桶高于随机，不是竞品大规模胜出 |
| noul | 45/60=0.7500；基线 0.5 | 同上，任务桶不可不加说明地合并 |
| choice ECE | 0.1175 → 0.2267 | 该评测集上校准后恶化 |
| noul ECE | 0.1562 → 0.1682 | 同样恶化 |
| parity | argmax 20/20 | 独立对拍样本中名次一致，不证明概率逐元素相等 |
| speed | MPS float32，warmup=3、n=30，P50=5.82ms | 当时入口的端到端计时，不是通用设备指标 |

这些数字同一报告中并存，最终 `passed=True` 并不矛盾；问题在于报告 gate 与高层目标不同。建议下一轮把“流程通过”和“能力判定”分栏，不追认 ECE 已改善。[P103]

## 4. P1-F3：校准独立性存在历史疑点与当前 fallback 缺口

历史 `1004-s3-calibrate-a3a2/config.yaml` 指向 `bench/p1-09/typed_decisions/test.jsonl`，同文件却写“NOT typed-decisions test”。这两条不能同时采信。它不足以证明独立校准收益；若要恢复当年的泛化结论，需要冻结数据 ID／内容清单补证，不能只相信说明文字。[P102]

当前复现脚本还有可直接确认的路径：

- `repro_p1.sh:66-69` 在评测文件缺失或 `EVAL_DATA=auto` 时使用 `eval_holdout.jsonl`。
- `:125-131` 向 `calibrate.jsonl` 和 `eval_holdout.jsonl` 写入**同一份 held 数据**。
- `:145-146` 用前者拟合温度，`:153-155` 用后者评测。

因此该 fallback 可以作为流程冒烟，**不是独立校准测试集**。日志披露来源有价值，但不改变同集拟合这个事实。后续应单列 train/calibration/test；没有新数据时保留冒烟身份，不声称泛化改善。[P105]

## 5. P1-F4：裸克隆复现条件尚未闭合

`bench/` 与模型权重按 `.gitignore` 不入库。`repro_p1.sh:62-64` 检查这些前置，缺件会 exit 2；并且脚本在体检前已创建日志目录，所以本轮没有为验证“拒跑”直接执行它。

提示给出的 `python -m sys1.data.transcribe` 不是完整重建 recipe：当前模块提供转写函数，没有相应 CLI 主流程。保留大权重和数据在仓库外是合理边界，但需要版本固定、输入来源、转写调用与校验值，否则原作者工作区的成功不能转化为学生复现成功。[P105]

## 6. P1-F5：负性能结果是成果，不是加速证据

历史 M3 Pro／TileLang 0.1.15 Metal：单核比较记为比 torch-MPS 慢 2–6×；训练 `28ba` 记录 3,048 vs 3,315 tok/s，推理 `ce64` 记录 11,497 vs 13,017 tok/s。日志提供了可能归因，但没有逐项消融时，不把每个候选原因都当作测定占比。[P104]

取舍：保留可微实现和参考价值；若性能成为交付条件，再建立同入口、同权重、同 shape 的基准。不能从 Mac 负收益断言 NPU 或 CUDA 一定慢，也不能反向宣称换设备一定快。

## 7. P1-F6：小档、大档与额外能力分开记账

早期 SFT 的 checkpoint 是 tiny 模型；后续 `1005-s1-mps-main-dc8c` 才记录 d=512/L=12 的长跑。原设计“~40M”与 README 后来区分的总参数／非嵌入参数是不同口径，不拿名称当模型身份。[P102–P103]

loss 下降说明优化过程有信号；“生成可读短句”还需要原采样和人工判读。后续新增 autograd 不包含在最初 forward-only 归档任务里；服务端也不是教师版 P1 的必做范围。正确定性是扩大了资产，而不是强行把全部后续工作塞回归档完成率。[G01]

## 8. 本轮验证与下一步

- P1 解释器实际导入 `scratch/sys1`；`tests/test_calibrate.py` **11 passed**，包含完美校准、故意错误分布、温度排序和分桶基线。证明数学组件的这些用例，不证明真实数据 ECE 改善。
- decision 已跟踪树内容相同；`__pycache__` 差异单独排除。
- 未运行训练、一键 repro、MPS benchmark 或全量 CI；不重报历史绿数为本轮结果。

下一轮建议顺序：先补生产入口见证和三分数据 recipe，再决定是否改内核门控；性能优化另设基准，冻结 P1 的修改仍需单独授权。对应教材：[张量与梯度](../learning_path/02-tensors-and-gradients.md)、[P1 训练与决策](../learning_path/03-p1-training-and-decisions.md)。
