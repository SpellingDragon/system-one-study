# 第三批：P2 的真实路径与研究有效性

## 结论

三栈与扩展的研究方向仍可保留，但当前证据主要支持**组件、替身和局部端到端流程**。正式 0.8B/910B 三栈收益尚不能据此成立。最优先的补差不是立即增加训练步数，而是训练数据入口、算法契约和正式消费者。[G03、P201–P210]

## 1. P2-F1：教师身份是验收的一部分

`1005-p202-teacher-adapters-b3-realpath-b570/b3_payload.json` 明确记录：

- 权重目录共 9,344,023,187 字节；这包含配置等文件，不全是权重字节。
- tokenizer 检验成功；`real_4b_load.attempted=false`，因为内存守卫阻止完整载入。
- 打分主体是 `StartLuxAI/StartLux-Decision-4B#scaffold-cpu`，36.6M 随机初始化替身。
- 该批 GLM 请求 10 次、实际调用 14 次、落标 2 条；这是该批格式和效果问题，不能外推所有 GLM 用途。[P201]

替身使管线开发得以继续，是合理工程选择；但 p2-02 的真教师入口验收不应由“权重在盘＋替身打分”替代。后续其他批次的视觉包另行计数，不将第一批 2 条当全仓总量。

SFT dev run 使用 0.6B 学生与带 scaffold 后缀的伪标身份。命中缓存、loss 下降能证明装配与优化信号，不能证明从真 4B 获得知识，也不证明目标 0.8B 的正式训练收益。[P203]

## 2. P2-F2：OPD 默认读取评测轴，既有守卫漏了这个入口

可直接核验的链：

1. `release/production/opd.py:920` 默认 `axis='quality'`。
2. `:1016` 将该值交给 `load_quality_records`，随后装配样本并训练。
3. `release/tests/test_train_axis_guard.py:62-71` 只枚举 SFT 默认值与 yaml，不枚举 OPD 默认值。
4. 本轮守卫测试 **7 passed**，但直接导入 OPD 打印仍为 `quality`。[P204、V02–V03]

这是测试范围漏掉生产消费者，不是“守卫断言写错”。当前未发现正式 OPD run，所以定性为**启动时的数据泄漏风险**，不声称已经污染某次正式训练。建议覆盖所有训练入口及显式错误 axis，并在公共装配路径验证 split；修复前不释放相应训练任务。

## 3. P2-F3：实际损失和采样不等于原计划中的算法

`opd.reverse_kl` 的正文计算 `Σ p_teacher·(log p_teacher−log p_student)`。正常有效概率域下是 `KL(teacher || student)`，相对教师目标通常称 forward KL。函数名和注释中的“reverse”不能改变数学。

本轮用 teacher=`[0.9,0.1]`、student=`[0.5,0.5]` 调实际函数得到 `0.3680641055`，与手算方向一致；相反方向约 `0.510826`。当前实现还包含极小概率 mask 和 log floor，极端输入不宜直接当作无截断理论 KL。[P204、V03]

采样方面，`sample_topk` 按分数选子集；`train` 在优化器循环前一次调用 `collect_onpolicy`，没有在更新中持续按新学生重新生成序列。它是**选项子集上的蒸馏实现**，不是文献 GKD 的自回归学生序列 on-policy 流程。

[GKD 原文摘要](https://arxiv.org/abs/2306.13649v3)明确强调学生自生成输出序列与分布不匹配问题，也允许不同散度；因此“用了 forward KL”不单独否定 GKD，真正要核对的是数据生成过程、概率空间和更新时机。

两条可选路线：承认并命名为选项级蒸馏，研究它本身；或者实现原计划的序列级 on-policy 契约。此选择会改变目标与资源预算，本轮不擅自改损失或名称。

## 4. P2-F4：LoRA 旁路、完整 backbone、自研布局是不同接线

`production/sft.py` 的 kernel 选项接入自研 LoRA 增量；`student_forward:1014` 仍调用 `student.body(input_ids, attention_mask, ...)`。其 CPU 对照是 LoRA 旁路等价性，不是完整 backbone 被替换的证明。[P203]

`production/backbone/layout.py` 与 loader 有独立实现和对拍，而正式 `assets.load_backbone`／SFT 入口是否消费这条转换，需要专门接线见证；一级标题写了布局转换，也不代表所有下游都在使用它。[P202]

LoRA 冻结大部分参数减少可训练参数，并不普遍消除冻结层的 backward。梯度检查点则以重算交换激活存储，也不保证任意模型／设备都省到相同内存。下一轮应使用实际 0.8B 输入与训练图核算，而非按 0.6B CPU 替身外推。

## 5. P2-F5：中文配置尚未成为训练行为

`zh_mix_30.yaml` 已明确声明 `zh_ratio/zh_corpus` 是待接入键。当前 SFT 没有这些消费字段；这不是悄悄生效的 30% 中文训练，直接喂配置还可能被未知键校验拒绝。

同文件“registry 尚无中文 train 数据”的说明已经落后于注册层。故后续任务应拆开：确认可用且合规的中文 train 来源 → 装配器实际控制抽样比例 → 记录实际样本占比 → 用未参与训练的中文评测集比较。两个 yaml 的差异只能证明配置意图。[P205]

## 6. P2-F6：多模态的组件门与报告门没有同步

p2-08 A1 有 processor／视觉塔探针；B1 加／去图 run 未勾，而 B2 的“≥200 题、acc、样本数、run-id、无图列”报告门已勾。现有证据定位到组件与单测，未取得兑现 B2 报告门的产物，应列**待补证／建议重开**，不把缺文件直接改成通过。[P206]

另一个事实冲突：二级 spec 仍要求 SigLIP/CLIP 和两段式解冻，而当前设计主路线是基座自带塔冻结、SigLIP 为备选。应按有效裁决修订 spec，不按过时 spec 自动重写实现。

`student_forward` 只传文本 IDs/mask，mm 组件没有自然地变成训练消费者。验收中需要真图进入模型、无图对照，以及图像改变时的可观测路径；仅拼出占位 token 不足以证明视觉信息被使用。

## 7. P2-F7：长文证据要按真实 token 数和题数读

`1006-p2-07-a2-needle-curve-v3-1035/metrics.jsonl` 共七条：

| 配置桶 | 实际 ctx_tokens | 命中 | 可支持的判断 |
|---|---|---|---|
| 8K | 7457、7474 | 1/2 | 这两个样本的观察 |
| 32K | 29563–29648 | 1/4 | 小样本方向性观察 |
| 128K | 118129 | 0/1 | 该样本运行且未命中，非完整 128K 能力界 |

最大 peak 为该条日志记录的 18.46GB。配置桶不是精确输入 token 长度，七题不是有统计说服力的长度曲线，更不支持 1M 扩窗。[P207]

本轮核对 [Qwen3.5-0.8B 官方卡](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/README.md) 的 Model Overview：24 层、6×(3 Gated DeltaNet＋1 Gated Attention)、原生 context 262,144。它是模型规格声明，不是本项目受限设备上的吞吐或检索实测；对 1M 增量参数和全注意力层数的旧估算应以实际配置复算，不沿用别的档位。

## 8. P2-F8：服务存在真实正面证据，基线 gate 不要误读

服务 run `1006-p210-serving-local-half-c5d6`：Qwen3-0.6B、CPU float32、warmup 20、n=50，P50=382.662ms；endpoint/local 40/40 argmax 一致，worst probability drift=2.05e-5；三次请求 prefix hits 增量 2。这是生产消费者正面证据，但不是目标 NPU 档。[P208]

双基线表明确标为 smoke24，DML 列为 `—`，full 待 C6。源码将真实 harness 行标 `gate=true`、卡面降级行标 false；**不能仅因 24 题就指控 gate 字段实现错误**。应澄清它是来源／可参与比较标记，不是显著超越的判定。24 题上的两基线相同准确率不支持 DML 已胜出；不同设备的卡面毫秒值不能进入本地 Δ。[P209]

## 9. P2-F9：正常未完成与复现欠账分开

RL 和 tech-report 的零勾选与当前未交付状态一致，不算虚报完成。个别 RL 探索 run 有 reward／梯度记录，但正式可复跑源码与充分 notes 尚未形成，不能把该目录当作三栈完成证据。[P210]

[Rewarding Doubt 原文摘要](https://arxiv.org/abs/2503.02623v6)讨论生成答案及表达置信度、基于 logarithmic scoring rule 的奖励。仅共享 log score，不足以声称本项目零生成选项决策与其算法同构。报告可表述“借鉴 proper scoring rule”，具体动作、概率、reward、梯度估计与 baseline 仍待设计和实跑。

## 10. 许可与研究主张边界

本轮读取 [CC BY-NC 4.0 正文](https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt)：§2(a)(1) 的授权具有 NonCommercial 条件。是否涵盖具体蒸馏产物、训练用途及第三方输出，需要结合作品、司法辖区、模型许可和 API 合同判断。旧 `docs/compliance.md` 中“低风险／大概率在范围外”是推断，不由内部 D11 自动保证。

GLM 账户对应服务条款、签约实体和适用版本本轮没有取得足够原文，保留为全量产标／发布前的授权核验项。本轮不作法律意见，也不把一个模型卡的 Apache 标签扩展到所有教师和数据。

## 11. 本批验证边界与建议顺序

训练轴守卫 7 项本轮通过；OPD 默认轴与 KL 方向已直接调用核实。首次测试因当前目录的未构建 tilelang 源码遮蔽已安装包，在收集期失败；显式 PYTHONPATH、Python `-P` 后通过。未安装依赖或修改环境配置，也未修改产品。

研究子代理报告过全量套件绿数，但主评估者未复跑完整套件，不将其列为本轮全量验收。未执行任何正式训练、产标或 NPU 测试。

建议先完成数据／算法契约和真实消费路径的补差，再报价正式实验；中文、视觉、1M 先做可判定最小对照，不再以配置存在或组件测试替代研究结果。
