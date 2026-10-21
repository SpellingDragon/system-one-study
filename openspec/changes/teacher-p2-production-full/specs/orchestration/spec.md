## ADDED Requirements

### Requirement: 跨变更启动门
本变更（P2）的任何二级子变更 SHALL NOT 开始实施，直至前置变更 teacher-p1-scratch-mps 已归档（或至少其 p1-04/p1-05/p1-07/p1-10 四域已按序合并入 main——即 decision/、calibrate、eval 四轴、backends 契约可用）；启动时 MUST 在一级 tasks.md 勾选下方“启动门”声明实际满足的分支。

#### Scenario: P1 未就绪即启动被拦
- **WHEN** P1 的 p1-07（decision-program）尚未入 main 而有人开始实施 p2-01
- **THEN** 依本契约拒绝：换脑冒烟依赖 decision/ 实现，接口冻结≠实现存在；同时活跃两变更时以归档顺序裁决

### Requirement: 二级子变更结构
一级变更 SHALL 以 `changes/p2-NN-<domain>/` 承载全部二级子变更，每个子变更 MUST 自包含 `{proposal.md, design.md, tasks.md, specs/<capability>/spec.md}` 四件套；一级 `specs/` 仅保留本编排契约（orchestration）。

#### Scenario: 子变更自包含
- **WHEN** 检查任一 `changes/p2-*/` 目录
- **THEN** 四件套齐备，且其 spec 能力名与一级 design D2 表一一对应

### Requirement: 一级任务只编排不实现
一级 tasks.md SHALL 仅含：编排跟踪图（mermaid，随进度更新）、子变更级 checkbox（完成 = 其孙任务全过 + spec 全过 + 按 D4 合并 + 训练类附 run-id）、一级直管收尾任务；MUST NOT 含实现级孙任务。

#### Scenario: 编排与实现分离
- **WHEN** 读一级 tasks.md
- **THEN** 无 `- [ ]` 项涉及具体代码文件编写；每子变更项附 `changes/<name>/` 路径

### Requirement: decision 契约零改动
本变更全程 SHALL 保持 `sys1/decision/` 零改动（两轨同构桥梁）；收尾 F1 以 diff 为空断言；任何子变更 PR 触及该目录 MUST 被拒（父 design D4-4）。

#### Scenario: 契约破裂被拦
- **WHEN** 某子变更 PR 含 `sys1/decision/` 的改动
- **THEN** review 依据本契约拒绝；若确有契约级缺陷，先开独立变更修订 P1 spec 再回来

### Requirement: 负结果入库
三栈（06/11）与扩展域的消融/冒烟 run SHALL 如实记录（含零增益、发散、OOM 降档）；负结果 run 不得删除，notes 必含结论行——教师版"踩坑"交付物的定义。

#### Scenario: 发散 run 保留
- **WHEN** RL tiny run 发散早停
- **THEN** run 目录保留且 notes 含稳定域标注；对应孙任务仍可勾选（交付物=坑记录而非绿曲线）

### Requirement: 变更级 DoD 收尾
一级收尾 SHALL 核验 design D5 六条并记录于最终 run notes 后方可 `openspec archive`。

#### Scenario: DoD 未过不归档
- **WHEN** F1–F3 有未过
- **THEN** F4（归档）不得执行


## 2026-10-05 修订（昇腾 910B 云端版，v3）

- 启动门增补：**C5 gate 训练须待轨 A（P1 mps_main 收官评审）完成 + C4 报价获批（¥600 熔断线，D12）**。
- 子域清单增补第 13 域 `ascend-runtime`（G9 验收面）；三栈（5/6/11）与 10 依赖其底座决策（C1 探针裁定全自研 or torch_npu 回退）。
- 版权边界（D11）为合并硬拦项：任何把 StartLux 权重用作训练初始化/基座/再发布的 PR 直接拒绝。
- 成本纪律（D12）：各云端段 run notes 无时长/费用行 = 孙任务未完成（四查之数字溯源）。
