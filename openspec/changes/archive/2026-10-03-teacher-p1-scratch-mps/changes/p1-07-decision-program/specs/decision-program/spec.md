## ADDED Requirements

### Requirement: systemone 渲染
`dmlaya/decision/render.py` SHALL 把 `/v1/systemone` 请求渲染为 chat prompt：固定 system 行 + `Evidence:\n<state>` + `Question: <instructions>` + `Options:\nA) ...`（字母序）；含 `RENDER_VERSION` 常量；`from_systemone()` 解析请求体并过 data-schema 校验。

#### Scenario: 渲染快照
- **WHEN** 渲染一个 state+3 选项的 choice 问题
- **THEN** 输出与教师版黄金快照逐字节一致（快照入 tests）

#### Scenario: 渲染确定性
- **WHEN** 同一请求渲染两次
- **THEN** 输出完全相同（无时间戳/随机序）

### Requirement: 末位字母读出
`dmlaya/decision/readout.py` SHALL 以"末位 hidden × 词表 26 字母行"计算选项 logits，随后按 qtype 应用类型温度再 softmax；读出 MUST 满足右 padding 因果不变性（读出位之前无 pad），且全程 `output_tokens = 0`（无生成循环）。

#### Scenario: 右 padding 不变性
- **WHEN** 同一样本单独前向 vs 批内右 pad 前向
- **THEN** 选项概率最大偏差 ≤ 1e-5

#### Scenario: 无生成断言
- **WHEN** 静态检查 decision/ 源码
- **THEN** 不存在 `.generate(`/采样循环调用（测试断言）

### Requirement: 类型温度应用
`dmlaya/decision/temperature.py` SHALL 按 qtype 施加 `calibrate.py` 产出的温度表（clamp 到 [0.2,5]），温度 MUST 不改变 argmax；缺温度表时 SHALL 用 1.0 并显式标注 `uncalibrated`。

#### Scenario: 温度不改 argmax
- **WHEN** 对含 tie-break 的 logits 集施加任意合法温度
- **THEN** argmax 序列与 T=1 时一致

### Requirement: >26 选项分组票选
`dmlaya/decision/wide.py` SHALL 对 k>26 的 choice 选项分组（每组 ≤25 + 剩余槽），多轮读出后 top-keep 进决赛、未入选者保留残差概率，输出分布和为 1。

#### Scenario: 五十选一分布合法
- **WHEN** k=50 选项走 `_wide` 票选
- **THEN** 输出恰好 50 个概率，和=1（容差 1e-6），且决赛组内序与字母序一致
