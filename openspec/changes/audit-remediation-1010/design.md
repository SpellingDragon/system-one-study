# audit-remediation-1010 · design

## 决策

**D1 范围=声明与证据对齐，非工程补做**：审计 §6.3 明示修复义务止于"让声明与证据对齐"；RLCD 实现、910B 数值、正式训练属 p2-11/p2-13/C5 正常推进项，本变更不代排产、不阻塞它们。被否方案：把评审当需求清单全量开工——会把审计修复混入产品路线，模糊两套 DoD。

**D2 LICENSE=MIT**：`refs/clone.sh:32` 既有"本仓 MIT 树"自称，仓库治理一直按 MIT 假设运行；补 MIT 文本（版权人=仓库署名者，年份 2026）使内外一致。被否：Apache-2.0（与既自称冲突且更重）；不加（保留一切权利，第三方克隆无依据，与"活教材"定位矛盾）。

**D3 README 重写=按现态 `dba5dc0` 对账，非评审快照 `b4a399f`**。两态之间的增量必须反映（否则修完即过期）：
| 主题 | b4a399f（评审所见） | dba5dc0（现态） |
|---|---|---|
| GDN 反向 | partial 挂账 | **kernelized**（attempts/F 六梯；BWD_STATUS 已改，测试同步） |
| 接口 home `ascend/kernels` | 7/8 件 SimtVF、从未 target=ascend 编译 | **9/9 target=ascend 编译 PASS**（含 asc_fill_l1/set_l1_2d；数值/单位 V1–V6 待卡） |
| 训练接线 | 无 autograd.Function 消费方 | autograd_asc 8 件 + train_step host 绿（loss 0.911→0.824） |
| 孙任务计数 | 80 条/56 勾 | 81 条/57 勾（仍非"全勾"） |
其余口径修正：A–Z=38..63（P1 自训 BPE 实测）与 32..57（P2 Qwen 分词器）分属两幕；参数量 54.2M（总）/37.8M（非嵌入）双口径标注；GQA 删词（实现为 MHA）；"296 绿"以实跑数为准；R 编号以 skill 文件为准（README 不再写死数字区间）。

**D4 训练口 axis:train 立即执行，不等 C5**：雷未爆（dev run 只出 loss），但配置即点火即爆；且修复廉价（数据侧 train 轴已在、有隔离测试）。守卫=新测试断言"训练配置的 axis 不得为 quality/test 轴"，configs 同步切。p2-05:49 原闸完成时引用本变更 run 作凭据。被否：仅改文档提醒——审计已证明门面不受口头纪律管辖。

**D5 CI release job 最小形态**：ubuntu + release/.venv 自建 + `[dev]` 补齐缺失声明（modelscope/transformers/PIL）+ `pytest tests -q -m "not integration"`；MPS-only 用例靠现有 skip 门自然跳过。被否：上 mac runner 跑 MPS 门——成本高且非本变更义务。

**D6 scratch 冻结原则**：P1 层问题（GQA 措辞、~40M 口径、A–Z 数字、Decoder 内核不可达、OPD 谱系命名）**一律 README/design 措辞修正与等强度披露**，不改 scratch 冻结代码；勘误以 notes/归档件留痕（AGENTS 既有规则）。

**D7 报告处置**：评审原文归档 `evidence/eval_report.md`（第三方产物，只存不改）；对其 C/D 级主张（法律结论、硬件账目）不转为行动项，仅 P2 档记录论证义务。

## 风险与回退
- README 重写风险=引入新失实 → 验收用 grep 旧词零命中 + 逐数字溯源表（agent 交付物），编排者复核。
- configs 切 axis 影响 dev 复跑可比性 → 保留旧值注释与 run 档原样（历史 run 不改写）；新 run 起用 train 轴。
- CI 新 job 可能暴露既有红（评审实测 6–12 败）→ 首版允许 `continue-on-error` 采集基线，红项归因后转正式门（R23 三分归因，不静默放行）。
