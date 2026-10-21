# sys1/eval/ — 唯一评测 harness（一切数字的出处）

> 契约：**两轨所有跑分只许出自这里**；论文中每个数字必须能溯源到一次带 run-id 的评测调用（PRODUCTION §9/§11-0）。

## 计划文件

| 文件 | 职责 |
|---|---|
| `registry.py` | benchmark 注册：数据集 id + **pin（git commit / HF revision）** + split + 采样参数 + seed；提供 `git rev-parse HEAD`/revision 清单自动导出（PRODUCTION §6-2） |
| `predict.py` | 统一 predict 接口：本地 `--model` 或 `--endpoint` 任一 `/v1/systemone` 服务（DML、laya、StartLux 三者同路） |
| `report.py` | 四轴同一分母连排 + 退出码即判定（repro_p1.sh 的评测段；p1-10 交付时新增于 design 清单之外，因 shell 无法保证同分母） |
| `quality.py / calibration.py / longctx.py / multimodal.py / speed.py / parity.py` | 六轴：决策质量 / ECE+可靠性图+弃权 / needle+截断回归 / 图文决策 / 延迟吞吐显存 / 跨后端一致性 |
| `baselines/` | **双强制基线亲跑产物**：laya、StartLux-0.8B（Jev 仅卡面数字作参考列）；对照表 `{metric, DML, laya, startlux-0.8b, Δ, (jev ref)}` |

## 口径不变量（写死在测试里）

- Uniform 基线必须复现 typed-decisions 卡面 `KL/TV/Brier = 0.444/0.381/0.238`（复现不出=harness 接错）。
- ECE：typed=15 bins top-label；JevBench-hard=10 bins（上游 `ece_top_label`）；**两种口径禁止混报**。
- 预测行：`{"id","answers":{q:{opt:p}},"ms"}`；键约定见 decision/README。
- 速度：注明设备与精度模式；**laya 速度基线仅 CUDA 有效**（MPS/NPU 只报 DML 自身）。
- 分桶随机基线（GUIDE §6）：choice `1/k`、noul `0.5`、score 均匀 MAE `(k²−1)/(3k)`，按 `qtype×选项数` 分桶报。

## 拉数据

benchmark 三件套与 Decision Index 的获取/运行命令在 PRODUCTION §5.0（pin：Intern-Decision@`2f81580`、jevbench@`7ce310c7`、typed-decisions@`f7a2487e`）；产物默认落仓库根 `bench/`（已 gitignore）。
