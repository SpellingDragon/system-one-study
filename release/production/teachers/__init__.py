"""production.teachers — 教师适配层（接口：渲染后请求 + 候选键 → 选项分布）。

三教师分工（父设计 D9 终态）：
- 文本决策教师 = StartLux-Decision-4B（本地权重，末位字母读出，`text.py`）
- 视觉教师 = GLM-5.3-Flash 离线伪标包（`vision.py`，训练/评测链零在线 API）
- 文本备选 = Qwen3.5-4B（权重不可得时降级，接口同 `text.py`）

版权边界（D11）：StartLux 权重**仅**用于教师打分/RL 判分/对照评测三种只读用途，
绝不作为训练初始化或基座；分发物不含其权重，本层产出的只是分布数值。

模块面：`TextTeacher` / `VisionTeacher` / `DistCache` / `TeacherStats`。
"""
