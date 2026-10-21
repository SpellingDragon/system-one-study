# p2-11 RL 产物（1010-p2-11-dev-rl-qwen3-0-6b-torch-bf64-3）

【来源链】
- 题目：registry 轴 `quality`，编成 8 条（过目 15047 条）
- 奖励：严格适当 proper_reward（log+sph+score类 RPS），**零教师**（D11：reward 来自自我打分）
- 起点：`未给起点（从 base 起步）`
- 底座：`Qwen/Qwen3-0.6B`（载入口 minimal）
- 旁路：LoRA r16/α32/dropout 0.05，实现档 torch@cpu
- 训练：6 步，reward -3.050061 → -4.407127；温度后置拟合=4.0；早停=True

> CPU 替身口径的本地半场冒烟；910B + C5 正式档待 C5。
