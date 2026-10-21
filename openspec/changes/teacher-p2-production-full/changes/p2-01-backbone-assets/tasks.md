# Tasks: p2-01-backbone-assets [W0 · 依赖 P1 decision 契约（冻结）]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/backbone-assets/spec.md`。

### 工作项 A 装载与校验

- [x] A1 `production/assets.py`：`load_backbone(name)`（ModelScope 优先/HF 兜底，fp16）+ repo/revision 写入 run config —— 验证：`python -m pytest tests/test_assets.py -k load -q`
- [x] A2 接缝三校验：字母单 token / letter_rows 随 hidden_size 重建 / 思考关闭模板快照（实测固化前缀） —— 验证：`python -m pytest tests/test_assets.py -k seams -q`
- [x] A3 mock 失败单测：三型接缝破坏各抛 `SeamCheckError` 且消息含接缝名 —— 验证：`python -m pytest tests/test_assets.py -k seam_fail -q`
- [x] A4 换脑冒烟：载重后以一阶段 `decision/` 程序决策 3 问（不 import 一阶段 model.py） —— 验证：`python -m pytest tests/test_assets.py -k swap_brain -q -m mps`

### 工作项 B 设备适配

- [x] B1 fp16 + MPS 迁移与显存基线记录（0.6B 前向峰值入 runs notes） —— 验证：run notes 含峰值内存行
- [x] B2 【真实路径验证，mock 不算完成】backbone 真拉真载一次：ModelScope 实下载 Qwen3-0.6B（非 monkeypatch/非本地缓存预置），记录下载字节量/耗时/镜像源入 run notes —— 验证：run notes 含真实下载实测行（P1 教训：远端代码路径 mock 通过≠路径可用，corpus 远端直到执行期才首验）
