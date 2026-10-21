# Tasks: p2-03-eval-registry [W0 · 无依赖]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/eval-registry/spec.md`。

### 工作项 A 注册与拉取

- [x] A1 `sys1/eval/registry.py` 骨架：三件套 pin（2f81580/7ce310c7/f7a2487e）+ `export_versions()` —— 验证：`python -m pytest tests/test_registry.py -k versions -q`
- [x] A2 `fetch --all` 幂等拉取（bench/ 落盘，失败集列名退出非零） —— 验证：`python -m sys1.eval.registry fetch --intern --typed && echo OK`
- [x] A3 中文/长文/多模态集注册：CMMLU/CLUE 子集、MMBench-CN 子集、LongBench-zh、合成 needle（seed 固定）+ 子集自持副本 —— 验证：`python -m pytest tests/test_registry.py -k extras -q`

### 工作项 B 锚点与调度

- [x] B1 Uniform 锚点自检：KL/TV/Brier 复现卡面 0.444/0.381/0.238（容差 0.01） —— 验证：`python -m pytest tests/test_registry.py -k anchor -q`
- [x] B2 六轴调度入口（`--axes all`，缺数据轴标 n/a） —— 验证：`python -m pytest tests/test_registry.py -k axes -q`
- [x] B3 【语义澄清，P1 教训回写】parity 轴在 P2 的对象裁定：被测主体是 HF backbone（无 backends 双路径可言），故 P2 的 parity 轴显式降级——或测 P1 自训 decoder 的延续路径，或标 n/a 并在报告注明"backbone 前向即 torch-MPS 单一路径"；裁定结论写入 spec 与报告模板 —— 验证：spec/报告模板含裁定结论，无悬空轴


## D 组 · 执行期回写（2026-10-05 p2-05 发现，R11 消费者漏枚举实例）

- [x] D1 登记 **train 分割**装配：typed-decisions train 真拉 + Intern-Decision train 分区（registry 现仅 test 口径——p2-05 训练装配复用同口导致训测同集假分风险）；`load_axis_records` 增 split 参数或新 axis `train`，qtype 三口一致 —— 验证：`.venv/bin/python -c "from sys1.eval.run import load_train_records; rs=load_train_records(); assert rs and all(r['split']=='train' for r in rs); print(len(rs))"` exit 0 且 run notes 追加 train 底账行
- [x] D2 spec 补 Scenario：训练数据消费面（train/test 隔离断言：任何 train 记录 id 不得出现在评测 quality 轴集内）—— 验证：`pytest tests/test_registry.py -k split_isolation -q` exit 0
- [x] D3 中文 train 语料登记（cmmlu/clue train 档决策化入 axis=train，与 test 考卷 id 互斥——承接 p2-09 隐患裁决，C5 前置闸之二）—— 验证：load_train_records() 含中文档 且 split_isolation 扩至中文对 exit 0
