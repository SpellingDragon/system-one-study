# run: 1005-p201-backbone-assets-realpath-5d5c

- 假设：Qwen3.5-0.8B 能从魔搭真下载并真载，四道载重接缝全过（字母单 token 是全计划最大单点），HF 权重换成自研栈布局后整层前向与 HF 的相对差进 1e-3，换脑后一阶段决策程序零改动仍出合法份额
- 观察：
- 真实下载实测（ModelScope，全新空目录首拉，非缓存命中、非 monkeypatch）：1,769,980,952 字节 / 311.3s / 镜像源 https://www.modelscope.cn / 14 个文件；repo=Qwen/Qwen3.5-0.8B revision=master。
- 真载实测（CPU + fp16）：load_seconds=17.0s，权重占用 1,706.0MB，hidden_size=1024，峰值常驻内存 peak_rss_mb_at_load=2027.6MB、整轮结束 2027.6MB。
- 校验①字母单 token：A–Z 26 格全占独号（32..57），52 大小写均单格，拼进渲染串后边界不吞并——通过；未触发降 0.6B 预案。
- 校验②letter_rows 随隐宽重建：形状 26×1024（26 行取自 lm_head，宽度跟 config）——通过。
- 校验③思考关闭模板快照：canonical prompt 80 格、sha256 前 16 位 a1891aca6c002e48，尾巴逐字节等于 decision 的关闭后缀——通过（快照已固化入单测）。
- 校验④多模态编号不撞号：{'image_token_id': 248056, 'video_token_id': 248057, 'vision_start_token_id': 248053, 'vision_end_token_id': 248054}，与字母号 32..57 无交集——通过。
- 权重布局对拍 layer3（自研栈布局 {'heads': 8, 'kv_heads': 2, 'head_dim': 256, 'rotary_dim': 64, 'group': 4}）：layer3 out=[8, 1024] max_abs=4.470e-07 mean_ref=7.233e-02 rel=6.180e-06 tol=1.0e-03 OK；行号抽样逐元素匹配=True，不匹配位=[]，转换产出 5 个张量键 ['gate.weight', 'k_norm.weight', 'o_proj.weight', 'q_norm.weight', 'qkv.weight']。
- 权重布局对拍 layer7（自研栈布局 {'heads': 8, 'kv_heads': 2, 'head_dim': 256, 'rotary_dim': 64, 'group': 4}）：layer7 out=[8, 1024] max_abs=1.788e-07 mean_ref=5.042e-02 rel=3.546e-06 tol=1.0e-03 OK；行号抽样逐元素匹配=True，不匹配位=[]，转换产出 5 个张量键 ['gate.weight', 'k_norm.weight', 'o_proj.weight', 'q_norm.weight', 'qkv.weight']。
- 换脑冒烟（一阶段 decision 程序原样跑，零改动、不 import 一阶段 model.py）：3 问 q1→A/a(Σp=1.0000), q2→A/a(Σp=1.0000), q3→A/a(Σp=0.9999)。
- 因果不变性右填充对照：批宽 81、长度 [80, 80, 81]，垫与不垫两种送法候选分最大绝对差 3.815e-06（同长度尺度 ~24.43），argmax 一致=True（['A', 'A', 'A'] vs ['A', 'A', 'A']）。
- 渲染对齐：串由 sys1/decision/render 只读产出（RENDER_VERSION=dmlaya_render_v1），编码走 Backbone.encode_prompt，与 decision 的字母序/候选序一一对上（单测 test_render_alignment 覆盖）。
- MPS 迁移冒烟（小张量，不载权重，因 MPS 被一阶段长跑占用）：{'available': True, 'matmul_finite': True, 'device': 'mps:0', 'dtype': 'torch.float16', 'allocated_bytes': 24576}；完整 0.8B 的 MPS 前向峰值按派单延后（CPU 口径已给峰值常驻内存行）。
- 测试复跑：全量 28 passed, 1 skipped, 1 warning in 57.14s（rc=0）；A4 口径 -k swap_brain -q -m mps → 1 skipped, 28 deselected in 1.37s（rc=0）。
- 更正（写入工具事故后补跑）：上面"测试复跑"那行的 28 passed 只含 tests/test_assets.py——tests/test_backbone_layout.py 在 20:16 被 IDE 写入工具的延迟落盘覆盖成 1 行占位，20:17 的复跑因此收不到它的用例。该文件已按 pytest 缓存里的 16 个用例名重写并复跑：`python -m pytest tests/test_backbone_layout.py -q` → 16 passed in 15.72s（含两处真权重用例：layer3/7 一层前向对拍 rel 6.180e-06/3.546e-06 ≤1e-3、512 行逐元素抽样 ok）。
- 全量真数（重写后）：`python -m pytest tests/test_assets.py tests/test_backbone_layout.py -q` → 44 passed, 1 skipped in 60.55s（唯一 skip 是换脑的 MPS 变体：本机 MPS 被一阶段长跑占用，按 SYS1_ALLOW_MPS 门控）。
- 逐孙任务口径复跑：`-k load` → 8 passed, 21 deselected in 14.17s；`-k seams` → 5 passed, 24 deselected in 6.60s；`-k seam_fail` → 15 passed, 14 deselected in 5.06s；`-k swap_brain` → 1 passed(CPU 实测), 1 skipped(mps 变体), 27 deselected in 18.67s；A4 原文口径 `-k swap_brain -q -m mps` → 1 skipped, 28 deselected in 1.60s。
- 口径说明（防 config 与本笔记数字被读成互相矛盾）：config.yaml 里 backbone_download_bytes=0 / seconds=3.3 是**本 run 复跑那次**的读数——快照已在 bench/ms_models 命中缓存，净增下载 0 字节；真实首拉的 1,769,980,952 字节 / 311.3s 记在上一条观察与 metrics step=0，来源是往空目录 bench/p201_dl_fresh 的首拉实测（日志 .out_p201_redownload.txt 尾行 PROVENANCE_JSON）。

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
结论：p2-01 六项孙任务落地：① 真下载 1,769,980,952 字节 / 311.3s（https://www.modelscope.cn，14 文件，空目录首拉）；② 真载 CPU+fp16 峰值常驻内存 2027.6MB（载重后）/2027.6MB（整轮）；③ 四校验全过——字母 A–Z 单 token 32..57（未触发降 0.6B 预案）、letter_rows [26, 1024]、思考关闭快照 80 格 / sha a1891aca6c002e48、多模态编号与字母号无交集；④ 布局对拍 2 层 rel 最大 6.2e-06 ≤1e-3；⑤ 换脑 3 问份额和为 1、右填充因果不变 max_abs=3.815e-06 且 argmax 一致；⑥ MPS 完整前向峰值按派单延后（本机 MPS 被一阶段长跑占用），本 run 只交 fp16→mps 小张量冒烟，CPU 口径峰值已入笔记。
结论：更正（补跑，零新下载）：布局层测试文件曾因写入工具延迟落盘被覆盖为占位，本 run 首跑的"全量 28 passed"因此少计了 tests/test_backbone_layout.py；该文件已重写并复跑 16 passed（真权重换算与一层前向对拍在内），全量真数 44 passed + 1 skipped（MPS 变体按占用状态 skip）。四校验、下载字节量/耗时、峰值内存各行数值不受影响。
补录（编排者，2026-10-06 规整）：bench/p201_dl_fresh 快照已按"可删皆删"授权清除（数字以本 notes 下载实测行为准）；原始下载日志迁存于本 run 的 process_artifacts/out_p201_redownload.txt；重建路径 = config.backbone_repo+revision 经 ModelScope snapshot_download。
