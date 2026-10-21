> **甲路挂起（2026-10-07 用户裁定）**：本域 C5 云端项全部冻结，解锁=p2-13 P2 行（三栈联调）达成；本地已完成项照旧。

# Tasks: p2-05-prod-sft [W1 · 依赖 p2-01+02+03]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/prod-sft/spec.md`。
>
> **执行环境按 design「执行环境（D6/D13 v3）」**：910B 全自研栈是正式路线，本地只做开发 + CPU 冒烟。
> 因此 B2 原文的"MPS fp16"属 v3 之前的旧口径，本波一律以 **CPU float32 + 0.6B 替身**执行（偏离已申报，
> 未改上游 spec）。云端项（B3b 正式档实跑、C1 对照表）**保持未勾，注记"待 C5"**。
> W1 开发路已交付本地半场；凭据均可复跑（`release/.venv/bin/python -m pytest tests/test_prod_sft.py -k <组> -q`）。

### 工作项 A 数据

- [x] A1 SFT 数据装配：hard 标签 + 教师伪标 soft（默认 50/50 可配），伪标全走缓存 —— 验证：`python -m pytest tests/test_prod_sft.py -k data -q` → **7 passed**
      真权重凭据：`head -1 release/runs/1005-p2-05-dev-rec-sft-qwen3-0-6b-kernel-4276/metrics.jsonl`
      → `asm_encoded 480 / asm_seen 480 / teacher_hits 480 / teacher_misses 0 / soft_coverage 1.0`（百步档 7389/50f1 同数，
      只是它们由"修记录口径前"的代码产出，装配行落在末行——数值不受影响，见 B2）；
      伪标全部来自 `bench/teacher_cache/p2_05_pseudo` 只读缓存（`DistCache` 以 read_only 打开，训练循环零教师前向，
      由 `test_cache_read_only_blocks_training_writes` 钉住）。数据口 = `sys1.eval.run.load_axis_records("quality")`。
- [x] A2 混合配比数值单测（soft 分量权重=配置） —— 验证：`python -m pytest tests/test_prod_sft.py -k mix -q` → **6 passed**
      覆盖 `mix_soft=0/0.5/1` 端点与线性混合、查不到伪标退 hard-only 且 `soft_coverage` 记的是事实值（不当成 0.5）、
      hard 走 p2-02 `hard_distribution`（字母 keys 传字母）不自己造 one-hot。

### 工作项 B 训练

- [x] B1 `production/sft.py`：LoRA r16/α32/lr1e-4/2ep 读出位 CE，复用 `decision/` 读出（grep 断言无平行实现） —— 验证：`python -m pytest tests/test_prod_sft.py -k loss_reuse -q` → **1 passed**（grep 门：`production/sft.py` 里读点只许出现 `option_scores`，无平行实现）
      附加对拍：`-k lora` → **4 passed**（LoRA 注入生效 + 与 peft 0.21 键名/初值/scaling/merge 口径逐条对拍）、
      `-k kernel` → **3 passed**（`kernel_backend: torch|kernel` 两路权重与梯度一致、合流增量的 delta 一致、cpu target 无 blocker）、
      `-k checkpointing` → **3 passed**（gradient_checkpointing 开关按/未按两枚钩子 + 无钩子时如实降级）。
      注释门：`release/.venv/bin/python scratch/tools/check_comments.py release/production/sft.py` → ✅ 通过（`wc -l release/production/sft.py` = 1521）。
- [x] B2 tiny SFT 冒烟 run（≤500 样本/≤200 step，~~MPS fp16~~ → **CPU float32 + 梯度检查点**，按 D6/D13 v3） —— 验证：runs/ 含 sft run-id，metrics.jsonl 非空
      **主曲线（kernel 主路线 100 步）**：`release/runs/1005-p2-05-dev-sft-qwen3-0-6b-kernel-50f1`
      （480 样本 / 235 桌 / metrics 101 行；loss 1.958338 → 1.260573，首尾各 10 步均值，**降 35.63%**；
      TileLang 实编 8 枚 `cpu|gemm` + `cpu|gemm_dw`；s/step 中位 28.10）
      **torch 参照档（同种子 100 步）**：`release/runs/1005-p2-05-dev-sft-qwen3-0-6b-torch-7389`
      （1.958339 → 1.260573；5.62 s/step 中位、73.4 tok/s；`model/adapter.safetensors` 224 张量，
      首键 `base_model.model.layers.0.self_attn.k_proj.lora_A.default.weight` [16,1024]）
      → **两条旁路 100 个逐点损失 max|Δ|=2.6e-5、均值 1.5e-6**（复跑：`.venv/bin/python` 逐行 zip 对比两份 metrics.jsonl）
      **gradient_checkpointing 真档**：`release/runs/1005-p2-05-dev-gc-sft-qwen3-0-6b-torch-25d7`（8 步，config 两处 `gradient_checkpointing: true`）
      **记录口径验证档**：`release/runs/1005-p2-05-dev-rec-sft-qwen3-0-6b-kernel-4276`（3 步；step0 装配行落在首行、自研件终局清点挂末步）
      **中止档（负结果入库，notes 有结论行）**：`...kernel-c11a`（step 4 中止：`loss_decrease_pct` 与展示值不同源，先修口径）、
      `...torch-5118`（step 29 中止：`limit 512` 越「≤500 样本」上限，改 `tiny_cpu.yaml: 480` 后重跑）。
      七条 run 前缀均为 `p2-05-dev`，结论行含"CPU 替身口径（本地开发档）；910B 正式档待 C5"边界声明。
- [x] B3 `configs/{gate08b,scaling_06b}.yaml`（diff 仅限 backbone/批规模/步数） —— 验证：`python -m pytest tests/test_prod_sft.py -k configs -q` → **3 passed**
      含 CLI 覆盖面断言；两档尾部按 design 补记**批规模折算比**（等效字数 = max_tokens × accum：gate08b 16384×4=65536 字/次下笔，
      scaling_06b 8192×8=65536 折算比 2:1；并声明"910B 是否 OOM 只有 C5 能给数，本地 s/step 不外推"）。
      计划外增第三档 `configs/tiny_cpu.yaml`（本地 CPU 冒烟专用，limit 480 / float32 / `kernel_backend: kernel` / `loader: minimal`），
      否则只能拿正式档去跑冒烟、必然越 spec 上限。
- [ ] 【执行期回写 2026-10-06·p2-09 隐患裁决】C5 发射前置两闸：① 训练口改 axis: train（现 quality 轴含 600 道中文决策考卷，照旧即训测同集假分）② p2-03 D3 中文 train 语料登记勾结
- [ ] B3b 【正式档实跑，P1 教训：配置存在≠实跑】gate08b 档真跑一段（LoRA 完整一步前向+反向+适配器落盘，MPS）：记录吞吐/显存/耗时入 run notes，并往 README 写命令指引供择机全量触发 —— 验证：runs/ 含 gate08b 正式档 run-id + README 含命令指引段
      **【待 C5】** gate08b 正式档 run-id（910B + 真 Qwen3.5-0.8B + 真伪标包 + 显存/吞吐实测）本地无卡无权重，跑不出这一行。
      可本地交付的两半已就位：① README「SFT（p2-05）命令指引」含云端触发命令（`--config production/configs/gate08b.yaml --run-prefix p2-05-sft`
      与 scaling 档，另注真伪标包只改 `teacher_cache`/`teacher_model_id` 两键）——验证：`grep -n 'gate08b.yaml' release/README.md`；
      ② 等价实路证据用本地替身档代跑（50f1 kernel / 7389 torch，完整"前向+反向+适配器落盘五件齐"），
      但这只证明通路，**不替代 gate08b 正式档实跑**，本项因此保持未勾。

### 工作项 C 验收

- [ ] C1 SFT 产物 typed-decisions acc 入对照表（与双基线并排） —— 验证：对照表 DML 列非 `—` 且带 run-id
      **【待 C5】** 两条硬缺口：① 对照表要收的必须是 910B 正式档产物，本地只有 CPU 替身权重；
      ② **数据侧缺口需上游确认**：quality 轴登记的四个集里，`envelope_ready=True` 的只有
      `typed-decisions`(split=test) 与 `intern-decision`(split=test)，两者合计即训练口的 14,447 问
      （choice 11555 / score 1618 / noul 1274），**registry 里没有 train split 条目**；
      `cmmlu-subset`(test)/`clue-subset`(validation) 尚未装配。
      复跑：`cd release && .venv/bin/python -c "from pathlib import Path; from sys1.eval import registry; from sys1.eval.run import axis_ids; m=registry.load_manifest(Path(registry.DATA_DIR)/registry.MANIFEST_NAME)['sets']; print({k:(m[k].get('split'),m[k].get('envelope_ready')) for k in axis_ids('quality')})"`
      —— 拿 test 集训练再报同集 acc 不成立，故本波不出任何评测分，正式档须先补 train split 登记（已在战报②上报）。
