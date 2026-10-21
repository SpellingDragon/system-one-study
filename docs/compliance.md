# 对外权利推演与条款核验（audit-remediation-1010 · R-P2-2 入档，2026-10-11）

> ## ⚠️ 非法律意见（本文抬头，不可裁剪）
> 本文是**工程账目下的权利推演与核验清单**，由本仓执行代理撰写，**不构成法律意见**，作者非法律从业者。
> 凡标注 ❓ 者为推断；凡标注"待核验"者为**未取得原文、不得当作已核验**。任何对外分发、商用、开源发布之前，
> 须取得执业法律意见，本文不能作为替代凭据。
>
> **本文不得把 D11 当法律结论**：D11（`openspec/changes/teacher-p2-production-full/design.md:168-172`）是本仓
> 2026-10-05 自订的**家规**，只约束本仓工程动作（什么能做什么不做），**不产生对外权利**，也不构成对第三方许可的
> 解释结论。评审 §2.5.2 原话："用内部规定（D11）为外部合规背书，是把家规当成了法律意见书"。门面已按此改口径
> （✅ `README.md:22`："D11 是**本仓自订**的版权边界，不等于对外合规结论"）。
>
> 三态标注（图例 `.agent/rules/common.md:90`，纪律 `:115`）：✅ 实测确认（本会话跑过命令）／⚠️ 文档承诺（只读到文字账目）／❓ 推断。
> 注：调用方转述的"项目纪律 §六.2"经本会话核对，三态**图例**在 `common.md:90`（§六.3），"结论区分文档承诺/实测确认/推断"
> 在 `common.md:115`（§十.2）——本文按实测行号引用，未沿用转述编号。

## 0. 结论速览

| 议题 | 本会话推演结论 | 风险等级 | 必须先重答的时点 |
|---|---|---|---|
| §1 CC BY-NC-4.0 教师输出用于蒸馏学生 | 依本地 CC 全文文本推演：**取用其"选项概率分布"训练学生，大概率落在许可射程之外**（§1(f)(g)(a)、§3(a) 的条件性），但结论系于两条未证前提；且**该动作至今未发生**（真教师输出 0 行入库，✅ 实测） | **低（现态）→ 中低（C3 产出真伪标后）** | ① C3 用真 4B 产标；② 对外分发/第三方商用学生产物；③ 改用自然语言伪标 |
| §2 GLM-5.3-Flash API ToS | **未核验**。全仓对 Z.ai/智谱服务条款的分析命中 0 行（✅ 实测 grep）；文书里的"红线合规"经追出处确认为本仓架构纪律（`scratch/PRODUCTION.md:366` §11-4），**同义反复已破** | **中**（执行侧就绪：25 次真调用已发生、12 行已落包、3–6k 全量已规划） | C3 全量产标开闸前必须完成核验并归档原文 |

---

## 1. CC BY-NC-4.0 教师输出用于蒸馏学生

### 1.1 事实基线（全部本会话实测，可整段复跑）

```bash
cd $REPO_ROOT
# 教师许可：仓内快照含卡面 + CC 全文 + NOTICE
ls release/bench/ms_models/models/StartLuxAI--StartLux-Decision-4B/snapshots/master/{LICENSE,NOTICE,README.md}
sed -n '2p;141,143p' release/bench/ms_models/models/StartLuxAI--StartLux-Decision-4B/snapshots/master/README.md
wc -c release/bench/ms_models/models/StartLuxAI--StartLux-Decision-4B/snapshots/master/LICENSE   # 19347 字节＝BY-NC 4.0 全文
grep -n "Produced Material" release/bench/ms_models/models/StartLuxAI--StartLux-Decision-4B/snapshots/master/LICENSE   # → 0 命中
# 教师输出的实际形态与身份（本会话最关键的新事实）
cd release/bench/teacher_cache && ../../.venv/bin/python -c "
import pyarrow.parquet as pq,glob
for f in sorted(glob.glob('**/*.parquet',recursive=True)):
    t=pq.read_table(f);print(f,t.num_rows,sorted(set(t.column('model_id').to_pylist())),sorted(set(t.column('source').to_pylist())))"
```

| 事实 | 取值 | 出处 | 三态 |
|---|---|---|---|
| 教师权重许可＝CC BY-NC **4.0** | `license: cc-by-nc-4.0`；卡面正文"model weights … CC BY-NC 4.0 … **Commercial use requires a separate license** … The inference code in `startlux_decision/` is Apache-2.0" | `…/StartLux-Decision-4B/snapshots/master/README.md:2,141-143`（0.8B 同口径 `:2`） | ✅ |
| CC 全文在仓内（不必靠记忆） | 19,347 字节 Legal Code + NOTICE（NOTICE 另声明权重是 Alibaba Apache 2.0 模型的**修改版**，tokenizer 等原文件仍 Apache 2.0） | 同目录 `LICENSE`、`NOTICE` | ✅ |
| **协议对"输出"沉默** | 全文 `grep "Produced Material"` = **0 命中**；定义面只覆盖 Licensed Material / Licensed Rights / Adapted Material / Share | `LICENSE`（无 CC 为模型专门化加的"Produced Material"条款） | ✅（文本事实）＋❓（其法律意义） |
| 真教师输出**尚未入库** | 缓存 4 个 shard：562 行 `…4B#scaffold-cpu`（同架构随机权重替身）＋12 行 `glm-5.3-flash`；**来自 CC BY-NC 真权重的分布/伪标 0 行** | `teacher_cache/**/*.parquet`（model_id 实测值，命令见上） | ✅ |
| 替身与真教师天然分家 | 缓存键 `sha256(model_id + prompt + sorted(keys))`，换 model_id 必 miss；有守卫钉住 | `release/production/teachers/cache.py:10,58`；`teachers/verify_b3.py:65`；`release/tests/test_opd.py:747` | ✅ |
| 真 4B 权重已下载、未载入 | 9,344,023,187 字节在场；载入被内存守卫拦（可回收 4395.1MB／次跑 3009.7MB），延后云端 C3 | run `1005-p202-teacher-adapters-b3-realpath-b570/notes.md` | ✅ |
| 真权重输出**已发生的用途**只有对照评测 | `smoke24.row.json` note 内印 `'d11': '仅对照评测推理'`，model_dir 指向 0.8B 快照 | `release/sys1/eval/baselines/startlux/smoke24.row.json` | ✅ |
| 兑现点已在配置里配好 | `teacher_model_id: StartLuxAI/StartLux-Decision-4B`（**无** `#scaffold-cpu` 后缀） | `release/production/configs/gate08b.yaml:24`、`scaling_06b.yaml:25`、`sft.py:150` | ✅ |
| 学生基座许可 | `license: apache-2.0` | `…/Qwen--Qwen3.5-0.8B/snapshots/master/README.md:3` | ✅ |

> **推演的时间性**：以下 §1.2 回答的是"C3 之后会发生什么"，不是"现在已经违规"。这一点必须写在结论里——
> 仓内当前所有蒸馏训练输入来自自家随机权重替身，权利问题尚未落地。

### 1.2 论证链（五环节，逐环节标三态与出处）

**环节 A｜CC 许可约束的对象是什么。**
`LICENSE` §1(f)：*"Licensed Material means the artistic or literary work, database, or other material **to which the Licensor applied this Public License**."* 本许可被施加的对象是权重（NOTICE 首段与卡面 :141 一致），推理代码另按 Apache-2.0 授出。
§1(g)：*"Licensed Rights … limited to all **Copyright and Similar Rights that apply to Your use of the Licensed Material** and that the Licensor has authority to license."* → 许可的射程是"对权重行使著作权及类似权利"的行为，不是"任何与权重有关的行为"。
**结论 A**：⚠️＋❓——文本上许可管的是权重的复制/改编/再分发（Share/Adapt），取用一次推理结果不在 §2(a)1 列举的四种被授权行为（reproduce/Share/produce·reproduce·Share Adapted Material）之内。

**环节 B｜模型输出是否构成"作品复制"。**
本项目教师输出＝**候选集上的概率向量**（`dist` 列，形如 `{"A":0.12,"B":0.71,…}`，✅ 实测列名与内容来自 `teachers/cache.py:237` 的 `_normalize_dist`），不含任何成段自然语言，不复制教师训练语料的表达性内容。
按 §1(c) "Copyright and Similar Rights"（含 Sui Generis Database Rights）的定义，落在该射程内的必须是受这些权利保护的内容；一个 4 维概率向量几乎不含可受保护的表达。
**结论 B**：❓ 推断（技术前提：概率分布不承载表达性内容）——本项目**当前主路**的输出形态，是全部环节里风险最低的一种。
**反面必须写明**：若改用自然语言伪标（生成解释、改写题面、长答 `<answer>` 之外的成段文字），输出可能含上游语料的表达性片段，环节 B 立即不成立。现状核查：✅ `release/production/teachers/vision.py` 兜底协议只抽 `<answer>` 内字母（模块 docstring 与 `retry_prompt` 路径），未见长文本入库。

**环节 C｜学生权重是否＝ Adapted Material（"衍生"这关能不能过）。**
§1(a)：*"Adapted Material means material … **derived from or based upon the Licensed Material** and in which the Licensed Material is translated, altered, arranged, transformed, or otherwise modified **in a manner requiring permission under the Copyright and Similar Rights held by the Licensor**."*
本项目的学生权重由 Apache-2.0 基座（Qwen3.5-0.8B）自身参数 + LoRA 垫片构成，**没有一步是把 NC 权重拷进学生**（✅ 代码落点：训练只吃 `DistCache` 的分布，`release/production/sft.py:1186` 报告行印"教师权重一律不进训练"；`opd.py:1131` 同口径"教师份额只从离线缓存取，训练循环里不现算"）。故学生权重不是"from or based upon" NC 权重的表达内容，其形成方式也"不需在许可人著作权下取得许可"。
**结论 C**：❓ 推断，且是**全链最薄的一环**——"蒸馏出的权重是否衍生自教师"在部分上游许可里被明文覆盖（本仓 `refs/` 即有此类条款的实例），而 CC BY-NC 4.0 全文未提模型输出/蒸馏（§1.1 "Produced Material" 0 命中）。文本沉默既可读作"不管"，也可读作"未授予"，本会话不把它写成定论。
**加重项（NOTICE 新事实）**：`NOTICE` 声明"权重是某 Apache 2.0 模型（Copyright 2026 Alibaba Cloud）的修改版"→ NC 层其实叠在一个 Apache 底座之上；若"学生是否衍生"存在争议，争议对象还包括 Alibaba 的 Apache 2.0（其自身不限制商用与衍生，只需保留 NOTICE）。这**降低**了整体风险，但也意味着对外分发时的 NOTICE/署名义务来源不止一处。

**环节 D｜NC 条款对"用输出训练"是否适用。**
§1(i)：*"NonCommercial means **not primarily intended for or directed towards** commercial advantage or monetary compensation."* 判点是"行为的意图与指向"，不是"产物的下游用途"。
本项目的用途定位：学习/科研，无收益、无对外商用（⚠️ 文档承诺：`scratch/PRODUCTION.md` 全篇为课程/论文目标；本仓从未声明商用）。→ 训练这一步**本身**满足 NC。
但 §2(a)(5)(b) 与 §3(a) 的适用条件是 "**If You Share the Licensed Material**"：本仓从未 Share 教师权重（D11 黑名单：任何形式权重再发布，design.md:171；✅ 分发面代码落点 `release/sys1/eval/baselines/table.py:197` 的 D11 脚注、`teachers/__init__.py:8`）→ **文本上 §3(a) 署名义务未被触发**。
**结论 D**：❓——非商业定位下"用输出训练"大概率在许可之内；署名的真正风险不在训练，而在环节 E。

**环节 E｜本仓自己放大的风险：MIT 对外授权 vs 教师的 NC。**
✅ 实测：`LICENSE:1-3` 本仓对外发 **MIT**（Copyright (c) 2026 SpellingDragon），即授予任何人商用权利；⚠️ 而教师侧 `README.md:142` 明示"Commercial use requires a separate license from StartLux Labs"。
于是：若环节 C 的"学生不是 Adapted Material"这一前提被推翻，本仓的 MIT 声明就等于**在替第三方素材做超出该素材许可的授权**——这个矛盾是本仓自己写下的，与教师无关，D11 的"合规落点"（design.md:172）也没覆盖这一层。
**这是本文最有价值的自纠**：风险不在"我们用了 NC 教师"，而在"我们以 MIT 承诺了无限制商用，同时学生可能带着 NC 教师的知识"。

### 1.3 反方风险清单（须重答的场景，按触发难度排序）

1. **第三方引用或商用学生产物**（MIT 已自动放行，无需本仓同意）→ 若届时才追问 NC，本仓无答辩材料（评审 §2.5.2 末句）。
2. **C3 真 4B 产标入库**：`gate08b.yaml:24` 现值即真教师 model_id，一步即兑现。
3. **改用自然语言伪标 / 把教师输出当文本语料**（环节 B 直接失效）。
4. **越出 D11 白名单**：真权重一旦用于训练初始化/基座/微调起点（黑名单，design.md:171），则不再是"取用输出"的解释问题，而是复制权重本身。
5. **NOTICE/License 存在仓外版本差异**：本仓快照的 LICENSE 是 19,347 字节标准 BY-NC 全文，但快照 `revision=master` 未 pin commit；上游若换条款，本文全部环节作废（⚠️ 待核：`grep -rn "PIN_\|revision" release/sys1/eval/registry.py:66-70` 只 pin 了数据集 sha，模型快照走 master 分支名）。

### 1.4 风险等级与缓解措施

- **等级（现态）**：**低**。三条支撑事实：真教师输出 0 行入库（✅）；输出形态为概率向量（✅）；从未分发教师权重（⚠️ 文档承诺＋✅ 代码落点）。
- **等级（C3 后）**：**中低**，且系于环节 C 这一未证前提。
- **升级触发**：对外分发/商用（中→高）、自然语言伪标（中→高）、真权重入训练（高，且直接违 D11）。
- **缓解措施（本文只登记，不改他人写面）**：
  1. **D11 白名单边界的机器化**（现状：家规靠注释与 run 记录，✅ 已核落点：`teachers/__init__.py:8`、`sft.py:1186`、`opd.py:1131`、`run_baseline.py:224`、`table.py:197`、缓存键分家 `cache.py:10`）。缺口：无"训练输入 model_id 扫描"的守卫（对比 §训测分离有 `test_train_axis_guard.py`）。建议守卫：训练 run 的 `config.teacher_model_id` 与 `teacher_cache` 内 model_id 集合若出现**非白名单来源**即 fail。
  2. **DoD ⑧ 自查（产出物零 StartLux 权重血缘）**：⚠️ 该自查目前是**未勾任务**——`openspec/changes/teacher-p2-production-full/design.md:102` 列 DoD ⑧，`tasks.md:68` 仍是 `- [ ] F3 …（含 D11 版权自查：产出物零 StartLux 权量）`。可执行口径（本文给出，未实跑）：对学生 ckpt 做逐张量 sha 对比教师分片＋对训练输入做 model_id 扫描，两条命令都写进 run notes。
  3. **署名要件低成本预置**：§3(a) 现未触发，但一旦分发产物需署名，应保留 creator/copyright notice/license 链接/修改声明（`LICENSE:227-252` 的 i–v 项）。建议产出物目录附 `NOTICE` 副本（教师卡与 NOTICE 已在仓内，复制成本为零）。
  4. **分发面口径**：⚠️ `README.md:57` 已写"蒸馏产物权利推演待补，见 P2 合规论证项"；本文补齐推演后，README 是否要加"学生产物商用需自行评估 NC 教师血缘"一句，属门面改动，**交编排者裁决**（本文不改 README）。

### 1.5 本节待核验项（禁编造，逐条给核验方法）

| 待核验 | 现状 | 核验方法 |
|---|---|---|
| CC BY-NC 4.0 **官方最新版**是否含模型专门化条款（"Produced Material"/"Specialized License"模块） | 仓内快照全文 0 命中（✅）；官方站未取（❌ 本会话无抓取工具） | 打开 `https://creativecommons.org/licenses/by-nc/4.0/`（URL 出处＝卡面 `README.md:141` 自引）与 `…/licenses/versions/`，检索 `Produced Material`；存档入 `openspec/changes/*/evidence/`，标 ⚠️ 文档承诺＋访问日期 |
| 教师上游仓库的 LICENSE/NOTICE 是否与快照一致（含是否被替换） | 未比对（快照走 `master` 名，未 pin sha） | `ls` 快照 → 与 `https://huggingface.co/startlux-models/StartLux-Decision-4B/blob/main/{LICENSE,NOTICE}`（URL 出处＝卡面 `README.md:143`）逐字节 diff，并把 sha 记入 registry pin |
| "蒸馏产物不构成 Adapted Material"的法律确定性 | ❓ 推断 | 须执业法律意见；本文不替代 |

---

## 2. GLM-5.3-Flash API 服务条款（ToS）

### 2.1 先破同义反复："红线合规"≠ 服务条款（✅ 全部实测）

追"红线"定义的唯一出处：

```bash
sed -n '360,366p' scratch/PRODUCTION.md      # :366 = §11 第 4 条
grep -rn "红线合规" --include="*.md" openspec release scratch   # 命中 5 处，全在 p2 计划文书
```

- `scratch/PRODUCTION.md:366` §11-4 原文：*"**端侧可服务**：正式版本体仍须能在自有硬件部署；**外部仅可作教师（离线产标/蒸馏），推理链路不得依赖第三方在线 API**。"*
  → 这是**架构纪律**（约束"推理链能不能依赖在线 API"），主语是"我们"，不是"服务方"。
- 使用"红线合规"字样的全部位置：`teacher-p2-production-full/design.md:109,129`；`changes/p2-02-teacher-adapters/design.md:6,9`；`p2-02-teacher-adapters/proposal.md:9`。每处括号内的解释都指向 §11-4 的"离线产标/零在线 API"，**无一处引用或转述 Z.ai/智谱条款**。
- 全仓 ToS 分析检索（发布版命令）：
  ```bash
  grep -rniE "服务条款|terms of service|用户协议|privacy policy|数据留存" \
    --include="*.md" --include="*.py" --include="*.yaml" \
    release/production release/sys1 release/tests scratch/sys1 scratch/*.md \
    openspec/changes/teacher-p2-production-full AGENTS.md README.md | wc -l    # → 1
  ```
  唯一命中是 `README.md:22` 里"GLM API 服务条款分析都还没成文"这句**自述缺口**。→ **除"尚未成文"这句话本身，全仓零 ToS 内容**（✅）。
- **判定**：评审 §2.5.3 成立。"红线合规"在本文档体系里改称 **"符合 §11-4 架构纪律"**；对外合规结论一律标注为**未核验**，不得再由 D11/§11-4 代答。

### 2.2 条款核验清单（要查什么——问题清单，不是答案清单）

| # | 核验点 | 为什么本项目必须查（事实依据） | 状态 |
|---|---|---|---|
| 1 | **是否限制"用输出训练模型"**，尤其"训练竞争性模型" | 本项目正在用输出训练一个决策模型（视觉伪标），且叙事是"击败其 0.8B 家族"（`p2-02-teacher-adapters/proposal.md:9`）——若条款含竞争性限制，这条叙事本身即风险点 | **待核验** |
| 2 | 输出可否用于**蒸馏/构建其他模型**（部分条款要求注明"数据来源"或禁止直接复用为训练集） | `vision.py` 主路把 0–10 信心折成 soft 分布直接进 `dist` 列当训练监督（`cache.py:237`） | 待核验 |
| 3 | **输入数据的出域与再分发**：本项目上传自有图片＋题面 | `teachers/vision.py` 走 base64 图像传输（✅ 模块 import `base64`、`image_digest`）；题目源自 mmbench-cn/longbench 等上游数据集，**上游数据许可另有约束**（属第二层问题，须一并核） | 待核验（两层） |
| 4 | **数据留存与"用于改进服务"条款**：输入是否被平台留存/用于其模型训练 | 本项目按 model_id+prompt+keys 落盘可复现（✅ `cache.py:10`），但对侧留存未知 | 待核验 |
| 5 | **批量/自动化调用、速率与配额** | 3–6k 次一次性产标（`p2-02 design.md:9`），属批量调用形态 | 待核验 |
| 6 | **商用与转售限制**（若产物商用） | 学生产物对外发 MIT（`LICENSE:1-3`） | 待核验 |
| 7 | 账号与密钥义务（不外泄、不共享） | ✅ 现符合：key 只取环境变量 `ZAI_API_KEY`，两批 run notes 均记"未落任何文件与日志" | 实测已符（本仓侧） |
| 8 | 地域/跨境与合规声明 | 服务方为境内厂商、endpoint 为 `open.bigmodel.cn`（✅ `vision.py:56`） | 待核验 |

### 2.3 核验状态：未能取到原文（如实标注，禁编造）

- ❌ **本会话未取得任何条款原文**。原因如实：本执行代理的工具面无网页抓取能力（无 WebFetch），且按纪律不得用 `curl/wget` 下载网络资源（`Bash` 约束第 4 条）。因此本文**不引用、不转述、不概括**任何 GLM/Z.ai 条款文字——避免把"主流大模型 API 普遍限制竞争性训练"这类常识写成"GLM 条款如此"。评审 §2.5.3 的该表述属 ❓ 一般性判断，本文不升级为事实。
- 核验入口（域名与命名均来自仓内实测，非猜测条款）：
  - `https://open.bigmodel.cn/api/paas/v4/chat/completions` —— 代码常量 `release/production/teachers/vision.py:56`（`DEFAULT_BASE_URL`）→ 协议文本应在同域的用户中心/法务页，**具体 URL 待人工在站内定位**。
  - `ZAI_API_KEY` —— 环境变量名（`vision.py:57` `API_KEY_ENV`，`p2-02 proposal.md:9` 记"已在位，长度 49"）→ 提示账号可能开在 z.ai 站点而非 bigmodel 站；**两站协议文本可能不同，须分别核验**（这是本文能给出的最明确的一条可执行提醒）。
- 核验方法（可交给执行者照做）：
  1. 取账号实际注册站点的《服务协议》《隐私政策》《可接受使用政策》三页原文，**存 markdown/PDF 入 `openspec/changes/<变更>/evidence/`**，文件名带访问日期；
  2. 逐条回答 §2.2 的 8 项，每项**引用原文行**（不是本文的问题措辞），标注 ⚠️ 文档承诺；
  3. 把结论写入本文件 §2.2 状态列（追加"已核验＋日期＋原文摘录"，勘误追加不改写，见 `docs/stats_plan.md` §3.4 同源纪律）；
  4. 若条款含"禁止用输出训练竞争性模型"→ 立即回到 §2.5 的降级路径，并升级为阻断项重开评审。

### 2.4 执行侧现状（实测账目：审查滞后于就绪，评审 §2.5.3 后半句成立）

| 账目 | 现值 | 出处 | 三态 |
|---|---|---|---|
| 已发生调用 | 首波 **14 次**（10 请求，产标 2 行，均延 2.714s，prompt 1470/completion 887 tokens）；次波 **11 次**（产标 10 行，均延 4.032s，reasoning 688）——合计 **25 次 / 12 行** | run `1005-p202-teacher-adapters-b3-realpath-b570/notes.md`、`…-b570-2/notes.md:8`；落包实测 `teacher_cache/shard-*-{2,10}.parquet`（model_id=glm-5.3-flash，source=api_label） | ✅ |
| 试点质量 | 首波判分口径曾错（把颜色名首字母当候选字母），离线复核实算 **10/10 押对**，代码已修 | `…-b570-2/notes.md:15`「更正」行 | ✅ |
| 全量规划 | **3–6k 次调用，人民币几十元级，一次产完入 parquet** | `p2-02-teacher-adapters/design.md:9`；台账行 `teacher-p2-production-full/tasks.md:13`："GLM API｜空闲(配额内)｜已产 10 行视觉 pack；3–6k 全量产标待 C3" | ⚠️ |
| 生产任务未开 | `tasks.md:53` `- [ ] C3 数据上云 + 教师伪标生产…GLM API 视觉伪标本地/任意端产包上云` | 同上 | ✅（未勾状态实测） |
| 成本凭证 | "几十元"是设计估算，**仓内无费用凭证**（README:22 已自陈） | ⚠️ |
| 训练侧误食风险 | **无**：缓存键含 model_id，文本训练查询（StartLux model_id）不会命中 GLM 行；但正式档 `teacher_cache: bench/teacher_cache` 根目录已含这 12 行 GLM 包，视觉伪标查询会命中 | `cache.py:10,126`（`glob("*.parquet")` 非递归、按目录合并）；`gate08b.yaml:25` | ✅ |

### 2.5 处置建议（合规审查滞后于执行就绪度）

1. **闸口前置（最低成本、最高收益）**：把"ToS 核验并归档原文"设成 **C3 全量产标的开闸条件**，与既有熔断线并列（成本闸口已在 `design.md:183` "熔断线 ¥600"有先例，合规闸口照此形态加一条即可）。本文不新建任务、不排产，仅登记为待裁决项。
2. **降级路径已存在，不是新方案**：⚠️ `p2-02-teacher-adapters/design.md:9` 原文即写"本地 Qwen3.5-9B 降备选（单跑纯推理 ~18GB 可行时）"，`proposal.md:9` 亦备 Qwen3.5-4B——若 ToS 核验不过，走既有备选，不改架构。
3. **规模事实对本案有利，须如实记录**：已发生的 25 次调用/12 行属通路试点，量级远低于全量；若核验不过，删除 12 行包并停用即可回滚（负结果入库不删 run 档：`.agent/rules/common.md:101`＋`PRODUCTION` §9.1）。
4. **措辞纠偏（登记，禁改区）**：`design.md:109,129`、`p2-02 design.md:6,9`、`p2-02 proposal.md:9` 共 5 处的"红线合规"应改称"符合 §11-4 架构纪律（ToS 待核验）"。这些文件属他人变更计划，本文**不代改**，交编排者裁决。
5. **等级与理由**：**中**。理由不是"推断违规"，而是"**未核验 + 执行侧已就绪并已小步执行**"这个组合本身构成风险敞口；核验完成后应回落到 低（若条款无限制）或升为 阻断（若有限制且计划继续）。

---

## 3. 本文未核验清单（诚实边界，汇总）

| 项 | 状态 | 备注 |
|---|---|---|
| GLM/Z.ai 服务条款原文任何一句 | **未取得** | 本代理无抓取工具；§2.2 全部 8 项＝待核验 |
| CC BY-NC 4.0 **官方站当前文本** | 未取得 | 但仓内快照含 19,347 字节全文（✅）——本文推演以该本地副本为文本依据，须与上游比对（§1.5） |
| "输出不受著作权保护/学生非衍生品"的法律确定性 | ❓ 推断 | 需执业意见，本文不替代 |
| 上游数据集（mmbench-cn/longbench/cmmlu/clue）对"用于训练"的许可 | 未核 | §2.2 第 3 项的第二层问题，须单列 |
| DoD ⑧ 权重血缘自查是否已跑 | **未跑**（`tasks.md:68` 未勾） | §1.4 措施 2 给了可执行口径 |
| 教师快照 sha 是否 pin | 未 pin（数据集 pin 见 `registry.py:66-70`，模型走 `master`） | 影响 §1.3 第 5 项 |

**复放凭据**：本文全部 ✅ 结论由以下命令产生——§1.1 的两段（模型卡/CC 全文/教师缓存 model_id 分布）、§2.1 的两段（`PRODUCTION.md:360-366`、"红线合规"与 ToS 关键词检索）、§2.4（两条 B3 run notes + `teacher_cache` 行数）。逐条命令均已写在正文对应小节内，可整段重放。
