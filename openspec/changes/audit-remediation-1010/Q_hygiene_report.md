# Q 波交付 · audit-remediation-1010 P1（R-P1-2 + R-P1-3）

代理：Q（卫生批 + R 编号口径）。基准 HEAD：`main` 工作区未提交态（禁 git commit，`git rm --cached` 已用）。
执行日期：2026-10-10。

---

## 一、R-P1-2 R 编号口径统一

**实测源（唯一真源 = skill）**

| 文件 | 实存 R 编号 | 证据命令 |
|---|---|---|
| `.agent/skills/openspec-multilevel-planning/SKILL.md` | R1–R18（84–106 行逐条定义） | `grep -oE "R[0-9]+" .agent/skills/openspec-multilevel-planning/SKILL.md \| sort -u` |
| `.agent/skills/openspec-multilevel-planning/apply-orchestration.md` | R6–R9、R15–R20（130–143 行"执行期反模式"详版） | `grep -nE "R[0-9]+" .../apply-orchestration.md` |
| **并集** | **R1..R20，连续无缺号**（`SKILL.md:99` 定义 R18，`apply-orchestration.md:139-140` 补 R20/R19） | `grep -rhoE "R[0-9]+" .agent/skills/ \| sed 's/R//' \| sort -n -u \| tail -1` → `20` |

**改动**

| 文件:行 | 原文 | 现文 |
|---|---|---|
| `AGENTS.md:37` | `反模式 R1–R14` | `反模式 R 系列（编号以本 skill 为唯一源，引用处不写死区间）` |
| `.agent/README.md:10` | `反模式 R1–R14` | `反模式 R 系列（编号以 SKILL.md 为唯一源，此处不写死区间）` |

采"指源不写死"而非"更新为 R1–R20"：写死区间会随下一条反模式回写再度失实（本次三处三个数的根因即如此），且 `README.md:29` 已声明"本页不写死区间……`AGENTS.md` 的旧区间随 audit-remediation-1010 的 P1 卫生批对齐"，指源式与门面口径一致。

**验证（三处写死区间零命中）**

```
$ for f in AGENTS.md .agent/README.md README.md; do printf "%s → %s 处\n" "$f" \
    "$(grep -oE "R[0-9]+ ?[-–] ?R[0-9]+" "$f" | wc -l | tr -d ' ')"; done
AGENTS.md → 0 处
.agent/README.md → 0 处
README.md → 0 处
```

`git diff --stat AGENTS.md .agent/README.md` = `2 files changed, 2 insertions(+), 2 deletions(-)`（各 1 行，无重排、无空行扰动）。

**顺带发现（白名单外，未动，待裁决）**

- `openspec/changes/audit-remediation-1010/design.md:29` 写"红项归因后转正式门（**R23** 三分归因，不静默放行）"——skill 实至 **R20**，**R23 不存在**；skill 中对应的制度是 `apply-orchestration.md:81 §归因与重派`（"归因三型"），它**不是 R 编号条目**。建议改法：把"（R23 三分归因…）"改为"（见 apply-orchestration §归因与重派，归因三型）"。design.md 不在 Q 白名单（N/O/P 写面之外的一级件，归编排者），故只上报。
- `scratch/tools/check_comments.py:15` 的 "R1-R3" 是**注释规范自有编号**（GUIDE §6.1 文件头三件套/白话段落/注释密度），与 skill 反模式 R 系列同名不同源，非失实，不动。
- `openspec/changes/audit-remediation-1010/evidence/eval_report.md`（162/474/616 行）与根目录未跟踪件 `system_one_study_eval.agent.final.md` 仍含 "R1–R14 / R1–R18" 字样：前者是第三方评审原文（design D7"只存不改"），后者是未入库的工作稿——均不属于门面/规范声明面，未动。

---

## 二、R-P1-3 工程卫生批（5 项）

### 1. egg-info 出库（10 文件取消跟踪 + gitignore）

```
before: git ls-files | grep -c egg-info   → 10
执行:   git rm -r --cached release/sys1.egg-info scratch/sys1.egg-info   （--cached 只动索引，磁盘未删）
after:  git ls-files | grep -c egg-info   → 0
```

- 磁盘完好：`ls -d release/sys1.egg-info scratch/sys1.egg-info` → 两目录仍在（editable 安装不破）。
- `.gitignore` 追加（原 30 行 → 32 行）：`# 构建产物：pip install -e 生成的 egg-info 元数据（audit-remediation-1010 R-P1-3 出库，不入库）` + `*.egg-info/`。
- 生效自证：`git check-ignore -v release/sys1.egg-info/PKG-INFO scratch/sys1.egg-info/requires.txt` → 两条均命中 `.gitignore:32:*.egg-info/`；`git status --short` 里 `?? …egg-info/` 已消失，只剩预期的 10 条 `D `（暂存的取消跟踪，待编排者 commit）。

### 2. orchestration spec Purpose 去 TBD

`openspec/specs/orchestration/spec.md:4`：
`TBD - created by archiving change teacher-p1-scratch-mps. Update Purpose after archive.`
→ `定义本仓多级 OpenSpec 变更的编排契约：一级只司编排与裁决、实现义务下沉子变更，并约束目录自包含、波次合并前置与变更级 DoD 收尾门，防止编排层与实现层互相污染。`

```
验证: grep -rn "TBD" openspec/specs/ | wc -l → 0
      git diff --stat openspec/specs/orchestration/spec.md → 1 file changed, 1 insertion(+), 1 deletion(-)
```

目的句按该 spec 实有 4 条 Requirement（二级子变更结构 / 一级只编排不实现 / 波次合并前置 / 变更级 DoD 收尾）自拟，未引入新主张。

### 3. refs/clone.sh 六 clone 钉 rev

网络可达，六条 sha 全部实测于 `git ls-remote <url> HEAD`（2026-10-10），无编造：

| 仓 | pinned sha（40 位） | 与远端 HEAD 复验 |
|---|---|---|
| llms-from-scratch-cn | `0c6bdb805bfc7b5bd5f23b641daf81d126e16c25` | MATCH |
| tilelang | `da9471ac9809230079c4b480cac6551ff8343325` | MATCH |
| TileKernels | `66258df6175d2f630ffecb04c5ab66bff8a2ae6a` | MATCH |
| laya | `68804629e8ccd9d616d48a40e87de9fabbeae069` | MATCH |
| Naive-N0.5-Flash | `3551de41a35f1470bc37831f11388a821c99162f` | MATCH |
| jev-cookbook | `f3c6c2b2b47886776b5c9a16e60602d7e466655a` | MATCH |

```
复跑（逐条比对文件内 sha 与实时远端 HEAD）:
$ grep -E "pinned 2026" refs/clone.sh | awk '{print $3, $4}' | while read u f; do
    r=$(git ls-remote "$u" HEAD | awk '{print $1}')
    [ "$r" = "$f" ] && [ ${#f} -eq 40 ] && echo "MATCH  ${f:0:12}  $u" || echo "MISMATCH $u file=$f remote=$r"; done
→ 6 行全部 MATCH
语法门: bash -n refs/clone.sh → OK
```

- 写法：sha 作为 `clone()` 第三位置参（函数原已支持 `[rev]` 并 `git checkout`），行尾注释 `# pinned 2026-10-10`；头部另加一行说明"pinned sha 实测于 2026-10-10：git ls-remote <url> HEAD（重锁=重跑该命令换新 sha，并同步更新行尾日期）"。
- `StartLux-Decision` 那条本就是注释态（未核实到公开仓，由课程方供本地包），无可钉对象，原样保留。
- **未做的验证**：`bash refs/clone.sh` 实跑（会真下载 6 个仓、且 `refs/` 目前只有 clone.sh，跑完即污染磁盘并吃掉大量带宽）。钉 rev 的通路是既有 `git checkout "$rev"` 分支，本次只新增入参；sha 为远端 HEAD 可达提交，风险低。若编排者要端到端自证，可在任意一台干净机上跑一次并看每行输出的 `OK <name> @ <短 sha>` 是否等于钉住的 sha。

### 4. run-notes "待填写" 清理（严格规则版）

规则与结果：

| 规则 | 判据 | 文件数 | 占位行 |
|---|---|---|---|
| ① 同前缀已有真实行 → 只删占位 | 该前缀（假设/观察/结论）在正文另有真实行 | 70 | 70（全部是 `结论：待填写`） |
| ② 无任何真实行的空壳 → 三行占位改单行显式标注 | 三前缀均无真实行 **且** `metrics.jsonl` 确为空 | 3 | 9 → 3 行标注 |
| ② 例外（措辞不成立）→ 原样保留并入豁免 | 三前缀均无真实行，但 `metrics.jsonl` 非空（1–6 行） | 4 | 12（未动） |
| ③ 拿不准 → 原样保留并入豁免 | 有真实 `结论` 但 `假设/观察` 从未记录（删占位即抹掉"三行必填未兑现"的痕迹，非 Q 可编造） | 58 | 116（未动） |

```
before: grep -rl "待填写" scratch/runs release/runs | wc -l   → 77     （出现次数 207：结论 77 / 假设 65 / 观察 65）
after : grep -rl "待填写" scratch/runs release/runs | wc -l   → 62     （出现次数 128：假设 62 / 观察 62 / 结论 4）
删除数: 207 - 128 = 79 = 70(①) + 9(②)
```

不编造/不改历史的机器自证：

```
$ git diff --numstat -- 'scratch/runs/**/notes.md' 'release/runs/**/notes.md'
      → files=73  added=3  deleted=79
$ git diff -- '…notes.md' | grep "^-" | grep -v "^---" | grep -vc "待填写"
      → 0        # 被删的每一行都含"待填写"，即：任何真实内容行一字未动
$ git diff -- '…notes.md' | grep "^+" | grep -v "^+++" | sort -u
      → 仅 1 种：+- （空跑壳：无假设/观察记录，metrics 空，如实保留——audit-remediation 显式豁免）
```

metrics/config/system 文件一律未碰（`git status` 无这些路径的改动）；4 个 metrics 非空的空壳整体未触碰（`git diff --name-only` 对它们 = 0 行）。

**②已标注的 3 个空壳**（notes 三行全占位 + metrics.jsonl 0 行，实测）：
`release/runs/1006-p2-07-a2-needle-curve-e6c8`、`release/runs/1006-p2-07-a2-needle-curve-v3-e4ba`、`release/runs/1006-p2-08-a1-multimodal-probe-e5a2`。
（README:29 点名的第三个 needle 空跑壳 `…-b46b` 不在此列——它的 notes 有真实 `结论：A2 召回曲线…` 行，按①删掉了 `结论` 占位，假设/观察 归入③豁免。metrics 亦为 0 行。）

**豁免清单（62 行 = after 文件数，逐文件）**

| # | run 目录 | 残留占位前缀 | metrics.jsonl 行数 | 豁免依据 |
|---:|---|---|---:|---|
| 1 | `release/runs/1006-p2-07-a2-needle-curve-v3-e4ba-2` | 假设、观察、结论 | 6 | ② 判据不成立（空壳但 metrics 非空）——待原域补写，不得由 Q 编造 |
| 2 | `release/runs/1006-p207-longctx-local-half-7b07` | 假设、观察、结论 | 1 | ② 判据不成立（空壳但 metrics 非空）——待原域补写，不得由 Q 编造 |
| 3 | `release/runs/1010-p2-11-dev-rl-qwen3-0-6b-torch-bf64` | 假设、观察、结论 | 6 | ② 判据不成立（空壳但 metrics 非空）——待原域补写，不得由 Q 编造 |
| 4 | `scratch/runs/1004-s1-mps-main-eca8` | 假设、观察、结论 | 2 | ② 判据不成立（空壳但 metrics 非空）——待原域补写，不得由 Q 编造 |
| 5 | `release/runs/1005-probe-08b-lora-mps-feasibility-5a64` | 假设、观察 | 5 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 6 | `release/runs/1005-probe-p2-external-assets-sweep-7e80` | 假设、观察 | 8 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 7 | `release/runs/1006-p2-07-a2-needle-curve-b46b` | 假设、观察 | 0 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 8 | `release/runs/1006-p2-07-a2-needle-curve-v3-1035` | 假设、观察 | 7 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 9 | `release/runs/1006-p2-08-a1-multimodal-probe-0224` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 10 | `release/runs/1007-p2-13-b1-910b-port-attempt-7793` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 11 | `release/runs/1007-p2-13-c1-ascend-probe-de15` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 12 | `release/runs/1008-p2-13-p02-bundle-ready-fda9` | 假设、观察 | 3 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 13 | `release/runs/1008-p2-13-p02-first-attempt-ee05` | 假设、观察 | 3 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 14 | `release/runs/1008-p2-13-p02-ocard-fe5f` | 假设、观察 | 4 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 15 | `release/runs/1008-p2-13-p02l-e2e-arc-d5f2` | 假设、观察 | 4 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 16 | `release/runs/1008-p2-13-p02l-life-line-9ba4` | 假设、观察 | 3 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 17 | `release/runs/1008-p2-13-p02l-local-env-a545` | 假设、观察 | 3 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 18 | `release/runs/1008-p2-13-upstream-issue-3448-be5c` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 19 | `release/runs/1009-p2-13-a-case-recon-e0db` | 假设、观察 | 4 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 20 | `release/runs/1009-p2-13-a3-gemm-l1-pass-d142` | 假设、观察 | 3 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 21 | `release/runs/1009-p2-13-a4-gemm-full-chain-5fd3` | 假设、观察 | 3 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 22 | `release/runs/1009-p2-13-issue-3448-engaged-f114` | 假设、观察 | 3 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 23 | `release/runs/1009-p2-13-p11-merge-a5a5` | 假设、观察 | 4 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 24 | `release/runs/1009-p2-13-p11b-dw-rope-attnsw-6223` | 假设、观察 | 4 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 25 | `release/runs/1010-p2-13-oncard-wave1-cc3d` | 假设、观察 | 4 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 26 | `scratch/runs/1004-bench-infer-e2e-mps-ce64` | 假设、观察 | 1 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 27 | `scratch/runs/1004-bench-kernel-autocograd-30step-28ba` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 28 | `scratch/runs/1004-bench-tilelang-vs-torch-mps-1a18` | 假设、观察 | 6 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 29 | `scratch/runs/1004-eval-report-1f94` | 假设、观察 | 1 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 30 | `scratch/runs/1004-p1-dod-verification-5ce3` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 31 | `scratch/runs/1004-s1-mps-main-2081` | 假设、观察 | 3 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 32 | `scratch/runs/1004-s1-mps-main-eca8-2` | 假设、观察 | 3 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 33 | `scratch/runs/1004-s1-smoke-tiny-7c3f` | 假设、观察 | 4 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 34 | `scratch/runs/1004-s1-smoke-tiny-ca2f` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 35 | `scratch/runs/1004-s1-smoke-tiny-ca2f-2` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 36 | `scratch/runs/1004-s1-smoke-tiny-ca2f-3` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 37 | `scratch/runs/1004-s1-smoke-tiny-ca2f-4` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 38 | `scratch/runs/1004-s1-smoke-tiny-ca2f-5` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 39 | `scratch/runs/1004-s1-smoke-tiny-ca2f-6` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 40 | `scratch/runs/1004-s2-decision-sft-17ac` | 假设、观察 | 7 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 41 | `scratch/runs/1004-s2-decision-sft-4978` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 42 | `scratch/runs/1004-s2-decision-sft-739e` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 43 | `scratch/runs/1004-s2-decision-sft-c0e7` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 44 | `scratch/runs/1004-s2-decision-sft-c6e1` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 45 | `scratch/runs/1004-s2-decision-sft-d1ec` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 46 | `scratch/runs/1004-s2-decision-sft-d828` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 47 | `scratch/runs/1004-s3-calibrate-5a9f` | 假设、观察 | 1 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 48 | `scratch/runs/1004-s3-calibrate-5e08` | 假设、观察 | 1 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 49 | `scratch/runs/1004-s3-calibrate-6d33` | 假设、观察 | 1 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 50 | `scratch/runs/1004-s3-calibrate-a3a2` | 假设、观察 | 1 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 51 | `scratch/runs/1004-s3-calibrate-acc2` | 假设、观察 | 1 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 52 | `scratch/runs/1004-s3-calibrate-de2d` | 假设、观察 | 1 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 53 | `scratch/runs/1005-s1-mps-main-8c04` | 假设、观察 | 86 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 54 | `scratch/runs/1005-s1-mps-main-dc8c` | 假设、观察 | 101 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 55 | `scratch/runs/1006-eval-report-aa36` | 假设、观察 | 1 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 56 | `scratch/runs/1006-s2-decision-sft-1e95` | 假设、观察 | 2 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 57 | `scratch/runs/1006-s3-calibrate-962f` | 假设、观察 | 1 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 58 | `scratch/runs/repro-p1/runs/1004-eval-report-225e` | 假设、观察 | 1 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 59 | `scratch/runs/repro-p1/runs/1004-eval-report-5cd4` | 假设、观察 | 1 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 60 | `scratch/runs/repro-p1/runs/1004-eval-report-8e95` | 假设、观察 | 1 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 61 | `scratch/runs/repro-p1/runs/1004-eval-report-8feb` | 假设、观察 | 1 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |
| 62 | `scratch/runs/repro-p1/runs/1004-eval-report-cc84` | 假设、观察 | 1 | ③ 假设/观察从未记录且该 run 非空壳；删占位会抹掉"三行必填未兑现"的证据，故原样保留 |

合计：62 文件 / 128 占位行（= after `grep -rl 待填写` 文件数，满足"清零或豁免表行数"口径）。


### 5. O 波上报的死代码（release/tests/test_backends.py）

- 删 `_isolate_global_state()`（原 37–41 行）：无 `@pytest.fixture` 装饰的孤函数、函数体与 `_clean_compile_cache` 同形、全仓零引用（`grep -Rn "_isolate_global_state" --include=*.py .` 只剩 `scratch/tests/test_backends.py:26`——那边带 autouse 装饰、是真 fixture，且 scratch 冻结/D6 禁改，未动）。
- `import pytest` 原第 14、16 行重复 → 去第 14 行一份，保留与 `torch` 同组的第三方块（不动其它 import 顺序）。
- 改动量：`git diff --stat` = **7 deletions, 0 insertions**（纯删除，无重排、无格式化）。

```
单文件: release/.venv/bin/python -m pytest tests/test_backends.py -q → 15 passed in 1.45s
全量回归: cd release && .venv/bin/python -m pytest tests -q -m "not integration"
        → 326 passed, 3 skipped, 3 deselected, 1 warning in 39.08s（收集期 0 error，无新红）
```

---

## 三、Q 波自报的坑（R15 当场抓到并修）

写 `refs/clone.sh` 时，我把 `Naive-N0.5-Flash` 的 sha 手敲截短成 35 位（`3551de41a35f1470bcf11388a821c99162f`，丢了 `783`）。写后三连（长度分布 `awk '{print length}'` + `bash -n` + 逐条与远端复验）抓到，改为**由 `git ls-remote` 的输出经环境变量直接注入文件**再全量复验，杜绝手抄。若没做这一步，clone.sh 会在 `git checkout` 处 "reference is not a tree" 直接失败——属 R15 的又一次实证。

---

## 四、写面自证与并发观察

```
$ git status --short | awk '{print $2}' | grep -vE "notes\.md$|^(AGENTS\.md|\.agent/README\.md|\.gitignore|refs/clone\.sh|openspec/specs/orchestration/spec\.md|release/tests/test_backends\.py|openspec/changes/audit-remediation-1010/)"
.github/workflows/ci.yml      ← O 代理（R-P1-1）在写，mtime 20:21，Q 未碰
release/pyproject.toml        ← O 代理 `[dev]` 补声明，mtime 20:18，Q 未碰
release/sys1.egg-info/*, scratch/sys1.egg-info/*  ← Q 的 `git rm --cached`（索引删除，非工作区改动）
system_one_study_eval.agent.final.md              ← 会话前既有未跟踪件（?? ），Q 未碰
```

- **禁改面零触碰自证**：`git status --short | grep -E "production/|ascend/"` → 空（`.github/`、`pyproject.toml` 的两处改动归属 O，见上）。
- **R17 实况**：`openspec/changes/audit-remediation-1010/tasks.md` 是 Q 与 O 共写账目。Q 勾 R-P1-2（19:59 段）后，O 在 20:20 勾 R-P1-1；Q 勾 R-P1-3 时先重读全文再定点替换，并写后即验三勾均在位（R-P1-1/R-P1-2/R-P1-3 同时为 `[x]`）。本次未发生丢勾，但该文件按 skill §一"账目单点归属"应归编排者统笔。
- 进度复核：`openspec instructions apply --change audit-remediation-1010 --json` → `progress: {total:12, complete:7, remaining:5}, state: ready`。

## 五、复跑清账（编排者四查用）

```bash
git ls-files | grep -c egg-info                                   # 期望 0
grep -rl "待填写" scratch/runs release/runs | wc -l               # 期望 62（= 豁免表行数）
grep -rc "TBD" openspec/specs/orchestration/spec.md               # 期望 0
grep -c "pinned 2026-10-10" refs/clone.sh                         # 期望 6
bash -n refs/clone.sh                                             # 期望 静默通过
grep -oE "R[0-9]+ ?[-–] ?R[0-9]+" AGENTS.md .agent/README.md README.md | wc -l   # 期望 0
grep -c "import pytest" release/tests/test_backends.py            # 期望 1
grep -c "_isolate_global_state" release/tests/test_backends.py    # 期望 0
(cd release && .venv/bin/python -m pytest tests -q -m "not integration") | tail -1   # 期望 326 passed, 3 skipped
```
