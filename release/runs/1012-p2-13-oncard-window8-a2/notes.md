# 1012 12:42–12:52 · 窗八（A2 预热判决，repo `05e0b10`，450s）——A2 证伪 + H-b 现象不可复现，私有 fork 反向调试宣告止损

## 判决（原始日志在 artifacts/，回读 grep 的 W7- 字面量坑已勘误）
| 臂 | 结果 |
|---|---|
| A0（`SYS1_DISABLE_WARMUP=1`） | `dbeta 8.002e-01` FAIL（对照符合预期，`WARMUP` 行=0） |
| **A2（默认预热）** | **`WARMUP gdn[ascend]|h2k8v8|c8ubh` + `WARMUP gdnb[ascend]|h2k8v8|c2ubz` 确实触发**（机制正常）——但第二次发射 `dbeta` **仍 `8.002e-01` FAIL**，其余五项 `1e-7` PASS |
| gradcheck | 6/8：`gdn_delta FAIL 4.698e+03`、`gdn_conv FAIL 1.431e+07`（数值跨窗漂移＝垃圾态相关，判据诚实） |
| D3 顺序探针 | **`none==dirty==zero==e609a6ab` 逐位同，自判 `ORDER-VERDICT 顺序无关(H-b 证伪)`** |

## 三条结论
1. **A2 预热证伪**：触发了、没修好——其唯一依据（窗四"any-prior-launch ⇒ correct"）**在本 boot 不可复现**（prefill 三态逐位同）。**warmup 已回退**（无独立依据的结构不留，§〇二；回退后 `gdn_asc.py=676c0793…`、`gdn_conv_asc.py=28cc045a…`，host 三腿逐字复原、340 passed）。
2. **`dbeta 8.002e-01` 跨 3 个 boot、4 个窗口、两种变体逐字复现** ⇒ 这不是随机未初始化读，是**确定性错误计算**；窗四的"预热后正确"降级为**不可复现异常**（boot 态相关）。
3. **私有 fork 的 GDN 反向调试止损**：M4 清零、A2 预热、v2 预清零三修皆被真机证伪，机理在生成码之下且 boot 态相关。**而 tilelang-ascend 底座上六件 GDN（含 chunk_o/chunk_h/wy_fast 同数学）真机 `rel=0`**（窗五）——substrate 是对的，错的是我们 fork 的这一层。**GDN 反向的正确性+性能路线改走 TA**（上游 bwd PR 合入前可先在其原语上自实现，判决用我们的 rigs）。

## 工装勘误
本窗回读 grep 因 sed 只换了部分 `W7-` 字面量漏了 `W8-HEAD/WARMUP` 行——**原始日志抢救后以日志为准**（教训：派生脚本要全量审一遍再用于取证窗）。
