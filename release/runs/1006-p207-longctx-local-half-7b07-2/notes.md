# run: 1006-p207-longctx-local-half-7b07-2

- 假设：证据段的编号是整段 prompt 编号的严格前缀，故 past 可按 state 复用；复用生效的硬凭据是符号计数（第 2 问起=问题段长度），不是墙钟时间；bf16 复用漂移可压在相对阈值 0.02 内
- 观察：A1 真 token 口径 8 行全部构建（8K/32K 全出 + 128K/256K 各 1 题），档位相对偏差上限 0.0009，副本落 $REPO_ROOT/release/bench/eval_data/assembled/needle-synthetic/needle-envelopes.jsonl；针位表与 registry 骨架同源
- 观察：B1b 复用路总符号 4657 vs 三问整段重算 6380（省 1723）；第 2/3 问 prefix_tokens_fed=0 且 suffix_tokens_fed=suffix_len；冷启一题 34.36s，past 225.9 MiB（state=2065 token）
- 观察：B1c 真 bf16 前向 max_abs_drift=3.846e-01、分数尺度 24.495 → 相对漂移 1.570e-02（阈值 0.02）passed=True，argmax_same=True
- 观察：B2 跨请求命中 3/3，命中题全部 suffix_only=True；缓存记账 {"hits": 7, "misses": 1, "hit_ratio": 0.875, "items": 1, "bytes_used": 236830720, "bytes_mib": 225.9, "stores": 1, "evictions": 0, "oversized": 0, "max_bytes": 8589934592, "max_items": 0, "memory_ledger": "cpu-rss/past-bytes，非 NPU 显存账"}
- 观察：C1 带状路单块峰值 782,336 字节 vs 全注意力 67,108,864 字节（比值 0.011658）；128K 档 KV 解析账 3,584.0 MiB → 3.5 MiB；窗上限 cap=256 可夹住 28/28 层
- 口径边界：以上全为 CPU + 0.6B 替身；A2 召回曲线与 1M/NPU 显存实测未做（待云端长窗真档）


> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
结论：prefix_cache 三级件与滑窗长文路径在本地半场成立：同 state 三问复用只付问题段符号（4657/6380 token，省 1723，第 2 问起 prefix_tokens_fed=0），跨请求命中 3/3，真 bf16 相对漂移 1.570e-02 在 0.02 阈值内且 argmax 不换人；滑窗带状路单块峰值降到全注意力的 0.011658 倍且与掩码路逐位同口径（kernel 路差 0.0），窗上限开关能把 28 层全夹住——D7 三件套'滑窗封顶'的第一件基座到位。A2 召回曲线与 NPU/1M 显存实测按派单留待云端。
