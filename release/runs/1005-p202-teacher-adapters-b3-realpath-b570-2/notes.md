# run: 1005-p202-teacher-adapters-b3-realpath-b570-2

- 假设：三教师真实路径各自可跑通：本地 tokenizer/config 接缝合格、CPU 打分通路成立、外部接口连通且可离线回放
- 观察：
- B3① StartLux-4B（本地）真下载真检：快照 30 个文件 / 9,344,023,187 字节（权重 9,319,828,240 字节，分片 ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']）；真载 tokenizer：词表 248044、26 字母独占一格且与 decision_config 的 letter_token_ids 完全一致（首末号 [32, 57]）、chat 模板 enable_thinking=False 以思考关闭后缀收尾（True）、小抄倍数 {'noul': 1.5905, 'choice': 1.3872, 'score': 1.4172}；视觉塔权重 297 个张号在场。
- B3① CPU 小批打分通路（真结构脚手架 StartLuxAI/StartLux-Decision-4B#scaffold-cpu，36.6M 参数）：3 条真评测请求（P1 render 产出）一次前向出份额，和为 1、逐条 argmax ['C/choice', 'C/choice', 'C/choice']；二次批量走缓存：命中 3 次、总前向 3 次；通路吞吐 125.9 tok/s（CPU，脚手架口径，不外推到 4B）。
- B3① 真载 4B 权重：按守卫跳过——可回收内存仅 3009.7MB < 守卫 14000MB（权重 9319828240 字节），本机 MPS 被一阶段长跑占用，按 D6 完整载入与吞吐实测延后至云端 C3；内存实测 {'page_bytes': 16384, 'free_mb': 86.9, 'inactive_mb': 2811.8, 'speculative_mb': 111.0, 'reclaimable_mb': 3009.7}。
- B3② GLM-5.3-Flash API 真调用：请求 10 条 / 实际发出 11 次，产标 10 条，平均延迟 4.032s/次，用量 {'calls': 11, 'prompt_tokens': 1375, 'completion_tokens': 831, 'reasoning_tokens': 688, 'seconds': 44.35000000000001}；颜色题答对 0/10；分布已落离线包 10 行，回放模式读回 10 行且在线请求 0 次（§11-4 全链零在线达成）。失败样例 无。（密钥全程取自环境变量 ZAI_API_KEY，未落任何文件与日志）
- B3③ 备选 Qwen3.5-4B：备选教师 Qwen3.5-4B 本域不验：主路线（StartLux-4B 文本 + GLM-5.3-Flash 视觉）已跑通，备选只在主路线降级时启用，届时按同一脚本补一次实测（verified=False）。
- 版权边界（D11）：D11：StartLux 权重仅作教师打分/判分/对照评测三种只读用途，绝不进任何训练初始化或基座。

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
结论：B3 三路实测完成：① StartLux-4B 真下载（9,344,023,187 字节）+ 真载 tokenizer 全项合格 + CPU 同架构通路打分跑通（缓存命中零前向，125.9 tok/s 脚手架口径），完整 4B 载入与吞吐按 D6 延后至云端 C3（本机可回收内存 3009.7MB 不足以安全载入）；② GLM-5.3-Flash 真调用 11 次，平均 4.032s，产标 10 行入离线包并可零在线回放；③ Qwen3.5-4B 按声明不验。
结论：更正（离线复核，零在线请求）：批次 143510 实算押对 10/10——上面那行实测里的准确率指标因把颜色名首字母当成候选字母比对而算错（显示 0/10），份额数值本身无误；代码已修（改用字母轮次比对）。
