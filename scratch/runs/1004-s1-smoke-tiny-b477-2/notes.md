# run: 1004-s1-smoke-tiny-b477-2

- 假设：本 run 由 tests/test_s1_train.py::test_e2e_smoke_tiny_end_to_end（C2，-m slow）在 DMLAYA_KEEP_ARTIFACTS=1 下真实跑出：同 smoke_tiny 档、同 seed，应当与 1004-s1-smoke-tiny-b477 得到逐字相同的曲线与采样——用它验证这条端到端路径可复现、CPU 用时在 10 分钟内。
- 观察：loss 9.8596（step=0）到 6.3722（step=300），16 行曲线无重复页码，读入 2,457,600 个片段，8 线程 CPU 实测 79.5s（预算 600s）；ckpt/step-000100/200/300 三份存档在位、latest 指向 300，存档与 model/ 均可被 Decoder.load 载回；采样 8 条原文与首个 run 逐字一致（同 seed 复现成立）。
- 结论：见文末结论行（由 learning/s1_pretrain_gpt.py 依实测数字自动写下，未手改）。

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 dmlaya.runs 校验：没有结论行就不许 finish()。

## 采样冒烟（温度采样，逐条原文）

1. （起头 '今天天气'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 4、英文词 ['have', 'can', 'The', 'for']）
   > 今天天气 have  can The to C for to  an and it that s This  other The the s’s (
2. （起头 '深度学习需要'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 7、英文词 ['the', 'the', 'one', 'foras']）
   > 深度学习需要的 a the B to the  B it on in or one  in foras be  can can other 
3. （起头 'The capital of France is'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 0、英文词 ['capital', 'France', 'one', 'the']）
   > capital of France is to one the on this it  p: have a the"   In a Theing: one ans. d
4. （起头 'Once upon a time'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 0、英文词 ['Once', 'upon', 'time', 'and']）
   > Once upon a time a and In the In and be  and  are s has in the to for your in can the
5. （起头 '下列选项中正确的是'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 9、英文词 ['that', 'with', 'with', 'and']）
   > 下列选项中正确的是 a that ( with with and ( can in C) to The 在 for的  which  The In can
6. （起头 'In 1969, humans'） 14 个片段，止于 stop_token：有像话片段（汉字连串 0、英文词 ['humans', 'are', 'your']）
   > In 1969, humans are in of to a your In
7. （起头 '床前明月光，'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 5、英文词 ['asy', 'and', 'the', 'and']）
   > 床前明月光， of asy and In the and and and the p are which a your The  toes  The s
8. （起头 'Evidence shows that'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 0、英文词 ['Evidence', 'shows', 'that', 'with']）
   > Evidence shows that  a in or be an S  d p with for to  was and and for the which this 

初筛：8/8 条形状上像话（判据：连续汉字>=2 或 连续英文词>=3 字母）；学习曲线 loss 9.8596 -> 6.3722。
人工判定（教师复核，与首个 run 同一份 8 条原文）：**没有一条构成人类可读的句子**。看得见的进步是分布层面的：英文侧反复吐出 the/and/of/with/for/one 这类高频功能词，也能接住「In 1969, humans ...」那个架子；中文侧只在起头之后维持一段汉字（第 5 条汉字连串 9、第 7 条连串 5），续出来的都不是词；第 6 条在 14 个片段后自己发出了整篇结束位，说明「这话说完了」这个信号确实学到了。按 spec「至少出现一条人类可读句」的口径：本档不达标，如实记。300 步 / 2.46M 编号 / 3.17M 参数的容量与数据量都远不足以成句；可读句子的判定要到 mps_main 档（~40M 参数 × 1.3 亿编号）再核。
结论：随机初始化起步练到 step=300（读入 2,457,600 个片段，设备 cpu）：loss 9.8596 -> 6.3722，确实往下走；采样 8 条，形状初筛像话 8 条（判据 连续汉字>=2 或 连续英文词>=3 字母）。最终模型在 model/，续训起点在 ckpt/（latest.json 指向），p1-09 可直接按模型目录消费。
