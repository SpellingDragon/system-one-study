# run: 1004-s1-smoke-tiny-2532

- 假设：本 run 是 CLI 级 --resume 的真实验证（不是单测桩）：先用 --max-steps 150 练一段并留下存档，再以 --resume runs/1004-s1-smoke-tiny-2532 --max-steps 300 接上，验「从断点步继续、曲线不出现重复页码」在真实 runs/ 产物上也成立（spec 断点续训场景）。
- 观察：段一练到 step=150（loss 9.8596 到 7.4905），存档 step-000150 被 latest.json 指到；--resume 认出原 run、验指纹通过（max_steps 属可调项不入指纹），从 step=150 接着练到 300，曲线 17 行、步号 [0,20,...,140,150,160,...,300] 单调且无重复。末值 6.7746 比一次跑完的 6.3722 略高，原因如实记下：学习率按本次配置的收手步数现算，续训段在 step=151 处重新拿到接近峰值的 lr，退火弧线比连续跑的那次短——是分两段练的固有代价，不是数值坏了，也不是曲线作假。
- 结论：见下面两条结论行（两段各一条，由脚本依实测数字自动写下，未手改）。

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 dmlaya.runs 校验：没有结论行就不许 finish()。
结论：随机初始化起步练到 step=150（读入 1,228,800 个片段，设备 cpu）：loss 9.8596 -> 7.4905，确实往下走；采样 0 条，形状初筛像话 0 条（判据 未采样）。最终模型在 model/，续训起点在 ckpt/（latest.json 指向），p1-09 可直接按模型目录消费。

## 采样冒烟（温度采样，逐条原文）

1. （起头 '今天天气'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 4、英文词 ['for', 'the', 'This', 'dtheir']）
   > 今天天气 d for the This dtheir for 在  he andy as al the out an other d out为 at
2. （起头 '深度学习需要'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 6、英文词 ['you', 'the', 'which', 'and']）
   > 深度学习需要  you the p which and the has  out or can what your) the? be  can caning1 
3. （起头 'The capital of France is'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 2、英文词 ['capital', 'France', 'The', 'other']）
   > capital of France is or f The f p other it in un a foring   of a The的的 out anThe (
4. （起头 'Once upon a time'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 0、英文词 ['Once', 'upon', 'time', 'you']）
   > Once upon a time you  for   to   for d and  p al have that for or  be that  在 for
5. （起头 '下列选项中正确的是'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 9、英文词 ['you', 'all', 'all', 'you']）
   > 下列选项中正确的是 you of all p all ay un不 you NanThe 在The al theed  it to y for 
6. （起头 'In 1969, humans'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 0、英文词 ['humans', 'have', 'you', 'which']）
   > In 1969, humans have you which The  you  )y to  with for N and which  he  on their p  
7. （起头 '床前明月光，'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 5、英文词 ['with', 'can', 'you', 'which']）
   > 床前明月光， a with can  be:   be s a you be s which has     with  a s 
8. （起头 'Evidence shows that'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 0、英文词 ['Evidence', 'shows', 'that', 'for']）
   > Evidence shows that he for. to to and al The  has other a ( y was bely and which  the be

初筛：8/8 条形状上像话（判据：连续汉字>=2 或 连续英文词>=3 字母）；学习曲线 loss 9.8596 -> 6.7746。
人工判定（教师复核 8 条原文）：**同样没有一条构成人类可读的句子**。与一次跑完那次相比，这段的续写更碎（8 条全部撞到 40 个片段的上限、无一条自己发出整篇结束位），英文侧仍是 the/for/and/you 这类高频词打转，中文侧只保住起头那几个字后面偶尔冒出「不」「为」「的的」这类单字；与末 loss 略高相互印证。按 spec 口径判为不达标，如实记；分两段练不改善可读性，改善要靠 mps_main 档的量级。
结论：随机初始化起步练到 step=300（读入 2,457,600 个片段，设备 cpu）：loss 9.8596 -> 6.7746，确实往下走；采样 8 条，形状初筛像话 8 条（判据 连续汉字>=2 或 连续英文词>=3 字母）。最终模型在 model/，续训起点在 ckpt/（latest.json 指向），p1-09 可直接按模型目录消费。
