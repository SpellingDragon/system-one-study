# run: 1004-s1-smoke-tiny-7c3f

- 假设：待填写
- 观察：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 dmlaya.runs 校验：没有结论行就不许 finish()。

## 采样冒烟（温度采样，逐条原文）

1. （起头 '今天天气'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 12、英文词 ['verymagInter', 'out', 'youmusicsem', 'allowsitive']）
   > 今天天气能够,即verymagInter00元,从蒺北,我们健赛ী out永 究荀固事冀 youmusicsem热allowsitive f"中华人民共和国民事诉讼法"第一百四十四res状尉 重大build梯two
2. （起头 '深度学习需要'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 18、英文词 ['cilreg', 'lecast', 'daily', 'adult']）
   > 深度学习需要固付增加解除劳动合同出庭(199殇cilreg村民lecast壮"最高人民法院关于审理规定的  y적daily啪,且埔adult 条减半跤e.物力)  andace固炉校19
3. （起头 'The capital of France is'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 8、英文词 ['capital', 'France', 'bluebeen', 'list']）
   > capital of France is 条担,造成 条bluebeen list forown  成绩温影浞罪犯每 母   村民条件collect适用法律回build梯上诉very回冀固h逾期
4. （起头 'Once upon a time'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 10、英文词 ['Once', 'upon', 'timebeen', 'for']）
   > Once upon a timebeen for 回na  fright   村民霈 individual父母 fight cur 判处有期徒刑问题的影ted  years   pe very comes 回
5. （起头 '下列选项中正确的是'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 12、英文词 ['effectse', 'known', 'arderal', 'curglvery']）
   > 下列选项中正确的是被告王 A.effectse.cl每known鬼arderal加班劶,本院不予支持. 条ne  curglvery"最高人民法院关于审理,即i 雅inc buildCounty project  条固close成绩万余
6. （起头 'In 1969, humans'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 10、英文词 ['humansalspoaebuildcomesushsaf', 'displaypeople', 'wardtal', 'nopopulsh']）
   > In 1969, humansalspoaebuildcomesushsaf同期究判处有期徒刑每,且 止月18日诚团除外酮편逾期总成绩附 在displaypeople拿wardtal nopopulsh,可晚上温turn 
7. （起头 '床前明月光，'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 9、英文词 ['cture', 'common', 'findous', 'noctureather']）
   > 床前明月光，究砷cture龛驳回原告common在 雅findous noctureather见 for otherlong征美国рking壮alfearestic y盟transment回迷万余最高向原告 
8. （起头 'Evidence shows that'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 10、英文词 ['Evidence', 'shows', 'that', 'out']）
   > Evidence shows that out征村民lling f f 条万余收到 work  out月16日温  重大or.编多次催 al状态或者约定不收到月18日inside村民udees no work

初筛：8/8 条形状上像话（判据：连续汉字>=2 或 连续英文词>=3 字母）；学习曲线 loss 9.8596 -> 8.5636。
人工判定：待复核（教师版按 design 口径，可读短句是过程信号不是 gate，不达标如实记）。
结论：随机初始化起步练到 step=60（读入 491,520 个片段，设备 mps）：loss 9.8596 -> 8.5636，确实往下走；采样 8 条，形状初筛像话 8 条（判据 连续汉字>=2 或 连续英文词>=3 字母）。最终模型在 model/，续训起点在 ckpt/（latest.json 指向），p1-09 可直接按模型目录消费。
