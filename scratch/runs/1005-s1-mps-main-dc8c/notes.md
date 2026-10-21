# run: 1005-s1-mps-main-dc8c

- 假设：待填写
- 观察：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。

## 采样冒烟（温度采样，逐条原文）

1. （起头 '今天天气'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 5、英文词 ['number', 'rifth', 'muters', 'per']）
   > 今天天气管, 2011.
A number of a rifth of 10 km of a muters per cent to d
2. （起头 '深度学习需要'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 32、英文词 []）
   > 深度学习需要的同时,将学习教育学习在当中,要认真学习教育学习,学习贯彻习近平新时代中国特色社会主义思想建设的党的二十大精神大局.要坚持以习近平新时代中国特色社会主义思想
3. （起头 'The capital of France is'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 0、英文词 ['capital', 'France', 'used', 'ensure']）
   > capital of France is used to ensure that it is the highest number of a company that has been the best retainer information
4. （起头 'Once upon a time'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 0、英文词 ['Once', 'upon', 'time', 'bath']）
   > Once upon a time or a bath in the ground. The size of the other can be used to provide a schec
5. （起头 '下列选项中正确的是'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 22、英文词 []）
   > 下列选项中正确的是,否则就算不算到底是有的,不能做的,要进行.
可查,在当前,必须要给本人所造成的后果会给自己和妻子的监护
6. （起头 'In 1969, humans'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 0、英文词 ['humans', 'were', 'the', 'world']）
   > In 1969, humans were in the world. As part of this new part of the National Wynaps in 2020, it 
7. （起头 '床前明月光，'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 14、英文词 []）
   > 床前明月光， 记 员 李春霞

附:相关法律条文

"中华人民共和国民事诉讼法"第一百七十条

第被告人的罪行在判决书生效后,不履行生效法律文书确定的义务,并将全部债务返还财产的
8. （起头 'Evidence shows that'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 0、英文词 ['Evidence', 'shows', 'that', 'was']）
   > Evidence shows that he was a new and more than a time in the day of those who did not make it much

初筛：8/8 条形状上像话（判据：连续汉字>=2 或 连续英文词>=3 字母）；学习曲线 loss 9.8506 -> 3.4286。
人工判定：待复核（教师版按 design 口径，可读短句是过程信号不是 gate，不达标如实记）。
结论：随机初始化起步练到 step=2000（读入 131,072,000 个片段，设备 mps）：loss 9.8506 -> 3.4286，确实往下走；采样 8 条，形状初筛像话 8 条（判据 连续汉字>=2 或 连续英文词>=3 字母）。最终模型在 model/，续训起点在 ckpt/（latest.json 指向），p1-09 可直接按模型目录消费。
