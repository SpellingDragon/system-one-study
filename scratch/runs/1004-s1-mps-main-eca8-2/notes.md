# run: 1004-s1-mps-main-eca8-2

- 假设：待填写
- 观察：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 dmlaya.runs 校验：没有结论行就不许 finish()。

## 采样冒烟（温度采样，逐条原文）

1. （起头 '今天天气'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 4、英文词 ['the', 'the', 'the', 'the']）
   > 今天天气 the the the the the  the the the the the     and attack     the 行政处罚  the
2. （起头 '深度学习需要'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 6、英文词 ['the', 'inal', 'the', 'the']）
   > 深度学习需要, the of inal the  the the the the the , families the the . world the a term  
3. （起头 'The capital of France is'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 0、英文词 ['capital', 'France', 'the', 'the']）
   > capital of France is the  the    the  the  in      境    the   to       ˈ
4. （起头 'Once upon a time'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 0、英文词 ['Once', 'upon', 'time', 'term']）
   > Once upon a time  term                                     
5. （起头 '下列选项中正确的是'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 9、英文词 ['the', 'the', 'the', 'not']）
   > 下列选项中正确的是 the the the 锡 not the the 
 the by term the the the of the the 支 the the
6. （起头 'In 1969, humans'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 0、英文词 ['humans', 'the', 'the', 'the']）
   > In 1969, humans 滘 to 鸭  in the the , the  not the the the the the the  of the the 
7. （起头 '床前明月光，'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 5、英文词 ['the', 'the', 'the', 'the']）
   > 床前明月光， the 脆 the the , the the 
 a to the  and  锡 the the the the   the
8. （起头 'Evidence shows that'） 40 个片段，止于 max_new_tokens：有像话片段（汉字连串 0、英文词 ['Evidence', 'shows', 'that', 'the']）
   > Evidence shows that the  No the the   the      the          ment  the   of 鸭

初筛：8/8 条形状上像话（判据：连续汉字>=2 或 连续英文词>=3 字母）；学习曲线 loss 9.8506 -> 8.2459。
人工判定：待复核（教师版按 design 口径，可读短句是过程信号不是 gate，不达标如实记）。
结论：随机初始化起步练到 step=30（读入 1,966,080 个片段，设备 mps）：loss 9.8506 -> 8.2459，确实往下走；采样 8 条，形状初筛像话 8 条（判据 连续汉字>=2 或 连续英文词>=3 字母）。最终模型在 model/，续训起点在 ckpt/（latest.json 指向），p1-09 可直接按模型目录消费。
