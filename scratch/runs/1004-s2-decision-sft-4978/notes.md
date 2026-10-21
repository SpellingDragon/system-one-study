# run: 1004-s2-decision-sft-4978

- 假设：待填写
- 观察：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 dmlaya.runs 校验：没有结论行就不许 finish()。
结论：s2 smoke 完成：train=540 holdout=60 epochs=1 lr=0.03；留出集各桶准确率 vs 分桶随机基线：{'choice': {3: {'accuracy': 0.6364, 'baseline': 0.3333333333333333, 'beats_baseline': True}}, 'noul': {2: {'accuracy': 0.5789, 'baseline': 0.5, 'beats_baseline': True}}}；全部桶超基线。
