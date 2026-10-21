# run: 1004-s3-calibrate-5e08

- 假设：待填写
- 观察：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 dmlaya.runs 校验：没有结论行就不许 finish()。
结论：s3 校准完成：n=60 min_samples=10 bins=15；温度表={'choice': 1.4, 'noul': 3.9}；逐类 ECE 前后={'choice': {'temperature': 1.4, 'ece_before': 0.0442, 'ece_after': 0.1128, 'improved': False, 'pooled': False}, 'noul': {'temperature': 3.9, 'ece_before': 0.1956, 'ece_after': 0.0005, 'improved': True, 'pooled': False}}；池化题型=无；存在 ECE 变差类（样本小/弱模型，如实记录）。
