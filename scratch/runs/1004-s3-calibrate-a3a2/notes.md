# run: 1004-s3-calibrate-a3a2

- 假设：待填写
- 观察：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 dmlaya.runs 校验：没有结论行就不许 finish()。
结论：s3 校准完成：n=100 min_samples=40 bins=15；温度表={'choice': 0.8, 'noul': 0.75}；逐类 ECE 前后={'choice': {'temperature': 0.8, 'ece_before': 0.0374, 'ece_after': 0.0256, 'improved': True, 'pooled': False}, 'noul': {'temperature': 0.75, 'ece_before': 0.0537, 'ece_after': 0.0014, 'improved': True, 'pooled': False}}；池化题型=无；各类 ECE 均未变差（Δ≤0）。
