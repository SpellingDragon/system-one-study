# run: 1005-probe-p2-external-assets-sweep-7e80

- 假设：待填写
- 观察：待填写

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 dmlaya.runs 校验：没有结论行就不许 finish()。
结论：sweep 结论：P2 外部资产 8 项核查完毕——6 通 2 需改道（MMBench→GitHub；StartLux 权重→hf-mirror 唯一源）+1 命令适配（HF_ENDPOINT）。方法论教训升格为 R15：计划引用的每个外部资产（模型/数据集 id、仓库、框架版本兼容）在规划期就应以 API 探测实证并 pin；'待拉取后确认'=把低成本验证计划性推迟，评审不许过。另：本次第一版探测脚本复现了 hasattr 静默 OK 的假绿模式，直连复核纠正——探测脚本自身也要防假绿。
