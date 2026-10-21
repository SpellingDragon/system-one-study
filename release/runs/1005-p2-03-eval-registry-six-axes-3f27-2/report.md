## 数据底账（真实下载量，非估算）

```
id                   source      revision     split seed       axis          samples       bytes status
-------------------------------------------------------------------------------------------------------
typed-decisions      hf          f7a2487e     test  -          quality          2000     824,881 cached
intern-decision      archive     2f81580      test  -          quality         12447  24,426,697 cached
jevbench             archive     7ce310c7     test  -          calibration       231   4,640,431 cached
cmmlu-subset         modelscope  master       test  20261005   quality           200   1,078,656 cached
clue-subset          modelscope  master       validation 20261005   quality           400     944,526 cached
mmbench-cn-subset    modelscope  master       dev   20261005   multimodal        200  96,700,059 cached
longbench-zh         modelscope  master       test  20261005   longctx           200 113,932,529 cached
needle-synthetic     synthetic   20261005     n/a   20261005   longctx            30         524 fetched
```

- 数据版本串：`eval-registry[typed-decisions@f7a2487e,intern-decision@2f81580,jevbench@7ce310c7] sets=8 samples=15708`
- 累计下载字节：242,548,303（出自 fetch 的真实网络计数；命中缓存时报的是当初真下载的那份）
- 登记条数合计：15,708（其中带人工答案 15,678）
- 执行后端：`cpu`；npu 为云端占位，接入前必被拒绝（见 p2-13）

## 六轴表

```
axis         status      sets(present/registered)     samples  w/gold reason
----------------------------------------------------------------------------------------------------------------------
quality      data-ready  4/4                            15047   15047 可判分信封 2/4 集（其余为原始题面副本，决策化归对应专域）
calibration  data-ready  1/1                              231     231 
longctx      data-ready  2/2                              230     200 题面已自持；判分件属 p2-07，本域不代打；可判分信封 0/2 集（其余为原始题面副本，决策化归对应专域）
multimodal   data-ready  1/1                              200     200 题面已自持；判分件属 p2-08，本域不代打；可判分信封 0/1 集（其余为原始题面副本，决策化归对应专域）
speed        n/a         0/0                                0       0 registry 里没有挂在这条轴上的集
parity       n/a         0/0                                0       0 parity=n/a：backbone 前向是单一实现路径（本机 CPU/MPS；云端换卡由 p2-13 承接），无双路
----------------------------------------------------------------------------------------------------------------------
backend=cpu  （npu 尚未接入：p2-13 sys1.kernels.backends）
parity 裁定：P2 一致轴只测 P1 自训 decoder 的延续路径；HF 主干显式 n/a
```

## 一致轴裁定（B3 · P1 悬空轴教训回写）

- 裁定：P2 一致轴只测 P1 自训 decoder 的延续路径；HF 主干显式 n/a
- 测什么：同一份输入下，自研算子路与纯 torch 路的名次是否换人（沿用一阶段 parity 口径）
- 不测：HF 主干（经 transformers 载入的那条前向）——它只有一条实现路径，自己跟自己比恒等于全对
- 不测：昇腾卡上的算子对拍——归 p2-13 的梯度对拍验收（G9），不在本域
- 不测：远端服务与本地权重的一致性——归 p2-10 serving

- 报告脚注（parity 为 n/a 时原样带上）：parity=n/a：backbone 前向是单一实现路径（本机 CPU/MPS；云端换卡由 p2-13 承接），无双路可对拍；自研件的 parity 验收见 p2-13 G9

## 读表须知

- `n/a` 不等于没跑：要么是没得测（单一实现路径），要么是数据还没拉，reason 列写清是哪种。
- `data-ready` 只保证题面与真值在盘上、来源可回溯；长文与多模态的判分件分属 p2-07 / p2-08。
- 本域不出模型分：给了 `--model` 或 `--endpoint` 才走一阶段预测出口，口径复用不改写。
