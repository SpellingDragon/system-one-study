# run: 1007-p2-13-p01-910b-inventory-885c

- 假设：950 依赖面**可被静态穷举并四分类**——A 已解 / B 兼容层可解 / C 需 lowering 改造 / D 只能上卡裁决；且 A+B 的占比决定『移植是数天工程还是数周工程』（B1 判决留下的乐观假设）。
- 观察：清单 **349 个去重符号 = A35 / B20 / C250 / D44**（12 族穷举；`./checklist.sh` → `AUDIT PASS: 穷举 354 个符号 100% 见于清单（unlisted=0）`，354 是族内 uniq 累加、349 是跨族去重）。**C 类占 72%** → 乐观假设不成立：移植主体不是缺头/缺别名（那只有 55 个），而是向量侧整个压在 950 SIMT/SIMD 方言上（160 个 `asc_*` + 谓词寄存器 + 线程模型）。最大未知数 **D2-AIC-API（14 符号）**：`asc_init/asc_mmad/asc_lock/asc_copy_*` 在 CANN 8.5.2 是否有声明，静态不可判；而 `asc_init();` 由 codegen **无条件**发在每个 kernel 体首行（codegen_ascend.cc:710-712）——这是 P0-2 vecadd 的生死线，也是 §4-墙0 质疑 design.md 证据链的落点。B 类 20 符号已离线闭环：compat 头在主机 clang16 三种后端配置下 `SELFCHECK PASS`（fp16 与 `_Float16` 神谕 20 万样本 mismatch=0），自检还抓出 2 个真实实现缺陷（RNE 偏置、有符号比较陷阱）。另 3 条新发现：reduce.h 漏罩、C++ 侧零 arch gate、七件 kernel 全用 T.SimtVF。

> 三行必填；失败的实验同样要留下结论行（PRODUCTION §9.1 负结果入库）。
> 收尾由 sys1.runs 校验：没有结论行就不许 finish()。
结论：P0-1 判决：**依赖面已穷举且完备性可自证**（349 符号 / A35 B20 C250 D44，audit unlisted=0 + gap 80 条已复核），但**甲路的真实门槛从『补头』移到『改 lowering』**——C 类 72%，且其公共前置件是『C++ 侧 arch 感知通道』（全仓现在没有）。给 P0-2 的开局建议：**先花 15 分钟跑 `probe910b.sh P1 P2 P3 P9` 再动任何一行移植代码**；P1/P2/P3 决定 AIC 通路是『补声明即可』还是『重写为 legacy AscendC』（工作量从数天跳到数周），P9 决定 compat 头要不要逐个 `-DTL_PORT910B_SKIP_<name>` 让位给 bisheng 内建。在此之前 INVENTORY §3 里所有『910B 等价物』字样一律按推断级看待，不得作为承诺依据。P1-1 的前置条件应改为 **C1（SIMT 载体决策）**，不是 D2。

勘误（写后校验三连之锚 grep 抽验时发现）：`asc_init();` 的发射行精确为 `src/ascend/codegen/codegen_ascend.cc:711`（`:710` 是 `PreFunctionBody` 签名行）；上文与 INVENTORY.md §0 原引 `710-712` 已改为 `711`。其余 13 个抽检锚点全部命中。
