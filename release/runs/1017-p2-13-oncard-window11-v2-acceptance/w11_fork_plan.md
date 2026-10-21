# D15 fork 修复：两车计划（实例已停，全部本地前置已就绪）

## 一、已确定的层级（静态分析到最后一跳）

```
T.Parallel(C,C) + g_ub[j]
  → ascend_lower_parallel_to_vector.cc
      CheckExpressionSupports2DVectorization   ✅ 判定正确
      CanBroadcast                             ✅ 返回 broadcast_dim=0
      CreateBroadcastSourceBuffer              ✅ 造出 [1,N] 视图
      GenerateBroadcastStmt                    ✅ 生成 Broadcast<T,2,0,false>
  → common.h:1661  AscendC::Broadcast<T, 2, axis=0, false>(...)
  → ❌ 实测 = 每列取 src[0]（行广播 axis=1 正常，列广播 axis=0 退化）
```

**未定的一步**：`AscendC::Broadcast` 的 axis=0 行为，是"TA 调用约定不对"还是"库本身缺陷"。**第一车就是判这一步。**

## 二、第一车（~10 分钟，只需 TA，不重编）

`bash .tmp_a1/w11bc3_fire.sh`

| 腿 | 内容 |
|---|---|
| **H** | 卡上 `find kernel_broadcast*.h` + grep `Broadcast(` 签名（零运行成本） |
| **v0** | `T.tile.broadcast([1,64] → [64,64])`，源用行切片填充 |
| **v1** | 同上，源用 1D→2D 复制填充（排除填充路径的影响） |
| **v2** | 行广播对照（`dst[i,j] = src[i] + 5`，已知正常） |

**健壮性（上次 w11bc2 的四个坑已修）**：
1. 探活 10×10s，不可达即 `ABORT`（不再静默空等）
2. TA 自恢复（36.5MB 包，实例重建后 /tmp 清空时自动重铺）
3. 日志 300s 无新行 ⇒ 主动 pkill（不再跑满 timeout 空转，这正是"NPU 满"的来源）
4. H 腿结果写入 all.log 并落盘

### 判读表（四分支的修复动作预先定死）

| v0 | v1 | v2 | 结论 | 第二车改哪 |
|---|---|---|---|---|
| ✅ | ✅ | ✅ | 列广播在独立 kernel 正常 ⇒ **问题在 scan2a 上下文**（UB 压力/形状） | 不动 lowering，查 scan2a 的 `CreateBroadcastBuffer` 生成（临时 buffer 16KB 是否与 UB 91% 冲突） |
| ❌ 退化 | ❌ | ✅ | **AscendC axis-0 缺陷** | `GenerateBroadcastStmt` 的 axis=0 分支：改用「transpose → axis=1 广播 → transpose」三段等价序列 |
| ❌ 507015 | — | ✅ | TA 的 broadcast 路径不可用（独立 kernel 也崩） | 同上，直接走三段序列（不调 AscendC::Broadcast） |
| ❌ 崩 | — | — | TA 调用约定错（srcShape 传法） | 改 `GenerateBroadcastStmt` 的 srcShape 参数（`{64,1}` vs `{1,64}`） |

## 三、第二车（~60–90 分钟，含重编 libtvm）

前置已就绪：
- **完整源码**：`3rdparty/tvm` submodule 已拉（`src/` 完整）
- **编译命令**：`cmake -S . -B build -DUSE_ASCEND=ON && cmake --build build -j$(nproc)`（`USE_ASCEND` 见 CMakeLists:130）
- **dev 布局**：运行时从 `build/lib/` 与 `build/tvm/` 加载（tilelang/env.py）

步骤：
1. 按判读表改 `GenerateBroadcastStmt`（**只改 TA，不动生产 kernel**）
2. 传源码到卡（`tilelang-ascend` 源码 + 已拉的 tvm；体积需先 `du -sh` 确认）
3. 卡上 cmake 配置 + 编译（**首个风险点**：依赖/CMake 选项）
4. 用 `_w11_bcast_probe.py`（5 行金丝雀）验证列广播变绿
5. 若金丝雀绿 ⇒ 跑 `w11prod_fire.sh`（S1G + L0）看六梯度

### 第二车的风险与兜底
| 风险 | 概率 | 兜底 |
|---|---|---|
| cmake 配置失败（缺依赖） | 中 | 先跑 `cmake -S . -B build -DUSE_ASCEND=ON` 只配置不编译，看报错 |
| 编译 30–60 分钟超时 | 中 | 分单元编（`cmake --build build --target tvm`），或只编改动文件所在对象 |
| 改后仍退化（三段序列也撞 backend） | 中 | 回到 issue 路线；C 方案（逐 token，20–60× 性能代价）保底 |
| TA 源码与 ta_pkg 的 libtvm 版本不匹配 | 低 | 改完必须整编，不混用旧 .so |

## 四、不做什么（明确边界）

- **不改** `common.h` 里的 `AscendC::Broadcast` 调用签名以外的生产 kernel 逻辑
- **不长期 fork**：patch 的定位是"给上游的参考实现 + 临时验证工具"，不是私养分支
- **不再试**第八/第九种算侧形态（七种已穷尽）

## 五、当前可交付的确定成果（与本计划并行保留）

| 成立 | 数据 |
|---|---|
| solve_a 的 A（g≠0） | rel 5.3e-05，rowsum 逐位一致 |
| scan1 的 H 链（g≠0） | rel 5.1e-04（< 5e-3 PASS） |
| DQ 通道 copy | 生成码 `(64,0,0)`，经 dq1_l0 直拷 |
| 缺陷证据链 | 5 行复现、七形态失败、910B/950 容量表、#1194/#1219 同族关系 |
