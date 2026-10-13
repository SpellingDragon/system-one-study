# 窗十台账：#1307 workaround 证伪 + 对齐版验收失败 + P1-1zi 一旗真判决

- 窗：`window10_card.sh` + `run_window10_local.sh`（port910b 根）；实例 1013 晚新 boot（/tmp 全清，ta_pkg 重传）
- 判决前序：查重发现 #1307（GetWithOffset size 不取整 32B）→ 衍生 workaround 假说 → 本窗判决

## [1] v14 A/B：#1307 workaround 假说**证伪**

- A 腿（`[1]` buffer 对照）：FAIL 507015（预期，复现基线）
- **B 腿（全部小 buffer 放大 `[8]`=32B）：仍 FAIL 507015**——size 对齐不救
- 结论：我们的 507015 不是（或不只是）#1307 的 size 不取整机制；issue 口径回撤为独立缺陷+引用相关族

## [2][3] 对齐版 ladder / L0：全 FAIL（同 507015）

- 对齐版 gdn_bwd（16246B，容器编译门 2/2 OK：src=25969/26226）上卡：ladder 7 档全 FAIL、L0 六 case 全 FAIL——与 v14 B 腿一致

## [4] P1-1zi 一旗：两枪工程坑后**真判决落地**

- 第一枪（字符串注入）：`ffi.Array` 类型拒——fork 的 `tl.device_compile_flags` 要 list 形态
- 第二枪（list 注入 + `env -u LD_LIBRARY_PATH` 过度清除修掉 libhccl.so）：**六项判决全部跑出**：
  ```
  dq 6.999e-08 PASS / dk 9.322e-08 PASS / dv 6.629e-08 PASS
  dbeta 8.002e-01 FAIL   ← 唯一败项，与无旗态逐字同值
  dg 1.016e-07 PASS / dh0 6.176e-08 PASS
  ```
- **判决：不对**——`--cce-auto-sync` 不救 dbeta；fork 缺核内 vector↔scalar 同步假说未获证实，P1-1zi 关闭（止损结论加固）
- **新信息**：fork GDN bwd 六项中五项真机 PASS（1e-7/1e-8 级），病灶收窄为 **dbeta 单输出**（此前 FAIL 早退从未见过全六项）

## 工程坑定案（复用价值）

1. fork 腿环境隔离律修正：`env -u PYTHONPATH -u TA_ROOT` **保留 `LD_LIBRARY_PATH`**（CANN 的 libhccl.so 在其中；窗九 c/d 两轮 autoload 失败同因）
2. `tl.device_compile_flags` 值必须是 **list**（`ffi.Array`），字符串直接 AttributeError 回退 eager
3. 一旗注入 sed 配方（list 版）与回滚已验证（W10B-REVERTED=0）

## 对 issue 的影响

- Prior-art 节更新：v14 证伪补入（帮上游排除 size-对齐假设）；发布形态维持"独立 bug + 引用 #1307/#1680 族"
- fork 侧新信息（dbeta 单项病灶）与 TA 侧 507015 是两个独立缺陷，issue 只报 TA 侧
