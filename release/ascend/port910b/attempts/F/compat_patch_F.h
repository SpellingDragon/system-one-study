// ════════════════════════════════════════════════════════════════════════════
// compat_patch_F.h — 910B(dav-2201) port910b_compat.h 缺口补丁（GAP-F）
// p2-13 甲路 P1-3 波（执行代理 F，GDN delta rule 前后向）
//
// 本块只补 **一个** 符号：`logf`（G-F0）。
//   · 触发形态：tilelang DSL `T.log(float32)` → codegen 发射标量 `logf`
//     （src/ascend/codegen/intrin_rule_ascend.cc:22 AscendMath，float32 一律 name+'f'；
//      :50 REGISTER_ASCEND_INTRIN("log")）。取证件：attempts/F/f_probe_logf.py
//     （未打补丁 → `use of undeclared identifier 'logf'`；打补丁 → COMPILE-PASS，
//      差分即为本缺口的正/负形对照）。
//   · 现状：trunk §11(GAP-C) 只补了 `expf/fabsf`，§GAP-B 补了 `rsqrtf`，
//     `logf/log2f/tanhf/powf/exp10f/sqrtf/floorf` 仍未补（compat_gap_C.md §4-4 已预告）。
//   · 生产必要性：GDN 的 decay 有两种入口口径——
//       (a) log 域直接给 g（a = exp(g)）：**本波 delta rule 件走这条，不需要 logf**；
//       (b) 线性域给 a∈(0,1]（fla/部分实现 g = log(a)；chunked 形态要 log 域累加和）：
//           需要 logf。P2 三栈联调替换 Conv2D/GDN 路径时若走 (b)，本块即前置条件。
//     ⇒ 结论："decay 的 log 缺件" 只在口径 (b) 成立；(a) 口径本波实测零新增缺口。
//
// 实现口径（与 §11 软件 expf 同风格）：纯标量、无 double、无 libcall。
//   范围规约：x = m·2^e，m∈[1,2)；m>√2 时折半使 m∈[0.7071,1.4142]，
//   则 z=(m−1)/(m+1)∈[−0.1716,0.1716]，ln(m)=2·z·(1+z²/3+z⁴/5+z⁶/7+z⁸/9)，
//   截断误差 ~z^10·(1/11)/(1−z²) ≤ 2.4e-10（ln(m) 的绝对误差）；ln(x)=ln(m)+e·ln2。
//   数值定标：attempts/F/f_probe_logf.py --accuracy 打 numpy 逐行复刻，
//   LOG-SWEEP 实测 max_rel（见 gap_F.md/RESULT.md 的真实数字，非纸面推演）。
//   边界：x<=0 → 返回 -FLT_MAX（**故意钳位而非 NaN/-inf**：910B 标量面无 NaN 传播契约，
//   且 decay 语义下 x<=0 属非法输入，钳位比 NaN 更早在校验里暴露）；
//   次正规数（exp 域全 0）按 e=-127 处理，误差可达 O(1) —— decay 值域不触及，
//   若将来用于任意 x 需另补次正规分支（已登记 G-F5）。
//
// ⚠️ **合并纪律（G-F1，实测坑）**：位视图 helper 在 GAP-C 里已占用名字
//   `tl910b_gap_f2u / tl910b_gap_u2f`（trunk 真源 :712-719 实测）。任何新 GAP 块
//   **照抄这两个名字**就会与 §11 撞"重定义"而整树编不过。本块的做法是：
//   能复用就复用（`#ifdef TL_PORT910B_COMPAT_GAP_C_H` 判 GAP-C 是否已在树内），
//   否则自带 F 专名副本（tl910b_gapF_*）。编排者合并时无需改动本块。
//
// 注入方式（**禁改主仓真源**）：attempts/F/patch_compat_F.py 把本块**追加**到容器
//   pip 运行副本 …/tilelang/src/tl_templates/ascend/port910b_compat.h 文件尾（自带守卫，
//   不动既有任何一行），由 run_F.sh 幂等装配；成果以本文件上报，由编排者波后合并。
// 退让：单个名字还给上游 → -DTL_PORT910B_SKIP_logf；整块不注入即完全无影响。
// ════════════════════════════════════════════════════════════════════════════
#ifndef TL_ASCEND_SIMT
#ifdef TL_PORT910B_NATIVE_TYPES
#ifndef TL_PORT910B_COMPAT_GAP_F_H
#define TL_PORT910B_COMPAT_GAP_F_H

// ── 位视图：优先复用 GAP-C 的 __aicore__ 件，避免同名重定义（G-F1）──
#ifdef TL_PORT910B_COMPAT_GAP_C_H
#define TL910B_F2U(x) tl910b_gap_f2u(x)
#define TL910B_U2F(b) tl910b_gap_u2f(b)
#else
__aicore__ inline unsigned int tl910b_gapF_f2u(float x) {
  float q = x;
  return *reinterpret_cast<unsigned int *>(&q);
}
__aicore__ inline float tl910b_gapF_u2f(unsigned int bits) {
  unsigned int u = bits;
  return *reinterpret_cast<float *>(&u);
}
#define TL910B_F2U(x) tl910b_gapF_f2u(x)
#define TL910B_U2F(b) tl910b_gapF_u2f(b)
#endif

__aicore__ inline float tl910b_logf(float x) {
  if (x <= 0.0f) return -3.4028234663852886e38f;  // 钳位（非 NaN），理由见文件头
  const unsigned int u = TL910B_F2U(x);
  int e = static_cast<int>((u >> 23) & 0xffu) - 127;
  const unsigned int mant = u & 0x7fffffu;
  float m = TL910B_U2F(0x3f800000u | mant);       // m ∈ [1,2)
  if (m > 1.4142135623730951f) {                 // 折半收窄到 [0.7071,1.4142]
    m = m * 0.5f;
    e += 1;
  }
  const float z = (m - 1.0f) / (m + 1.0f);
  const float z2 = z * z;
  const float s = 1.0f + z2 * (0.3333333333333333f + z2 * (0.2f +
                  z2 * (0.14285714285714285f + z2 * 0.1111111111111111f)));
  return 2.0f * z * s + static_cast<float>(e) * 0.6931471805599453f;
}

// codegen 发射的是无修饰全局名 → 函数式宏接走（宏而非重载：上游补真声明也不撞种类）
#ifndef TL_PORT910B_SKIP_logf
#define logf(x) tl910b_logf(x)
#endif

#endif  // TL_PORT910B_COMPAT_GAP_F_H
#endif  // TL_PORT910B_NATIVE_TYPES
#endif  // TL_ASCEND_SIMT
