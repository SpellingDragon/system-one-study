# T_RESULT.md — 代理 T / p2-13 **P1-1g G-gate 修复**（ascend_env._compile_probe 探针换 910B 合法载体）

**判决环境**：容器 `cann910b-i`（py3.10 + CANN 8.5.0 + patched tilelang 0.1.15，§12+patch_bisheng
就绪；`source /usr/local/Ascend/cann-8.5.0/set_env.sh; export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201`，release 挂 `/work`，无卡、只判编译）；
host `release/.venv`（py3.12）跑全量回归。
**结论一句话**：ascend 探针已从 SimtVF(950 载体) 重写为 910B 合法最小件（T.copy 进 UB →
T.serial 标量乘二 → T.copy 出），容器 **`T-PROBE-ASC-COMPILE-PASS`**，monkeypatch 设备门后端到端
**`backend_available("ascend")=True`**；host **332 passed**（326 基线+6 新增）零回归，
host 无卡时 ascend 仍**因设备门**判 False（行为未改坏）。

## ① 完成情况（逐条附凭据）

- [x] 根因坐实·病灶 A（任务书所指 SimtVF 载体）：把旧探针 body **原样**（SimtVF/T.Parallel）
  绕开病灶 B 单独喂编译器 → 容器 `python3 /root/probe_simt.py` →
  ```
  SIMT-ISOLATE-FAIL: RuntimeError Ascend device compilation failed. | Command:
  /usr/local/Ascend/cann-8.5.0/aarch64-linux/ccec_compiler/bin/bisheng -O2 -fPIC -std=c++20
  -DTL_PORT910B_NATIVE_TYPES ...
  ```
  （与 P1-4 "SimtVF 件在 patched 910B tilelang 下 COMPILE-FAIL" 同型——载体病灶独立成立）
- [x] 根因坐实·病灶 B（本波新取证，**旧探针在 py3.10 环境死于构造期**）：改码前跑
  `E._compile_probe(tgt)` 全 traceback（`/root/probe_tb.py`）→
  ```
  File "/work/ascend/kernels/ascend_env.py", line 159, in _compile_probe
      def probe_asc(A: t.Tensor((64,), "float32"), ...):
  ...
  TypeError: Forward references must evaluate to types. Got buffer.
  ```
  ascend/cpu 两探针同型（line 159/178）。机理：本模块顶部有 `from __future__ import annotations`，
  def 处注解一律是字符串；tilelang eager builder 的 `get_type_hints` 走 `typing._eval_type`，
  py3.10 的 `_type_check` 对"前向引用求值结果不是类型"直接拒（py3.12 宽容，故 host 无恙）。
  → **修法必须连构造形态一起换**，否则容器（py3.10）判不了 PASS。
- [x] `_compile_probe` ascend 分支重写（唯一产品改动，`ascend_env.py` +22/−9 行，只动该分支）：
  `def probe_asc(A, C)` 无注解体 → 手工挂**即评** `__annotations__ = {"A": t.Tensor((64,),
  "float32"), "C": ...}` → `prog = t.prim_func(probe_asc)`（绕开字符串注解路）；body 为
  910B 已判决同族形态（参照 add_ln_asc_impl 的 copy 进 UB + serial 标量、gdn_conv_asc 的
  UB→GM `T.copy(out_ub[0:length], Y[...])` 搬出）：
  ```
  t.copy(A[0:64], a) ; for i in t.serial(64): c[i] = a[i] * 2 ; t.copy(c, C[0:64])
  ```
  验证（容器，改码后）：`cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 300 python3 /root/probe_env.py` →
  ```
  [TileLang] TileLang begins to compile kernel `probe_asc` with `out_idx=[1]`
  [TileLang] TileLang completes to compile kernel `probe_asc`
  T-PROBE-ASC-COMPILE-PASS
  ```
- [x] 门端到端判决（容器，monkeypatch 设备门，手法照 reconcile/verify_*.py）：
  `E.reset(); E._npu_present = lambda: True; E.backend_available("ascend")` →
  ```
  backend_available(ascend)= True blockers= {}
  T-GATE-ASC-AVAILABLE-PASS
  ```
  （旧代码同位置 False→九件内核全被旁路的门，现在开了；`get_compiled`/路由逻辑一行未动）
- [x] **cpu 探针与 TARGET_CPU 路一行未动**：`git diff` 仅 `_compile_probe` ascend 分支
  （+22/−9）；cpu 分支/`backend_available`/`active_backend`/`get_compiled` 零改动。
- [x] 新守卫 `tests/test_ascend_env_probe.py`（6 用例）：①静态守卫——ast 切 `if target ==
  TARGET_ASCEND:` 分支体，零 `SimtVF/Parallel` + 正向锚 `probe_asc.__annotations__`/
  `t.prim_func(probe_asc)`/禁 `@t.prim_func` 回装饰器形态（病灶 B 同防）；②cpu 探针 host 判真；
  ③双方言构造（monkeypatch tilelang.compile 不实编；ascend 在 host 只许败
  "op 未注册"环境账，不许败在载体/语法）；④host 无卡 ascend 仍 False 且败因=设备门。
  验证：`pytest tests/test_ascend_env_probe.py -v` → **6 passed**。
- [x] host 全量：`pytest tests -q -m "not integration"` → **332 passed, 3 skipped, 3 deselected**
  （= 改码前基线 326 passed + 本波新增 6，零回归；基线在改码前同命令实测）。
- [x] 写后三连：`wc -l` ascend_env.py=333 / test_ascend_env_probe.py=148；`py_compile` 双件 OK；
  分支体 grep `SimtVF|Parallel` 仅命中教训注释行（守卫切片从 def 起，不含之）。

## ② 错误与阻塞（含已解决的）

1. **构造期 TypeError 先于载体病灶**（病灶 B，见①-2）：容器旧探针根本没走到 bisheng 就死。
   → 已解决：ascend 分支改即评注解+显式 prim_func 形态。**代价如实报**：容器内 cpu 探针
   （`@t.prim_func`+字符串注解）仍有同型构造 TypeError——既有缺陷、H 代理已记档
   （verify_addln.py 头注"ascend_env 属禁改件"），本波按白名单**未动** cpu 路；待解禁时
   同法可修，host（py3.12）不受影响。
2. 容器判决不能用装饰器+字符串注解形态在 py3.10 通过（即 1），曾考虑过去 module 顶部
   `from __future__ import annotations` 一行——**否决**：改动半径覆盖全模块（含 cpu 分支与
   所有签名语义），违反白名单。
3. 试跑期小坑（已解决）：host 原型经 stdin 喂入致 `inspect.getsourcelines` 取不到源
   （eager builder 按 AST 变异需要真实文件）→ 落盘重试；`docker exec` heredoc 传 stdin 丢内容
   → 判决脚本改 `printf` 落容器 `/root/`（见③-2）。

## ③ 疑惑点与自行决策

- **白名单内但计划外的必要改动**：探针构造形态（装饰器→手工 `__annotations__` 挂载）。任务书
  只点名 SimtVF 载体；病灶 B 是容器取 FAIL 原文时新坐实的第二道死因，不修则"P1-1g 容器
  compile-PASS"判据不可能达成。属"探针 kernel 本体"射程，未越界。
- 容器判决脚本放 `/root/probe_env.py|probe_tb.py|probe_simt.py|probe_gate.py`（容器临时 fs），
  **不入仓**（白名单只有 test 与本文书三件产物）；复跑配方全文贴在下方⑤。
- `probe_asc` 保留 `t.Kernel(1) as bx` 与双 UB 缓冲（a 进 c 出）——与 cpu 探针严格同构、
  只换方言载体，遵守任务书"与 cpu 探针同构"条；未引入 `T.Vector()`（add_ln_asc 无它亦 PASS）。
- 静态守卫用"正向锚死构造形态"替代"只看有没有装饰器"（被否方案见测试 docstring）——两病灶
  同防，且不改产品代码风格测试就失效，防"实现被改、守卫失配"的漂移。

## ④ 偏离记录

- 计划"只换探针 kernel 本体、不用 SimtVF/Parallel" → 实际还换了**注解求值形态**（③-1 详述，
  判据驱动的必要修复；除该分支外零扩散）。
- 其余与任务书零偏离：cpu 路一行未动、`backend_available`/`get_compiled`/路由未动、
  未碰任何 `*_asc.py`/`*_kernel.py`/trunk/compile_all_asc.py/既有 tests。

## ⑤ 复跑配方（容器判决脚本）

```bash
# 判决件（探针直通）：/root/probe_env.py
printf '%s\n' 'import sys' 'sys.path.insert(0, "/work")' \
  'from ascend.kernels import ascend_env as E' 'try:' \
  '    E._compile_probe("ascend")' '    print("T-PROBE-ASC-COMPILE-PASS")' \
  'except Exception as e:' '    print("T-PROBE-ASC-COMPILE-FAIL")' \
  '    print("ERRTYPE:", type(e).__name__)' '    print("ERR:", str(e)[:1200])' > /root/probe_env.py
# 运行（三判决共用此前缀）：
docker exec cann910b-i bash -lc 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; export \
  BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201; \
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 300 python3 /root/probe_env.py'
# 门端到端：/root/probe_gate.py = E.reset(); E._npu_present=lambda: True;
#   ok=E.backend_available("ascend"); print("T-GATE-ASC-AVAILABLE-PASS" if ok else "FAIL")
# host：cd release && .venv/bin/python -m pytest tests -q -m "not integration"   # 332 passed
```
