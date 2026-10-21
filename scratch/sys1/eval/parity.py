"""sys1/eval/parity.py — 跨后端一致轴：同一份权重走内核路与走纯 torch 路，答案不许换人。

【做什么】
    把同一个模型的同一道题算两遍。一遍走 `sys1/kernels/backends.py` 的分发（本机有
    Metal 方言就用方言，用不了就由入口层回退到普通写法，报告如实标注这次走了哪条），
    一遍走模型自带的纯 torch 前向。两遍都只从末位那一格读各候选的胆量分，然后比名次：
    一致率必须百分之百，另附逐元素最大偏差与"这条路上到底真开了几个模具"的凭据。

【怎么做】
    内核路是把手上的四件模具拼成一层 pre-norm 块：查表得到的起始数字先过一次"残差加＋
    整形"（`add_ln`，把上一层的加法与本层的整形融成一次），整形结果进 `gemm` 打成一份
    打包的 qkv，就地 `rope` 转前两槽，摊成三摞后交 `attn_sw` 做只看左边的滑窗汇总，
    回投再过 `gemm`；接着又是"残差加＋整形"（融的是下一段整形），升宽 `gemm`、
    gelu 关卡、压回 `gemm`，最后一次"残差加＋整形"把残差留在 fp32、把整形结果送下一块。
    两处细节是刻意的：① 长度先补到 16 的整块（`attn_sw` 的方言要求行数是块高的整倍），
    补出来的行只做键的邻居、因果序下真实位置看不见"后面"，所以前若干行的读数与不补
    时逐位相同，出口按真长度切掉；② 残差流全程 fp32，只有喂给模具的那一份降到 fp16——
    与 p1-04 的精度纪律同口径。两路末位数字都在 CPU 上点 26 字母行，比的是同一条读出。
    被表达的口径缺口（gelu 不在模具支持的截断表里）不硬凑：那一格由 torch 侧补做，
    报告里以 `fallback_ops` 列出哪些算子这次走了普通写法。

【为什么】
    这一轴是 M1 的最终验收：单个算子对得上不代表串起来还对，误差会顺着残差流堆到末位
    读数上；而决策业务只认一件事——名次不能换人。所以判据取 argmax 一致率（100%）而不是
    均值偏差，另报最大偏差供人看幅度。
    被否方案一：只在算子层两两对拍、模型层不再比——p1-04 已做该层，本轴若重复就等于
    没有覆盖"组装后的漂移"，而组装漂移正是 fp16 残差降位宽这类事故唯一的暴露面。
    被否方案二：把"回退到普通写法"当成通过（毕竟两边都是 torch，一致率必为 100%）——
    那是自证式的假通过；故本轴强制附 `backend`/`compiled_keys`/`compile_count`/`blockers`
    四个凭据，走了几次真模具、有没有失败登记，一目了然（与 kernels/README 的对拍纪律同源）。
    被否方案三：允许一致率 99% 并注明"个别样本 fp16 抖了"——一阶段的红线写的是 100%，
    把阈值松一格，报告就再也分不清"内核没错"与"错得少"。
"""
from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from sys1.decision import option_scores
from sys1.kernels import add_ln_kernel, attn_sw_kernel, backends, gemm_kernel, rope_kernel
from sys1.model import Decoder, rope_tables

#: 方言侧的整块行数（attn 的块高/键块宽、add_ln 的行块都是 16），补齐到这里就不需要边界守卫
LADDER = 16
#: 内核路里这次没走模具、由 torch 侧补做的算子（写进报告，不与"全走了模具"混报）
EXTERNAL_OPS = ("gelu",)


def _field(ex: Any, key: str, default: Any = None) -> Any:
    """样本字段取用两栖：dict 按键取，`EncodedSample` 这类对象按属性取（少一层转换噪声）。"""
    if isinstance(ex, dict):
        return ex.get(key, default)
    return getattr(ex, key, default)


def _pad_to_ladder(length: int, ladder: int = LADDER) -> int:
    """把行数抬到整块倍数（已是倍数则原样返回）。"""
    return ((length + ladder - 1) // ladder) * ladder


def _h16(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """权重降到 fp16 并搬到目标设备（模具的操作数只认这个位宽）。"""
    return t.detach().to(torch.float16).to(device).contiguous()


def _f32(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """偏置/整形参数保 fp32（与 p1-04 的精度纪律一致：累加与残差流不降位宽）。"""
    return t.detach().to(torch.float32).to(device).contiguous()


class KernelForward:
    """把已载好的模型权重摊成"模具能吃"的一套参数，并按四件模具串出一层块前向。

    它不是第二个模型实现：所有数字都从传进来的 `Decoder` 现取，形状与角度表也按那份
    config 算，只是把逐块计算交给 `sys1/kernels/*_kernel.py` 的入口（由 backends 决定
    用方言还是普通写法）。
    """

    def __init__(self, model: Decoder, device: torch.device | str) -> None:
        """按模型 config 备好角度表与 fp16/fp32 两套参数（同一次评测只备一遍，逐样本复用）。"""
        self.device = torch.device(device)
        cfg = model.config
        self.cfg = cfg
        self.blocks = list(model.blocks)
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim
        self.tables: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._w: dict[str, torch.Tensor] = {
            "tok_emb": _f32(model.tok_emb.weight, self.device),
            "final_w": _f32(model.final_norm.weight, self.device),
            "final_b": _f32(model.final_norm.bias, self.device),
        }
        for i, block in enumerate(model.blocks):
            pre = f"b{i}."
            self._w[pre + "norm1_w"] = _f32(block.norm1.weight, self.device)
            self._w[pre + "norm1_b"] = _f32(block.norm1.bias, self.device)
            self._w[pre + "norm2_w"] = _f32(block.norm2.weight, self.device)
            self._w[pre + "norm2_b"] = _f32(block.norm2.bias, self.device)
            self._w[pre + "qkv_w"] = _h16(block.attn.qkv.weight, self.device)
            self._w[pre + "qkv_b"] = _f32(block.attn.qkv.bias, self.device)
            self._w[pre + "proj_w"] = _h16(block.attn.proj.weight, self.device)
            self._w[pre + "proj_b"] = _f32(block.attn.proj.bias, self.device)
            self._w[pre + "up_w"] = _h16(block.mlp.up.weight, self.device)
            self._w[pre + "up_b"] = _f32(block.mlp.up.bias, self.device)
            self._w[pre + "down_w"] = _h16(block.mlp.down.weight, self.device)
            self._w[pre + "down_b"] = _f32(block.mlp.down.bias, self.device)

    def _angles(self, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按补齐后的行数取 cos/sin 两张 fp32 角度表（同一长度只算一次，逐块逐样本复用）。"""
        hit = self.tables.get(rows)
        if hit is None:
            hit = rope_tables(rows, self.head_dim, self.cfg.rope_theta, device=self.device)
            self.tables[rows] = hit
        return hit

    @torch.no_grad()
    def last_hidden(self, input_ids: list[int], pad_id: int) -> torch.Tensor:
        """编号串进、末位那串数字出（`(T, d)` 的 fp32），全程走 kernels 入口层的分发。

        白话：把这串编号查成一行行数字，然后一层层加工——每一层都先把上一笔账合进底账、
        再把底账摆整齐送进机器；机器算完的位置只看它自己和它左边的那些，写完最后一格
        留下的那串数字就是交出去的东西。多垫的那几行只用来凑整块，收尾时一刀切掉。
        """
        seq = len(input_ids)
        rows = _pad_to_ladder(seq)
        ids = torch.full((rows,), pad_id, dtype=torch.long, device=self.device)
        ids[:seq] = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        cos, sin = self._angles(rows)
        half = self.head_dim // 2
        eps = self.cfg.norm_eps

        h = self._w["tok_emb"][ids]                                   # (rows, d) fp32 底账
        y, h = add_ln_kernel.forward(h.to(torch.float16), torch.zeros_like(h),
                                     self._w["b0.norm1_w"], self._w["b0.norm1_b"], eps=eps)
        for i in range(self.cfg.L):
            pre = f"b{i}."
            qkv = gemm_kernel.forward(y, self._w[pre + "qkv_w"], self._w[pre + "qkv_b"],
                                      act="none").view(rows, 3, self.heads, self.head_dim).contiguous()
            qkv = rope_kernel.forward(qkv, cos.reshape(rows, half), sin.reshape(rows, half))
            q, k, v = (qkv[:, slot].transpose(0, 1).contiguous() for slot in range(3))
            ctx = attn_sw_kernel.forward(q, k, v, window=rows).transpose(0, 1).reshape(rows, self.cfg.d)
            att = gemm_kernel.forward(ctx, self._w[pre + "proj_w"], self._w[pre + "proj_b"], act="none")
            y, h = add_ln_kernel.forward(att, h, self._w[pre + "norm2_w"], self._w[pre + "norm2_b"], eps=eps)
            up = gemm_kernel.forward(y, self._w[pre + "up_w"], self._w[pre + "up_b"], act="none")
            # gelu 不在模具已落地的截断口径里（gemm_mps.SUPPORTED_ACTS 只有 none/relu）：由 torch 侧补做
            up = F.gelu(up.to(torch.float32), approximate="tanh").to(torch.float16)
            down = gemm_kernel.forward(up, self._w[pre + "down_w"], self._w[pre + "down_b"], act="none")
            last = i == self.cfg.L - 1
            nxt_w = self._w["final_w"] if last else self._w[f"b{i + 1}.norm1_w"]
            nxt_b = self._w["final_b"] if last else self._w[f"b{i + 1}.norm1_b"]
            y, h = add_ln_kernel.forward(down, h, nxt_w, nxt_b, eps=eps,
                                         out_dtype=torch.float32 if last else torch.float16)
        return y[:seq].to(torch.float32).cpu()

@torch.no_grad()
def torch_last_hidden(model: Decoder, input_ids: list[int], device: torch.device) -> torch.Tensor:
    """纯 torch 路：模型自己的前向（fp32，含因果与补洞屏蔽），交 `(T, d)` 的末位数字。

    白话：完全不碰那些机器模具，就用模型自带的算法把同样的编号走一遍，取同样位置的
    那串数字当参照——这一路是本轴比较的基准。
    """
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    return model(ids, None)[0].to(torch.float32).cpu()


def readout_scores(last_hidden: torch.Tensor, head_weight: torch.Tensor, letter_ids: list[int]) -> list[float]:
    """末位数字 × 26 字母行 = 各候选的胆量分（两路共用同一条读出，杜绝口径分叉）。

    白话：拿写完最后一格留下的那串数字，去点每个候选字母对应的那几行，点出来的大小
    就是各候选的胆量分。内核路与普通路都必须用这同一个点法，否则比出来的名次差就说不
    清是模具带来的还是点法不同带来的。
    """
    return option_scores(last_hidden.unsqueeze(0), head_weight, letter_ids, None)[0].tolist()


def run_parity(model: Decoder, examples: list[dict[str, Any]], *, device: str | torch.device = "mps",
               head_weight: torch.Tensor | None = None, timeout_ms: float | None = None) -> dict[str, Any]:
    """逐样本跑两条路并汇总一致率、最大偏差与后端凭据（100% 是本轴的合格线）。

    参数 `examples` 每项需带 `input_ids / letter_ids / length`（预测出口编码好的样本，
    dict 或 `EncodedSample` 都收）；
    `head_weight` 缺省取模型自己的输出层权重（CPU 副本，与预测出口的读数落点一致）。
    `timeout_ms` 是给 CI 的软上限：单样本两路耗时超过它只作记录，不判失败。

    白话：同一道题喂两遍——一遍用机器模具算，一遍用普通写法算，各自在写完的最后一格
    收几个候选的胆量分，再看两边各排第几的名次有没有换人；同时记下两边数值最大差了多少、
    这次模具真开过几个、有没有开不动而改用手写的地方，免得"两边都手写"冒充成"模具通过"。
    """
    dev = backends.resolve_device(device)
    model.to(dev)                    # 两路必须同设备：模具路在 dev 上，torch 路也得在 dev 上
    hw = (model.lm_head.weight.detach().cpu() if head_weight is None else head_weight.detach().cpu())
    kernel = KernelForward(model, dev)
    pad_id = int(model.config.pad_token_id)
    mode = backends.active_backend(dev)
    before = backends.compile_count()
    t0 = time.perf_counter()

    rows: list[dict[str, Any]] = []
    agree = 0
    max_diff = 0.0
    for ex in examples:
        ids = list(_field(ex, "input_ids"))
        length = int(_field(ex, "length", len(ids)))
        ids = ids[:length]
        letter_ids = list(_field(ex, "letter_ids"))
        k_h = kernel.last_hidden(ids, pad_id)
        k_scores = readout_scores(k_h, hw, letter_ids)
        t_h = torch_last_hidden(model, ids, dev)
        t_scores = readout_scores(t_h, hw, letter_ids)
        pick_k = max(range(len(k_scores)), key=lambda i: (k_scores[i], -i))
        pick_t = max(range(len(t_scores)), key=lambda i: (t_scores[i], -i))
        diff = max(abs(a - b) for a, b in zip(k_scores, t_scores))
        max_diff = max(max_diff, diff)
        agree += int(pick_k == pick_t)
        rows.append({"id": _field(ex, "id"), "n": len(k_scores), "kernel_pick": pick_k, "torch_pick": pick_t,
                     "same": pick_k == pick_t, "max_abs_err": float(diff)})

    elapsed = (time.perf_counter() - t0) * 1000.0
    n = len(rows)
    dialect_ran = backends.compile_count() - before
    report = {
        "axis": "parity",
        "device": str(dev),
        "backend": mode,                                   # tilelang（有模具）/ torch_eager（全回退）
        "consistent": agree,
        "total": n,
        "argmax_agreement": agree / n if n else 0.0,
        "passed": bool(n) and agree == n,
        "max_abs_err_scores": float(max_diff),
        "fp16_tolerance": 2e-2,
        "dialect_compiles_this_run": dialect_ran,
        "compiled_keys": sorted(backends.compiled_keys()),
        "blockers": dict(backends.blockers()),
        "external_ops": list(EXTERNAL_OPS),
        "precision": "residual=fp32, dialect operands=fp16",
        "elapsed_ms": round(elapsed, 3),
        "per_sample": rows,
    }
    if timeout_ms is not None:
        report["per_sample_budget_ms"] = round(elapsed / max(1, n), 3)
    return report


def format_table(report: dict[str, Any]) -> str:
    """把对拍结论压成两行摘要（第一行是合格判定，第二行是幅度与凭据）。

    白话：给人一眼看得懂的两句：第一句说"几道题、名次换没换人、过没过"，第二句说
    "这回用的是哪条路、数值最大差多少、模具开了几个、有没有偷偷回退"。
    """
    head = (f"parity  argmax 一致 {report['consistent']}/{report['total']} "
            f"({report['argmax_agreement']:.1%})  passed={report['passed']}")
    detail = (f"        device={report['device']} backend={report['backend']} "
              f"max|err|={report['max_abs_err_scores']:.3e} compiles=+{report['dialect_compiles_this_run']} "
              f"blockers={len(report['blockers'])} external_ops={','.join(report['external_ops'])}")
    return f"{head}\n{detail}"


__all__ = [
    "EXTERNAL_OPS",
    "LADDER",
    "KernelForward",
    "format_table",
    "readout_scores",
    "run_parity",
    "torch_last_hidden",
]
