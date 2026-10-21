"""letter_readout 探针：抽行对 torch.index_select、回加对 index_add_，并与 autograd 交叉验证。"""
import torch

from ascend.kernels import letter_readout_asc, letter_readout_kernel, ascend_env

torch.manual_seed(3)
TOTAL, DIM, PICK = 17, 8, 5
rows = torch.randn(TOTAL, DIM)
ids = torch.randint(0, TOTAL, (PICK,), dtype=torch.int32)
ids[1] = ids[0]  # 故意重复点同一行：反向必须累加而不是覆盖
print("ids", ids.tolist())

out = letter_readout_asc.forward(rows, ids, torch.float32, "cpu")
print("gather_err", float((out - rows[ids]).abs().max()))

dy = torch.randn(PICK, DIM)
acc = letter_readout_asc.backward(TOTAL, ids, dy, "cpu")
ref = torch.zeros(TOTAL, DIM).index_add_(0, ids.long(), dy)
print("scatter_err", float((acc - ref).abs().max()))

# 交叉验证：本件的反向 == torch 自动微分对同一条 gather 的梯度
r2 = rows.clone().requires_grad_(True)
g = torch.autograd.grad((r2[ids] * dy).sum(), r2)[0]
print("autograd_err", float((letter_readout_asc.backward(TOTAL, ids, dy, "cpu") - g).abs().max()))

e = letter_readout_kernel.forward(rows, ids, torch.float16, "cpu")
print("entry_dtype", e.dtype, float((e.float() - rows[ids]).abs().max()))
print("nc", ascend_env.compile_count(), "blk", ascend_env.blockers(),
      "keys", sorted(ascend_env.compiled_keys()))
