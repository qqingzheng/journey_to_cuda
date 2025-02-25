import torch

import triton
import triton.language as tl

device = torch.device("cuda")

@triton.jit
def flash_attention_fwd_kernel(
    output_ptr,
    q_i,
    k_j,
    v_j,
    l_i,
    m_i
):
    q_i = tl.load(q)
    k_j = tl.load(k)
    v_j = tl.load(v)
    pass


def flash_attention_fwd(
    q,
    k,
    v,
):
    B_r = 5
    B_c = 5
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    pass

if __name__ == "__main__":
    q = torch.randn(10, 16, device=device)
    k = torch.randn(10, 16, device=device)
    v = torch.randn(10, 16, device=device)
    flash_attention_fwd(q, k, v)
