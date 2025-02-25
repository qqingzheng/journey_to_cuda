import torch

import triton
import triton.language as tl

device = torch.device("cuda")

@triton.jit
def matmul_kernel(
    output_ptr,
    a_ptr,
    b_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(m, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n, BLOCK_SIZE_N)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    
    real_group_size_m = min(GROUP_SIZE_M, num_pid_m - first_pid_m)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % real_group_size_m)
    pid_n = ((pid % num_pid_in_group) // real_group_size_m)
    
    
    offset_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offset_m[:, None] * stride_am + offset_k[None, :] * stride_ak
    b_ptrs = b_ptr + offset_k[:, None] * stride_bk + offset_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k_ in range(tl.cdiv(k, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offset_k[None, :] + k_ * BLOCK_SIZE_K < k, other=0.0)
        b = tl.load(b_ptrs, mask=offset_k[:, None] + k_ * BLOCK_SIZE_K < k, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    output_ptrs = output_ptr + offset_m[:, None] * stride_cm + offset_n[None, :] * stride_cn
    tl.store(output_ptrs, acc)
    
    

def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    GROUP_SIZE_M = 16
    BLOCK_SIZE_K = 128
    
    grid = lambda META: (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    matmul_kernel[grid](
        c, a, b, 
        M, N, K, 
        a.stride(0), a.stride(1), 
        b.stride(0), b.stride(1), 
        c.stride(0), c.stride(1), 
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    return c

if __name__ == "__main__":
    a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    b = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    c = matmul(a, b)
    print(torch.allclose(c, a @ b))