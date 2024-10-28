import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import ensure_contiguous, is_hip


def get_amd_triton_config_list():

    waves_per_eu = [0, 1, 2]
    matrix_instr_nonkdim = [16, 32]
    num_stages = [0, 1, 2]
    num_warps = [4, 8, 16]

    config_list = []

    for wpe in waves_per_eu:
        for kdim in matrix_instr_nonkdim:
            for ns in num_stages:
                for nw in num_warps:
                    config_list.append(
                        triton.Config(
                            {
                                "waves_per_eu": wpe,
                                "matrix_instr_nonkdim": kdim,
                            },
                            num_stages=ns,
                            num_warps=nw,
                        )
                    )
    return config_list


def get_nvidia_triton_config_list():

    return [
        triton.Config(
            {},
            num_warps=32,
        )
    ]


@triton.autotune(
    configs=(
        get_amd_triton_config_list() if is_hip() else get_nvidia_triton_config_list()
    ),
    key=["embedding_dim", "BLOCK_SIZE_M", "BLOCK_SIZE_N"],
)
@triton.jit
def embedding_forward_kernel(
    embeddings_ptr,
    indices_ptr,
    output_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < n_elements
    indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < embedding_dim

    embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]
    embeddings = tl.load(
        embeddings_ptr + embedding_offsets,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )

    output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
    tl.store(
        output_ptr + output_offsets, embeddings, mask=mask_m[:, None] & mask_n[None, :]
    )


def get_amd_triton_config_list():

    waves_per_eu = [0, 1, 2]
    matrix_instr_nonkdim = [16, 32]
    num_stages = [0, 1, 2]
    num_warps = [4, 8, 16]

    config_list = []

    for wpe in waves_per_eu:
        for kdim in matrix_instr_nonkdim:
            for ns in num_stages:
                for nw in num_warps:
                    config_list.append(
                        triton.Config(
                            {
                                "waves_per_eu": wpe,
                                "matrix_instr_nonkdim": kdim,
                            },
                            num_stages=ns,
                            num_warps=nw,
                        )
                    )
    return config_list


def get_nvidia_triton_config_list():

    return [
        triton.Config(
            {},
            num_warps=32,
        )
    ]


@triton.autotune(
    configs=(
        get_amd_triton_config_list() if is_hip() else get_nvidia_triton_config_list()
    ),
    key=["embedding_dim", "BLOCK_SIZE_M", "BLOCK_SIZE_N"],
)
@triton.jit
def embedding_backward_kernel(
    grad_output_ptr,
    grad_weight_ptr,
    indices_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    mask_m = offsets_m < n_elements
    indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0)
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offsets_n < embedding_dim

    grad_output = tl.load(
        grad_output_ptr + offsets_m[:, None] * embedding_dim + offsets_n[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )

    grad_weight_offsets = indices[:, None] * embedding_dim + offsets_n[None, :]

    tl.atomic_add(
        grad_weight_ptr + grad_weight_offsets,
        grad_output,
        mask=mask_m[:, None] & mask_n[None, :],
    )


class LigerEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, embeddings: torch.Tensor, indices: torch.Tensor):
        ori_shape = indices.shape
        indices = indices.view(-1)
        output = torch.empty(
            indices.shape[0],
            embeddings.shape[1],
            device=indices.device,
            dtype=embeddings.dtype,
        )

        n_elements = indices.numel()
        embedding_dim = embeddings.shape[1]

        BLOCK_SIZE_M = triton.next_power_of_2(min(128, embedding_dim))
        BLOCK_SIZE_N = triton.next_power_of_2(min(128, embedding_dim))
        grid = (
            triton.cdiv(n_elements, BLOCK_SIZE_M),
            triton.cdiv(embedding_dim, BLOCK_SIZE_N),
        )

        embedding_forward_kernel[grid](
            embeddings,
            indices,
            output,
            n_elements,
            embedding_dim=embedding_dim,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        ctx.save_for_backward(indices, embeddings)

        return output.view(*ori_shape, -1)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_output: torch.Tensor):
        indices, embedding_table = ctx.saved_tensors
        grad_output = grad_output.contiguous().view(-1, embedding_table.shape[1])

        grad_weight = torch.zeros_like(embedding_table)

        n_elements = indices.numel()
        embedding_dim = embedding_table.shape[1]

        BLOCK_SIZE_M = triton.next_power_of_2(min(128, embedding_dim))
        BLOCK_SIZE_N = triton.next_power_of_2(min(128, embedding_dim))
        grid = (
            triton.cdiv(n_elements, BLOCK_SIZE_M),
            triton.cdiv(embedding_dim, BLOCK_SIZE_N),
        )

        embedding_backward_kernel[grid](
            grad_output,
            grad_weight,
            indices,
            n_elements,
            embedding_dim=embedding_dim,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        return grad_weight, None
