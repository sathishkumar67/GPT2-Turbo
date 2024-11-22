import torch
from typing import Tuple



def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Precompute the frequencies for the rotary position embedding.

    Args:
        dim: The last dimension of the input tensor.
        end: The first dimension of the input tensor.
        theta: The temperature in the computation of the frequencies.

    Returns:
        A tensor of shape `(end, dim)` with the precomputed frequencies.
    """
    freqs = scale_factor / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.bfloat16) # here we use bfloat16
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Reshape the precomputed frequencies for rotary embeddings to be broadcasted to x.

    The shape of freqs_cis is (seq_len, dim) and the shape of x is (batch_size, seq_len, dim).
    We want to reshape freqs_cis to be (1, seq_len, 1, dim) so that it can be broadcasted to
    x.

    Args:
        freqs_cis: The precomputed frequencies for the rotary embeddings.
        x: The tensor to which the rotary embeddings will be applied.

    Returns:
        The reshaped frequencies.
    """
    assert 0 <= 1 < x.ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    return freqs_cis.view(*[d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x.shape)])


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the rotary embeddings to the query and key tensors.

    The rotary embeddings are precomputed using the `precompute_freqs_cis` function.
    The query and key tensors are reshaped to have shape (batch_size, seq_len, dim),
    and the precomputed frequencies are reshaped to have shape (1, seq_len, 1, dim)
    so that they can be broadcasted to the query and key tensors.

    The rotary embeddings are applied by element-wise multiplying the query and key
    tensors with the precomputed frequencies.

    Args:
        xq: The query tensor.
        xk: The key tensor.
        freqs_cis: The precomputed frequencies.

    Returns:
        The query and key tensors after applying the rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
