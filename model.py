from __future__ import annotations
from dataclasses import dataclass
import gin
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F


@gin.configurable
@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int 
    n_embd: int 
    batch_size: int 
    learning_rate: float
    weight_decay: float
    eps: float
    betas: Tuple[float, float]
    seed: int
    epochs: int
    training_backend: str
    device: str
    model_name: str
    clip_grad_norm_val: float
    dtype: torch.dtype = torch.bfloat16


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """
        Initializes the RMSNorm module.

        Args:
            dim: The dimension of the input tensor.
            eps: The epsilon value used to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x) -> torch.Tensor:
        """
        Computes the RMSNorm of a tensor.

        Given an input tensor `x`, compute its RMSNorm by dividing it by the root
        mean square of its elements.

        Args:
            x: The input tensor.

        Returns:
            The RMSNorm of the input tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x) -> torch.Tensor:        
        """
        Computes the RMSNorm of a tensor and applies a learnable scale factor.

        Args:
            x: The input tensor.

        Returns:
            The RMSNorm of the input tensor multiplied by a learnable scale factor.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        """
        Initializes the CausalSelfAttention module.

        Args:
            config: The GPTConfig.

        Notes:
            The `config` object should have the following attributes:

                - `n_head`: The number of attention heads.
                - `n_embd`: The dimension of the input tensor.
        """
        super().__init__()
        self.config = config
        # check that the dimension is a multiple of n_head
        assert self.config.n_embd % self.config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.config.n_embd, 3 * self.config.n_embd)
        # output projection
        self.c_proj = nn.Linear(self.config.n_embd, self.config.n_embd)
        
    def forward(self, x):
        """
        Computes the causal self-attention of the input tensor.

        Given an input tensor `x` of shape `(B, T, C)`, computes its causal self-attention
        using the `scaled_dot_product_attention` function from PyTorch.

        Args:
            x: The input tensor.

        Returns:
            The causal self-attention of the input tensor.
        """
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.config.n_embd, dim=2)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) 
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) 
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2) 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) 
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        """Initializes the MLP module.

        Args:
            config: The GPTConfig.

        Notes:
            The `config` object should have the following attributes:

                - `n_embd`: The dimension of the input tensor."""
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        """Computes the feedforward neural network of the input tensor.

        Given an input tensor `x` of shape `(B, T, C)`, computes its feedforward neural
        network using a linear layer, a GeLU activation function, and another linear layer.

        Args:
            x: The input tensor.

        Returns:
            The feedforward neural network of the input tensor."""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        """Initializes the Block module.

        Given a GPTConfig object, initializes the Block module, which consists of
        a causal self-attention layer, a layer normalization layer, a feedforward
        neural network layer, and another layer normalization layer."""
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        """Computes the block of the transformer.

        Given an input tensor `x` of shape `(B, T, C)`, computes its block using a
        causal self-attention layer, a layer normalization layer, a feedforward
        neural network layer, and another layer normalization layer.

        Args:
            x: The input tensor.

        Returns:
            The block of the input tensor."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        """Initializes the GPT module.

        Given a GPTConfig object, initializes the GPT module, which consists of an
        embedding layer, a positional embedding layer, a stack of transformer
        blocks, a layer normalization layer, and a linear layer for generating
        logits. The weights of the embedding layer are shared with the weights of
        the linear layer.

        Args:
            config: The GPTConfig object."""
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight


    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        """
        Computes the forward pass of the GPT model.

        Given an input tensor `idx` of shape `(B, T)`, computes its forward pass
        using the transformer blocks, layer normalization, and a linear layer for
        generating logits.

        Args:
            idx: The input tensor.

        Returns:
            The forward pass of the input tensor.

        Notes:
            The `config` object should have the following attributes:

                - `n_embd`: The dimension of the input tensor.
                - `vocab_size`: The size of the vocabulary.
                - `block_size`: The maximum block size of the input tensor.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss 