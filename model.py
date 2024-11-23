from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import gin
import inspect
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.nn import functional as F
from rope import *

@gin.configurable
@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int 
    n_embd: int
    attn_dropout: float
    batch_size: int
    epochs: int
    clip_grad_norm_val: float
    training_backend: str
    learning_rate: float
    weight_decay: float
    eps: float
    betas: Tuple[float, float]
    base_theta: float
    scale_factor: float
    dtype: torch.dtype = torch.bfloat16
    fused_optimizer: bool = "fused" in inspect.signature(torch.optim.AdamW).parameters
    do_init_params: Optional[bool] = False
    rng_seed: Optional[int] = 42
    rng_device: str|torch.device = torch.device("cpu")
    model_device: Optional[str|torch.device] = torch.device("cpu")
    rng_generator: Optional[torch.Generator] = None

    def __post_init__(self) -> None:
        self.head_dim = self.n_embd // self.n_head
        self.intermediate_size = 4 * self.n_embd   
        if self.rng_generator is None:
            self.rng_generator = torch.Generator(device=self.rng_device)     


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        """
        Initializes the RMSNorm module.

        Args:
            dim: The dimension of the input tensor.
            eps: The epsilon value used to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

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
        return self._norm(x.float()).type_as(x) * self.weight + self.bias
    

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:

        super().__init__()
        self.config = config
        # check that the dimension is a multiple of n_head
        assert self.config.n_embd % self.config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.qkv_proj = nn.Linear(self.config.n_embd, 3 * self.config.n_embd, bias=True)
        # output projection
        self.o_proj = nn.Linear(self.config.n_embd, self.config.n_embd, bias=True)
        self.o_proj.SCALE_INIT = 1
        
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # query, key, value projections  
        qkv = self.qkv_proj(x)

        # splitting q, k, v
        q, k, v = qkv.split(self.config.n_embd, dim=2)

        # reshape q, k, v for multiple heads
        q = q.view(B, T, self.config.n_head, C // self.config.n_head)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head)  
        v = v.view(B, T, self.config.n_head, C // self.config.n_head)

        # apply rotary embeddings to q and k
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # reshape q, k, v for attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # scaled dot product attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.config.attn_dropout) 

        # reshape for output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
    
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:

        super().__init__()
        self.config = config

        # projections 
        self.gate_proj = nn.Linear(self.config.n_embd, self.config.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.config.n_embd, self.config.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.config.intermediate_size, self.config.n_embd, bias=True)
        self.down_proj.SCALE_INIT = 1

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
    

class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.ln_1 = RMSNorm(config.n_embd, eps=config.eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.eps)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis: torch.Tensor) -> torch.Tensor:   
        # attention block
        x = x + self.attn(self.ln_1(x), freqs_cis=freqs_cis)
        # mlp block
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, eps=config.eps)
        ))


        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # precompute the frequencies for the rotary embeddings
        self.freqs_cis = precompute_freqs_cis(dim=self.config.head_dim,
                                              end=self.config.block_size,
                                              theta=self.config.base_theta,
                                              scale_factor=self.config.scale_factor
                                              )
        
        if config.do_init_params and config.rng_generator is not None:
            # initialize the parameters
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.config.rng_generator.manual_seed(self.config.rng_seed))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.config.rng_generator.manual_seed(self.config.rng_seed))
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        # optim groups for weight decay and no weight decay
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # counts of parameters that will be decayed or not
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        # print parameter count info
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer and use the fused version if it is available
        print(f"using fused AdamW: {self.config.fused_optimizer}")
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.learning_rate, betas=self.config.betas, eps=self.config.eps, fused=self.config.fused_optimizer)

        return optimizer
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # idx is of shape (B, T)
        _, T = idx.size()

        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # forward the token 
        x = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        self.freqs_cis = self.freqs_cis.to(x.device)
        
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x, freqs_cis=self.freqs_cis)
        
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        # compute the loss if targets are provided
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss # (B, T, vocab_size), loss (if targets are provided)
    

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super(CosineWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1  # Start from 0
        if current_step < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * current_step / self.warmup_steps for base_lr in self.base_lrs
            ]
        else:
            # Cosine decay
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [
                self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + torch.cos(torch.pi * progress))
                for base_lr in self.base_lrs
            ]

#  CosineWarmupScheduler       
# from torch.optim.lr_scheduler import LambdaLR
# # Scheduler definition
# scheduler = LambdaLR(optimizer, lr_lambda)

# def lr_lambda(current_step: int):
#     if current_step < warmup_steps:
#         return current_step / warmup_steps  # Linear warmup
#     else:
#         progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
#         return 0.5 * (1 + math.cos(math.pi * progress))  # Cosine decay