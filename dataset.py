from __future__ import annotations
import torch
from torch.utils.data import Dataset
from model import GPTConfig
from typing import Tuple, List




class TokenDataset(Dataset):
    def __init__(self, model_args: GPTConfig, input_ids: List) -> None:
        """
        Initializes the TokenDataset.

        Args:
            model_args: An instance of ModelArgs containing model configuration
                parameters, including the maximum sequence length.
            input_ids: A tensor containing tokenized input data.

        Attributes:
            input_ids: Stores the tokenized input data.
            block_size: The block size for dividing the input data, determined by
                the maximum sequence length in model_args.
        """
        self.input_ids = input_ids
        self.block_size = model_args.block_size

    def __len__(self) -> int:
        """
        Returns the number of blocks in the dataset.

        Since the input_ids are divided into blocks of size block_size, the number of
        blocks is calculated as the length of the input_ids minus one, divided by the
        block size.
        """
        return (len(self.input_ids) - 1) // self.block_size

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:     
        """
        Returns a tuple of two tensors, x and y, where x is the input tensor slice
        and y is the output tensor slice. The slices are of size block_size, and are
        taken from the input_ids tensor at the given index.

        Args:
            idx: The index of the block to retrieve.

        Returns:
            A tuple of two tensors, x and y.
        """
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        
        x = self.input_ids[start_idx:end_idx]
        y = self.input_ids[start_idx+1:end_idx+1]
        
        return torch.LongTensor(x), torch.LongTensor(y)