from __future__ import annotations
import torch
from torch.utils.data import Dataset
from typing import Tuple
from numpy import ndarray

class TokenDataset(Dataset): 
    """
    A  dataset for tokenized input data.
    """
    def __init__(self, block_size: int, input_ids: ndarray, pad_token_id: int) -> None:
        """
        Initializes the TokenDataset.

        Args:
            block_size: The block size for dividing the input data.
            input_ids: A list containing tokenized input data.
            pad_token_id: The token ID to use for padding.
        """
        self.block_size = block_size
        self.input_ids = input_ids.tolist()
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        If the length of the input_ids is divisible by the block size, it returns the
        length of the input_ids divided by the block size. Otherwise, it returns the
        length of the input_ids after padding to the block size.

        Returns:
            The length of the dataset.
        """
        if (len(self.input_ids) - 1) % self.block_size == 0:
            print("The length of the input_ids is divisible by the block size")
            return (len(self.input_ids) - 1) // self.block_size
        
        else:
            remainder = (len(self.input_ids) - 1) % self.block_size
            padding_length = self.block_size - remainder
            self.input_ids.extend([self.pad_token_id] * padding_length)
            print("The length of the input_ids is not divisible by the block size")
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
        
        # Return the input and output slices
        return torch.tensor(self.input_ids[start_idx:end_idx], dtype=torch.long), torch.tensor(self.input_ids[start_idx+1:end_idx+1], dtype=torch.long)