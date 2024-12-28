from __future__ import annotations
import torch
import numpy as np
from typing import Tuple, List
from numpy import ndarray
from torch.utils.data import Dataset



def process_input_ids(input_ids: ndarray, block_size: int, pad_token_id: int) -> List[int]:
    """
    Processes the input_ids to ensure their length is divisible by block_size.

    Args:
        input_ids (list[int]): The list of input IDs.
        block_size (int): The size of the blocks.
        pad_token_id (int): The ID used for padding.

    Returns:
        list[int]: The processed input IDs.
    """
    # convert input_ids to a list
    input_ids = input_ids.tolist()

    # check if the length of the input_ids is divisible by the block size
    if (len(input_ids) - 1) % block_size == 0:
        print("The length of the input_ids is divisible by the block size.")
        return input_ids
    else:
        remainder = (len(input_ids) - 1) % block_size
        padding_length = block_size - remainder
        input_ids.extend([pad_token_id] * padding_length)
        print("The length of the input_ids is not divisible by the block size.")
        return input_ids



class TokenDataset(Dataset): 
    """
    A  dataset for tokenized input data.
    """
    def __init__(self, block_size: int, input_ids: List[int]) -> None:
        """
        Initializes the TokenDataset.

        Args:
            block_size: The block size for dividing the input data.
            input_ids: A list containing tokenized input data.
        """
        self.block_size = block_size
        self.input_ids = input_ids

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
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
        
        # Return the input and output slices
        return torch.tensor(self.input_ids[start_idx:end_idx], dtype=torch.long), torch.tensor(self.input_ids[start_idx+1:end_idx+1], dtype=torch.long)