from __future__ import annotations
import torch
from torch.utils.data import Dataset
from typing import Tuple, List




class TokenDataset(Dataset): # need to pad tokens if the length is less than the block size
    def __init__(self, block_size: int, input_ids: List[int], pad_token_id: int) -> None:
        """
        Initializes the TokenDataset.

        Args:
            block_size: The block size for dividing the input data.
            input_ids: A list containing tokenized input data.
            pad_token_id: The token ID used for padding.
        """
        self.block_size = block_size
        self.pad_token_id = pad_token_id
                
        # Pad input_ids to make its length a multiple of block_size
        remainder = len(input_ids) % block_size
        if remainder != 0:
            padding_length = block_size - remainder # calculate the padding length
            input_ids.extend([pad_token_id] * padding_length) # pad the input_ids with pad_token_id to make the length a multiple of block_size

        # Store the input_ids
        self.input_ids = input_ids

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
        
        # Return the input and output slices
        return torch.tensor(self.input_ids[start_idx:end_idx], dtype=torch.long), torch.tensor(self.input_ids[start_idx+1:end_idx+1], dtype=torch.long)