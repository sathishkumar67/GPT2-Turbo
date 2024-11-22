from __future__ import annotations
import os
import gin
import torch
import time
import numpy as np
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from huggingface_hub import hf_hub_download
from model import GPTConfig, GPT
from dataset import TokenDataset


# checking if need to download the dataset and model files
DO_DATASET_DOWNLOAD = True
DO_MODEL_DOWNLOAD = False  

# preparing the dataset
DATA_REPO_ID = "pt-sk/pretraining-dataset"
DATA_REPO_TYPE = "dataset"
DATA_FILENAME = "tokens/CC-MAIN-2013-20---000_00000.npy"

# preparing the model
MODEL_REPO_ID = ""
MODEL_REPO_TYPE = ""
MODEL_FILENAME = ""

# checkpoint load flag to load the model and optimizer states if needed
LOAD_CHECKPOINT = False

# local directory to save the downloaded files
LOCAL_DIR = "/kaggle/working"

# Download the dataset and model files if needed
if DO_DATASET_DOWNLOAD and DO_MODEL_DOWNLOAD:
    print("Downloading dataset files....")
    hf_hub_download(repo_id=DATA_REPO_ID, filename=DATA_FILENAME, repo_type=DATA_REPO_TYPE, local_dir=LOCAL_DIR)
    print("Downloading model files....")
    hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME, repo_type=MODEL_REPO_TYPE, local_dir=LOCAL_DIR)

elif DO_DATASET_DOWNLOAD:
    print("Downloading dataset files....")
    hf_hub_download(repo_id=DATA_REPO_ID, filename=DATA_FILENAME, repo_type=DATA_REPO_TYPE, local_dir=LOCAL_DIR)

elif DO_MODEL_DOWNLOAD:
    print("Downloading model files....")
    hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME, repo_type=MODEL_REPO_TYPE, local_dir=LOCAL_DIR)


# Load the dataset
print("Loading dataset....")
tokens = np.load(f"{LOCAL_DIR}/{DATA_FILENAME}", allow_pickle=True)
print(f"Number of tokens: {len(tokens)}")


# Load the model configuration
print("Loading model configuration....")
gin.parse_config_file("config/gpt2-small.gin")
config = GPTConfig()
print(config)


# Set the seed for reproducibility
print("Setting seed for reproducibility....")
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)


if LOAD_CHECKPOINT:
    # load the checkpoint
    print("Loading checkpoint....")
    checkpoint = torch.load("/kaggle/working/2/checkpoint.pth")
    print("Checkpoint loaded....")



def trainer(rank, world_size):

    # Initialize the Process Group
    dist.init_process_group(backend=config.training_backend, rank=rank, world_size=world_size)

    # Set the Device for the Current Process
    torch.cuda.set_device(rank)
    device = torch.device(config.device, rank)

    # Define Model and 
    print("Creating model....")
    model = GPT(config)
    
    if LOAD_CHECKPOINT:
        # Load the model state
        print("Loading model state....")
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model state loaded....")
    
    model.to(config.dtype).to(device)
    model = DDP(model, device_ids=[rank])  # Wrap model in DDP

    # Define Optimizer    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay)
    
    if LOAD_CHECKPOINT:
        # Load the optimizer state
        print("Loading optimizer state....")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Optimizer state loaded....")


    # Create DataLoader
    dataset = TokenDataset(config, tokens)
    # Use DistributedSampler to partition data among distributed processes
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    # Use DataLoader to manage batches
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, drop_last=True)
    print(f"dataloader size: {len(dataloader)}")


    # Training Loop
    model.train()
    for epoch in range(config.epochs) :  # Loop over the dataset multiple times
        sampler.set_epoch(epoch)  # Shuffle data per epoch for 
        
        for batch, (inputs, labels) in enumerate(dataloader):
            start_time = time.time()
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            _, loss = model(inputs, labels)
            
            # Zero gradients before backward pass
            optimizer.zero_grad()
            
            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm_val)
            
            # Update weights and biases
            optimizer.step()

            end_time = time.time() - start_time
            if rank == 0:
                print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}, Gradient Norm: {grad_norm.item()}, Time Spent: {round(end_time, 2)} seconds")


    # Log training loss and gradient norms
    if rank == 0:
        # Save the model and optimizer states for checkpointing
        torch.save(
            {
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            "checkpoint.pth",
        )

    # Cleanup
    dist.destroy_process_group()


def run_ddp_training():
    world_size = torch.cuda.device_count()  # Number of available GPUs
    mp.spawn(trainer, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())  # Total number of GPUs on this node
    os.environ['RANK'] = '0'  # Rank 0 for a single-node setup

    run_ddp_training()