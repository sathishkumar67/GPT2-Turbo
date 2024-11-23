from __future__ import annotations
import os
import gin
import torch
import time
import numpy as np
import torch.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, CosineAnnealingLR
from huggingface_hub import hf_hub_download
from model import GPTConfig, GPT, CosineWarmupScheduler
from dataset import TokenDataset
# import warnings
# warnings.filterwarnings("ignore")


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
    hf_hub_download(repo_id=DATA_REPO_ID, filename=DATA_FILENAME, repo_type=DATA_REPO_TYPE, local_dir=LOCAL_DIR)
    hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME, repo_type=MODEL_REPO_TYPE, local_dir=LOCAL_DIR)

elif DO_DATASET_DOWNLOAD:
    hf_hub_download(repo_id=DATA_REPO_ID, filename=DATA_FILENAME, repo_type=DATA_REPO_TYPE, local_dir=LOCAL_DIR)

# elif DO_MODEL_DOWNLOAD:
#     hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME, repo_type=MODEL_REPO_TYPE, local_dir=LOCAL_DIR)


# Load the dataset
tokens = np.load(f"{LOCAL_DIR}/{DATA_FILENAME}", allow_pickle=True)
tokens = tokens[:40000000]
print(f"Number of tokens: {len(tokens)}")



if LOAD_CHECKPOINT:
    # load the checkpoint
    checkpoint = torch.load(f"{LOCAL_DIR}/{MODEL_FILENAME}")
    


def trainer(rank, world_size):
    torch.set_float32_matmul_precision('medium')  # Set the matmul precision to medium

    # Load the model configuration
    gin.parse_config_file("config/gpt2-small.gin")
    config = GPTConfig(model_device="cuda")

    # Initialize the Process Group
    dist.init_process_group(backend=config.training_backend, rank=rank, world_size=world_size)

    # Set the Device for the Current Process
    torch.cuda.set_device(rank)
    device = torch.device(config.model_device, rank) # Set the device for the current process
    config.model_device = device

    # Create DataLoader
    dataset = TokenDataset(config, tokens)
    # Use DistributedSampler to partition data among distributed processes
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    # Use DataLoader to manage batches
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, drop_last=True, num_workers=1, pin_memory=True, pin_memory_device=f"{device.type}:{rank}", prefetch_factor=2, persistent_workers=True)
    print(f"dataloader size: {len(dataloader)}")
        
    # Initialize the model with the configuration 
    model = GPT(config)
    if LOAD_CHECKPOINT:
        # Load the model state
        print("Loading model state....")
        model.load_state_dict(checkpoint["model_state_dict"])
    

    model.to(config.dtype).to(device)
    model = DDP(model, device_ids=[rank])  # Wrap model in DDP
    
    # Define Optimizer    
    optimizer = model.module.configure_optimizers() 
    if LOAD_CHECKPOINT:
        # Load the optimizer state
        print("Loading optimizer state....")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Define Scheduler
    # scheduler =CosineWarmupScheduler(optimizer, warmup_steps=1000, total_steps=len(dataloader), eta_min=config.learning_rate*0.1)

    # Warmup scheduler
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: step / 10)

    # Cosine annealing after warmup
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) - 10, eta_min=1e-5)

    # Combine warmup and cosine
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[10])


    # Training Loop
    model.train()
    for epoch in range(config.epochs) :  # Loop over the dataset multiple times
        sampler.set_epoch(epoch)  # Shuffle data per epoch for distributed training 
        
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

            # Update learning rate
            scheduler.step()

            end_time = time.time() - start_time
            if rank == 0:
                print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}, Gradient Norm: {grad_norm.item()}, Time Spent: {round(end_time, 2)} seconds")
                print(f"Learning Rate: {scheduler.get_last_lr()[0]}")

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