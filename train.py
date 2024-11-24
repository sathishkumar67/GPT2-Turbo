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
from model import GPTConfig, GPT
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
tokens = np.load(f"{LOCAL_DIR}/{DATA_FILENAME}", allow_pickle=True)[:24379393]


if LOAD_CHECKPOINT:
    # load the checkpoint
    checkpoint = torch.load(f"{LOCAL_DIR}/{MODEL_FILENAME}")
    print('Checkpoint loaded....')


def trainer(rank, world_size):
    # Enable the cudnn backend for better performance
    torch.backends.cudnn.benchmark = True

    # set the master process
    master_process = (rank == 0)
    torch.set_float32_matmul_precision('medium')  # Set the matmul precision to medium
    
    # parse the gin config file
    gin.parse_config_file("config/gpt2-small.gin")
    # Load the model configuration
    config = GPTConfig(model_device="cuda")

    # Initialize the Process Group
    dist.init_process_group(backend=config.training_backend, rank=rank, world_size=world_size)

    # Set the Device for the Current Process
    torch.cuda.set_device(rank)
    config.model_device = torch.device(config.model_device, rank) # Set the device for the current process

    # Create DataLoader
    dataset = TokenDataset(config.block_size, tokens)
    # Use DistributedSampler to partition data among distributed processes
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    # Use DataLoader to manage batches
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, drop_last=True, num_workers=1, pin_memory=True, pin_memory_device=f"{config.model_device.type}:{rank}", prefetch_factor=8, persistent_workers=True)
        
    # Initialize the model with the configuration 
    model = GPT(config)
    if LOAD_CHECKPOINT:
        # Load the model state
        model.load_state_dict(checkpoint["model_state_dict"])
        print('Model loaded....')

    model.to(config.dtype).to(config.model_device)  # Move model to respective device
    model = DDP(model, device_ids=[rank])  # Wrap model in DDP
    
    # Define Optimizer    
    optimizer = model.module.configure_optimizers() 
    if LOAD_CHECKPOINT:
        # Load the optimizer state
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print('Optimizer loaded....')

    # setting the total steps and warmup steps for the scheduler
    config.steps_per_epoch = len(dataloader)
    config.total_steps = (config.steps_per_epoch * config.epochs)//config.gradient_accumulation_steps
    config.warmup_steps = int(config.total_steps * config.warmup_steps_ratio)

    if master_process:
        print(f"Total Steps: {config.total_steps}, Warmup Steps: {config.warmup_steps}")
        print(f"Steps per Epoch: {config.steps_per_epoch}, Total Tokens: {len(tokens)}")

    # Warmup scheduler
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: step / config.warmup_steps)

    # Cosine annealing after warmup
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config.total_steps - config.warmup_steps, eta_min=config.eta_min)
    
    # Combine warmup and cosine
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[config.warmup_steps]) 

    # Training Loop 
    model.train()
    for epoch in range(config.epochs) :  # Loop over the dataset multiple times
        sampler.set_epoch(epoch)  # Shuffle data per epoch for distributed training 
        
        loss_accum = 0.0
        start_time = time.time()  
        for batch, (inputs, labels) in enumerate(dataloader):
            gradient_accum_cond = ((batch + 1) % config.gradient_accumulation_steps == 0) or ((batch + 1) == config.steps_per_epoch)
            model.require_backward_grad_sync = gradient_accum_cond
 
            # Move data to device
            inputs, labels = inputs.to(config.model_device), labels.to(config.model_device)

            # Forward pass
            _, loss = model(inputs, labels)
            
            # scale the loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps

            # Accumulate loss
            loss_accum += loss.detach()

            # Backward pass
            loss.backward()

            if gradient_accum_cond:
                # Gradient clipping before stepping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm_val)

                # Update parameters
                optimizer.step()

                # Update learning rate for the next iteration
                scheduler.step()
                
                # Zero gradients for next iteration
                optimizer.zero_grad()

                # time taken for the gradient accumulation step
                end_time = time.time() - start_time

                if master_process:
                    print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss_accum.item()}, Gradient Norm: {grad_norm.item()}, Time Spent: {round(end_time, 2)} seconds")

                loss_accum = 0.0
                grad_norm = None
                start_time = time.time()

            
    # Save the model and optimizer states            
    if master_process:
        torch.save(
            {
                "name": f"{DATA_FILENAME}_{len(tokens)}",
                "steps": config.total_steps,
                "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
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