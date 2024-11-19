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


hf_hub_download(repo_id="pt-sk/GPT2_Finetune_dataset", filename="file_3.npy", repo_type="dataset", local_dir="/kaggle/working")
# hf_hub_download(repo_id="pt-sk/GPT2-small_Finetune", filename="2/checkpoint.pth", repo_type="model", local_dir="/kaggle/working")


tokens = np.load("/kaggle/working/file_3.npy")
tokens = tokens[:107808979]
print(f"Number of tokens: {len(tokens)}")



gin.parse_config_file("config/gpt2-small.gin")
config = GPTConfig()


np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

# load the checkpoint
checkpoint = torch.load("/kaggle/working/2/checkpoint.pth")




def trainer(rank, world_size):
    # Initialize the Process Group
    dist.init_process_group(backend=config.training_backend, rank=rank, world_size=world_size)

    # Set the Device for the Current Process
    torch.cuda.set_device(rank)
    device = torch.device(config.device, rank)

    # Define Model and Optimizer
    model = GPT(config)
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(config.dtype).to(device)
    model = DDP(model, device_ids=[rank])  # Wrap model in DDP

    # Define Optimizer    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay)
    # Load the optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Create DataLoader
    dataset = TokenDataset(config, tokens)
    # Use DistributedSampler to partition data among distributed processes
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    # Use DataLoader to manage batches
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, drop_last=True)
    print(f"dataloader size: {len(dataloader)}")


    # Training Loop
    model.train()
    training_loss = []
    gradient_norms = []
    time_spent = []
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

            # Log training loss and gradient norms
            training_loss.append(loss.item())
            gradient_norms.append(grad_norm.item())
            time_spent.append(time.time() - start_time)
            if rank == 0:
                print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}, Gradient Norm: {grad_norm.item()}, Time Spent: {round(time_spent[-1], 2)} seconds")


    # Log training loss and gradient norms
    if rank == 0:
        np.save("training_loss.npy", np.array(training_loss))
        np.save("gradient_norms.npy", np.array(gradient_norms))
        np.save("time_spent.npy", np.array(time_spent))

        # Save the model and optimizer states for checkpointing
        torch.save(
            {
                "model_state_dict": model.state_dict(),
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