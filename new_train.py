from __future__ import annotations
import os
import gin
import torch
import time
import numpy as np
import lightning as L
import torch.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, CosineAnnealingLR
from huggingface_hub import hf_hub_download
from model import GPTConfig, GPT, GPTWrapper
from dataset import TokenDataset, process_input_ids
import lightning as L
from lightning.pytorch import Trainer


# from tqdm import tqdm


# checking if need to download the dataset and model files
DO_DATASET_DOWNLOAD = True
DO_MODEL_DOWNLOAD = True  
LOAD_CHECKPOINT = True # checkpoint load flag to load the model and optimizer states if needed

# preparing the training dataset
TRAIN_DATA_REPO_ID = "pt-sk/pretraining-dataset"
TRAIN_DATA_REPO_TYPE = "dataset"
TRAIN_DATA_FILENAME = "tokens/CC-MAIN-2013-20---000_00001.npy"

# preparing the evaluation dataset
EVAL_DATA_REPO_ID = "pt-sk/pretraining-dataset"
EVAL_DATA_REPO_TYPE = "dataset"
EVAL_DATA_FILENAME = "tokens/wikipedia_512_pretraining-test_split.npy"

# preparing the model
MODEL_REPO_ID = "pt-sk/GPT2-Turbo"
MODEL_REPO_TYPE = "model"
MODEL_FILENAME = "43/checkpoint.pth"

# local directory to save the downloaded files
LOCAL_DIR = "/kaggle/working"

# set the tokens count 
TRAIN_TOKENS_COUNT = 29360128

# set the training tokens
TRAINING_START = 482082816   
TRAINING_END = 511442945

# set the eval tokens
EVAL_START = 5000000
EVAL_END = 6000000

# set the block size and padding token id
BLOCK_SIZE = 1024
PAD_TOKEN_ID = 100278


# Download the dataset and model files if needed
if DO_DATASET_DOWNLOAD and DO_MODEL_DOWNLOAD:
    hf_hub_download(repo_id=TRAIN_DATA_REPO_ID, filename=TRAIN_DATA_FILENAME, repo_type=TRAIN_DATA_REPO_TYPE, local_dir=LOCAL_DIR)
    hf_hub_download(repo_id=EVAL_DATA_REPO_ID, filename=EVAL_DATA_FILENAME, repo_type=EVAL_DATA_REPO_TYPE, local_dir=LOCAL_DIR)
    hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME, repo_type=MODEL_REPO_TYPE, local_dir=LOCAL_DIR)

elif DO_DATASET_DOWNLOAD:
    hf_hub_download(repo_id=TRAIN_DATA_REPO_ID, filename=TRAIN_DATA_FILENAME, repo_type=TRAIN_DATA_REPO_TYPE, local_dir=LOCAL_DIR)
    hf_hub_download(repo_id=EVAL_DATA_REPO_ID, filename=EVAL_DATA_FILENAME, repo_type=EVAL_DATA_REPO_TYPE, local_dir=LOCAL_DIR)


# Load the training dataset and eval dataset
tokens = np.load(f"{LOCAL_DIR}/{TRAIN_DATA_FILENAME}", allow_pickle=True)[TRAINING_START:TRAINING_END]
# process the input_ids(tokens) to ensure their length is divisible by block_size
tokens = process_input_ids(tokens, BLOCK_SIZE, PAD_TOKEN_ID)

eval_tokens = np.load(f"{LOCAL_DIR}/{EVAL_DATA_FILENAME}", allow_pickle=True)[EVAL_START:EVAL_END]
# process the eval_tokens to ensure their length is divisible by block_size
eval_tokens = process_input_ids(eval_tokens, BLOCK_SIZE, PAD_TOKEN_ID)

print(f"Dataset loaded with {len(tokens)} tokens....")
print(f"Evaluation Dataset loaded with {len(eval_tokens)} tokens....")

if LOAD_CHECKPOINT:
    # load the checkpoint
    checkpoint = torch.load(f"{LOCAL_DIR}/{MODEL_FILENAME}", weights_only=True, map_location="cpu")
    print("Checkpoint loaded....")

# parse the gin config file
gin.parse_config_file("config/gpt2-small.gin")
# Load the model configuration
config = GPTConfig()

# Initialize the model with the configuration 
model = GPT(config)
if LOAD_CHECKPOINT:
    # Load the model state
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f'Model loaded....')

model.to(config.dtype)


# Define Optimizer    
# optimizer = model.configure_optimizers() 
# if LOAD_CHECKPOINT:
#     # Load the optimizer state
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     print(f'Optimizer loaded....')
    
# setting the total steps and warmup steps for the scheduler
# config.steps_per_epoch = len(dataloader)
# config.total_steps = (config.steps_per_epoch * config.epochs)//config.gradient_accumulation_steps
# config.warmup_steps = int(config.total_steps * config.warmup_steps_ratio)
# print(f"Total Steps: {config.total_steps}, Warmup Steps: {config.warmup_steps}")
# print(f"Steps per Epoch: {config.steps_per_epoch}, Total Tokens: {len(tokens)}")

gpt_wrapper = GPTWrapper(config=config, model=model)
gpt_wrapper.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
print(f'Optimizer loaded....')

# trainer = Trainer(
#     accelerator="cuda",
#     strategy="ddp",
#     devices=[0, 1],
#     num_nodes=1,
#     logger=False,
#     precision="bf16-true",
#     fast_dev_run=True,
#     max_epochs=config.epochs,
#     num_sanity_val_steps=-1,
#     accumulate_grad_batches=config.gradient_accumulation_steps,
#     gradient_clip_val=config.clip_grad_norm_val,
#     benchmark=True,
#     inference_mode=True,
#     use_distributed_sampler=True
# )
dataset = TokenDataset(config.block_size, tokens)
dataloader = DataLoader(dataset, batch_size=config.batch_size, drop_last=True) # Use DataLoader to manage batches

trainer = Trainer(
    accelerator="cuda",
    strategy="ddp",
    devices=2,
    max_epochs=config.epochs,
    accumulate_grad_batches=config.gradient_accumulation_steps,
    gradient_clip_val=config.clip_grad_norm_val,
)

trainer.fit(gpt_wrapper, dataloader)