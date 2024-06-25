import wandb
import tiktoken
import torch.nn as nn
import torch.optim as optim
import requests
import tiktoken
import random
import time
from nway_attention.cfgs import Config
from nway_attention.utils_misc import get_device
from nway_attention.modules.transformer_models import Transformer
from nway_attention.utils_misc import get_log_probs
from datasets import load_dataset
from transformers import AutoTokenizer
import time
import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset
import gc
import os
import numpy as np
import torch
import torch.nn.functional as F
import math
import random
from dotenv import load_dotenv

load_dotenv()

wandb_api_key = os.getenv("WANDB_API_KEY")

if wandb_api_key:
    wandb.login(key=wandb_api_key)
    print("Logged in to Weights & Biases.")
else:
    print("WANDB_API_KEY not found. Please check your .env file.")

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token


class DataLoaderLite:
    def __init__(self, B, T, split, data_dir='edu_fineweb10B/'):
        self.B = B  # batch size
        self.T = T  # sequence length
        assert split in {'train', 'val'}
        self.split = split
        self.data_dir = data_dir

        # Get the shard filenames
        shards = [f for f in os.listdir(data_dir) if f.startswith(f'edufineweb_{split}_') and f.endswith('.npy')]
        self.shards = sorted([os.path.join(data_dir, s) for s in shards])

        assert len(self.shards) > 0, f"No shards found for split {split} in {data_dir}"
        print(f"Found {len(self.shards)} shards for split {split}")

        self.reset()

    def reset(self):
        # Initialize at shard zero
        self.current_shard = 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def load_tokens(self, shard_path):
        npt = np.load(shard_path)
        npt = npt.astype(np.int32) # added after video
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self.load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y


    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()



def get_lr(it, warmup_steps, max_steps, max_lr, min_lr):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  
    return min_lr + coeff * (max_lr - min_lr)

def train(model_a, train_loader, val_loader, optimizer, criterion, device, config, compile=False, wandb_logging=False):
    if wandb_logging:
        wandb.init(project="jun_tri_512_4-b", config=config)
    if compile:
        model = torch.compile(model_a)
    else:
        model = model_a
    model.to(device)
    t0 = time.time()
    n2, rl = 0, 0
    for step in range(config.max_steps):
        # Update learning rate
        lr = get_lr(step, config.warmup_steps, config.max_steps, config.max_lr, config.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if step % config.val_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(config.val_steps):
                    x, y = next(val_loader)
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        logits = model(x)
                        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    val_loss += loss.item()
            val_loss /= config.val_steps
            print(f"Step {step}: Validation Loss: {val_loss:.4f}")
            if wandb_logging:
                wandb.log({"val_loss": val_loss, "step": step})
        if step > 0 and step % config.generate_interval == 0:
            if compile:
                modelg = Transformer(cfg.to_dict())
                modelg.load_state_dict(model_a.state_dict())
                modelg.to(device)
                textg = generate_text(modelg, device, config, enc)
                del modelg
            else:
                textg = generate_text(model, device, config, enc)
            print(f"Generated text: {textg}")
            if wandb_logging:
                wandb.log({"generated_text": textg, "step": step})
        model.train()
        optimizer.zero_grad()
        
        # Gradient accumulation
        accumulated_loss = 0
        for _ in range(config.accumulation_steps):
            x, y = next(train_loader)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / config.accumulation_steps
            accumulated_loss += loss.item()
            loss.backward()
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        n2 = 0.99 * n2 + 0.01 * norm
        rl = 0.99 * rl + 0.01 * accumulated_loss
        optimizer.step()
        t1 = time.time()
        dt = t1 - t0
        if step % config.log_interval == 0:
            print(f"Step {step}: Loss: {accumulated_loss:.4f}, LR: {lr:.6f}, Time: {dt:.2f}s, norm = {norm:.4f}, n2 = {n2:.4f}, rl = {rl:.4f}")
            if wandb_logging:
                wandb.log({
                    "train_loss": accumulated_loss,
                    "learning_rate": lr,
                    "grad_norm": norm,
                    "n2": n2,
                    "rl": rl,
                    "step": step
                })
        if step > 3000 and step % 4000 == 3999:
            try:
                if step> 12000:
                    model_a.push_to_hub("tri-fw-512-4b", config=cfg)
                else:
                    model_a.push_to_hub("tri-fw-512-4", config=cfg)
            except Exception as e:
                print(f"Error pushing to hub: {e}")
        if step > 1500 and step % 2000 == 0:
            config.accumulation_steps = min(config.accumulation_steps*2, 4)
    return model


def generate_text(model, device, config, enc):
    model.eval()
    if random.random() < 0.33:
        start_sentence = "Once upon a time, in a far away land "
    elif random.random()<0.5:
        start_sentence = "Hello, I'm a language model"
    else:
        start_sentence = "The quick brown fox jumps over the lazy dog "
    start_tokens = enc.encode(start_sentence)
    context = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)

    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(24)  # Add a seed to config for reproducibility

    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(context)  # (B, T, vocab_size)

            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)

            top_k = getattr(config, 'top_k', 45)
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
            next_token = torch.gather(topk_indices, -1, ix)  # (B, 1)
            context = torch.cat([context, next_token], dim=1)

    generated_text = enc.decode(context[0].tolist())
    return generated_text

cfg = Config(
    d_model = 512,
    d_head = 64,
    n_heads = 8,
    n_layers = 4,
    dropout = 0.1,
    d_vocab = 50304,
    #attn_type = "Attention",
    attn_type = "Trittention",
    attn_eq = True,
    order_attn = False,
)

model = Transformer(cfg.to_dict())


B, T = 384, 64
optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), lr=1e-5, weight_decay=0.1)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoaderLite(B=B, T=T, split="train")
train_loader.current_shard = 1
val_loader = DataLoaderLite(B=B, T=T, split="val")
class TrainConfig:
  max_steps = 22_000
  warmup_steps = 5_000
  min_lr = 7e-6
  max_lr = 7e-4
  val_interval = 2_000
  val_steps = 200
  generate_interval = 2_000
  log_interval = 250
  max_grad_norm = 1.0
  start_tokens = [1]  # <|endoftext|>
  max_new_tokens = 50
  accumulation_steps = 1


config = TrainConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train(model, train_loader, val_loader, optimizer, criterion, device, config, compile = True, wandb_logging=True)