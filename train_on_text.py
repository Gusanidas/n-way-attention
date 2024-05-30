from cfgs import Config
from transformer_models import TriformerMixed
from utils_misc import get_log_probs
from datasets import load_dataset
from transformers import AutoTokenizer
import time
import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset
import gc




class CustomDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, idx):
        item = self.tokenized_dataset[idx]
        return torch.tensor(item['input_ids'])
class CustomIterableDataset(IterableDataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __iter__(self):
        for item in self.tokenized_dataset:
            yield torch.tensor(item['input_ids'])

def tokenize_fine(examples):
    te = examples['text']
    r = tokenizer(te, truncation=True, max_length=1024, padding = "max_length", pad_to_multiple_of=None)
    return r


batch_size = 10

#dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
dataset = load_dataset("Locutusque/UltraTextbooks", split="train")
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#tokenized_train_dataset = dataset.map(tokenize_fine)
train_dataset = dataset.select(range(2_250_000, 2_500_000))

#tokenized_train_dataset = dataset.map(tokenize_function)#, batched=True, num_proc=4)
tokenized_train_dataset = train_dataset.map(tokenize_fine)#, batched=True, num_proc=4
#train_dataset = CustomIterableDataset(tokenized_train_dataset)
#train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

train_dataset = CustomDataset(tokenized_train_dataset)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model_cfg = Config(
    d_model=512,       
    d_vocab=32001,        
    init_range=0.02,      
    n_ctx=1024,           
    d_head=64,           
    dt_head=64,
    d_mlp=768*4,         
    n_heads=6,           
    nt_heads=2,        
    n_layers=8,       
    mlp_type="all",
    order_attn = True,
    window_size=32,
    autopad=True,
)

model = TriformerMixed.from_pretrained('Gusanidas/mixnet_12b')
#model = TriformerMixed(model_cfg.to_dict())
print(model)
print(model.cfg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.cfg.window_size = 32
num_epochs = 4
lr = 3e-4
max_lr = 5e-5
accumulation_steps = 25
total_steps = 3000
warmup_steps = 160
validation_interval = 2000
validation_steps = 400

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=max_lr, total_steps=total_steps, 
    anneal_strategy='cos', cycle_momentum=False, 
    div_factor=7, final_div_factor=1e4
)

total_count = 0
t0 = time.time()
rl = 0
rl2 = 0
for epoch in range(num_epochs):
    model.train() 
    for batch_idx, data in enumerate(train_dataloader):
        total_count += 1
        data = data.to(device)
        
        tokens = data 
        logits = model(tokens)
        
        log_probs = get_log_probs(logits, tokens)
        loss = -log_probs.mean() / accumulation_steps
        
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            scheduler.step()

        rl = loss.item() * accumulation_steps*0.02 + rl*0.98
        rl2 = loss.item() * accumulation_steps*0.0004 + rl2*0.9996
        if batch_idx % 99 == 0:
            padding = tokens == tokenizer.pad_token_id  
            current_lr = scheduler.get_last_lr()
            print(f'Batch {batch_idx}, Loss: {loss.item() * accumulation_steps}, Time = {time.time() - t0}, LR = {current_lr}, rl = {rl}, rl2 = {rl2} total_step = {total_count}')

        if total_count % 1000 == 999:
            model.push_to_hub("mixnet_12", config=model_cfg.to_dict())

        if batch_idx>12000:
            break

model.push_to_hub("mixnet_12", config=model_cfg.to_dict())

