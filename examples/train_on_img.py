import torch.nn as nn
import wandb
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
import time
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from nway_attention.utils_misc import get_device
from nway_attention.modules.vit import ViTMixed, ViTtri
import random

# Load a subset of ImageNet
dataset = load_dataset("imagenet-1k")

# Explore the dataset
print(dataset)

batch_size = 160

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

class CustomImageNetDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        if image.mode == 'L':
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def print_num_parameters(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_parameters}')

print(f"batch_size = {batch_size}")

train_dataset = CustomImageNetDataset(dataset['train'], transform=train_transforms)
val_dataset = CustomImageNetDataset(dataset['validation'], transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


cfg = {
    'image_size': 64,
    'patch_size': 16,
    'num_classes': 201,
    'dim': 254,
    'depth': 3,
    'heads': 4,
    'mlp_dim': 368 * 4,
    'pool': 'cls',  
    'channels': 3, 
    'dim_head': 64,
    'dropout': 0.1,
    'emb_dropout': 0.05 
}

vit_modelc = ViTtri(cfg)
#vit_modelc = ViTtri.from_pretrained("Gusanidas/tri-vitb", config=cfg)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(vit_modelc.parameters(), lr=0.0001, weight_decay=0.05)
scheduler = OneCycleLR(optimizer, max_lr=0.00014, steps_per_epoch=12000, epochs=5, anneal_strategy='cos', div_factor=21, final_div_factor=1e4)

def evaluate(model, dataloader, criterion, device, max_steps = 200000):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if i>max_steps:
                break
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / min(len(dataloader.dataset), max_steps)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

wandb.init(project="vit-tri")
device = get_device()
vit_modelc = vit_modelc.to(device)
print_num_parameters(vit_modelc)

num_epochs = 30
eval_interval = 500
t0 = time.time()
rl = 0
for epoch in range(num_epochs):
    vit_modelc.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        if random.random() <0.03:
            print(f"batch = {i}, epoch = {epoch}, time = {time.time()-t0:.2f}, running_loss = {rl}, acct = {correct/(total+1)}, total = {total}")

        outputs = vit_modelc(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        rl = loss.item()*0.02 + rl*0.98

        if i>600:
            break

        if i % eval_interval == 0:
            train_loss = running_loss / (eval_interval * train_loader.batch_size)
            train_acc = correct / total
            max_steps = 250 if i%eval_interval*10 == 9 else 30
            val_loss, val_acc = evaluate(vit_modelc, val_loader, criterion, device, max_steps)
            current_lr = scheduler.get_last_lr()
            wandb.log({
                "epoch": epoch + 1,
                "iteration": i,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": current_lr[0]
            })

            print(f'Iteration {i}, Epoch {epoch+1}/{num_epochs}, time = {time.time()-t0:.2f}, current lr = {current_lr}')
            print(f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}, max_steps = {max_steps}')
            if max_steps>100:
                print("-------------------------")
                print("*(************************************")

            running_loss = 0.0
            correct = 0
            total = 0
            if epoch == 0 and i <2505:
                vit_modelc.push_to_hub("tri-vitb", config=cfg)
            else:
                vit_modelc.push_to_hub("tri-vita", config=cfg)

wandb.finish() 
