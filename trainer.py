import torch.optim.lr_scheduler as lr_scheduler
from dataclasses import dataclass
import torch as t
from typing import List, Optional, Dict
from torch.utils.data import DataLoader
import wandb
from time import time
from utils_misc import get_log_probs_last, pad_cat

device = t.device('cuda' if t.cuda.is_available() else 'cpu')


@dataclass
class TransformerTrainingArgs:
    batch_size: int = 128
    epochs: int = 252
    max_steps_per_epoch: int = 500
    lr: float = 7e-4
    weight_decay: float = 1e-2
    wandb_project: Optional[str] = "day2-demotransformer"
    wandb_name: Optional[str] = None
    decay_scheduler: str = 'cosine'



class Trainer:
    def __init__(self, args, model, train_list: List[t.Tensor], test_list: List[t.Tensor]):
        super().__init__()
        self.model = model
        self.args = args
        self.optimizer = t.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.step = 0
        self.train_list = train_list
        self.test_list = test_list
        self.t_0 = time()
        if args.decay_scheduler == 'cosine':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        elif args.decay_scheduler == 'linear':
            lambda_lr = lambda epoch: 1 - epoch / args.epochs
            self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)
        elif args.decay_scheduler == 'cyclic':
            self.scheduler = lr_scheduler.CyclicLR(self.optimizer, base_lr=args.lr / 10, max_lr=args.lr, step_size_up=5)
        elif args.decay_scheduler == 'exponential':
            self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        else:
            raise ValueError("Unsupported scheduler type provided.")

    def training_step(self, batch: Dict[str, t.Tensor]) -> float:
        tokens = batch.to(device)
        logits = self.model(tokens)
        loss = -get_log_probs_last(logits, tokens)
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        wandb.log({"train_loss": loss.item()}, step=self.step)
        return loss.item()

    def validation_step(self, batch: Dict[str, t.Tensor], only_last: bool = True) -> float:
        tokens = batch.to(device)
        logits = self.model(tokens)[:, :-1]
        predicted_tokens = logits.argmax(dim=-1)
        if only_last:
            correct_predictions = (predicted_tokens[:, -1] == tokens[:, -1]).flatten()
        else:
            correct_predictions = (predicted_tokens == tokens[:, 1:]).flatten()
        return correct_predictions.float().mean().item()

    def train(self):
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader()):
                loss = self.training_step(batch)
                if i >= self.args.max_steps_per_epoch:
                    break
                if i==42 and epoch == 1 and t.cuda.is_available():
                    total_memory, allocated_memory = t.cuda.mem_get_info()
                    total_memory_gb = total_memory / (1024 ** 3) 
                    allocated_memory_gb = allocated_memory / (1024 ** 3)  
                    free_memory_gb = total_memory_gb - allocated_memory_gb
                    used_memory_pct = (allocated_memory_gb / total_memory_gb) * 100

                    print(f"Total GPU Memory: {total_memory_gb:.2f} GB")
                    print(f"Allocated Memory: {allocated_memory_gb:.2f} GB")
                    print(f"Free Memory: {free_memory_gb:.2f} GB")
                    print(f"Used Memory Percentage: {used_memory_pct:.2f}%")


            accuracy_list = [self.validation_step(batch) for batch in self.test_loader()]
            accuracy = sum(accuracy_list) / len(accuracy_list)
            wandb.log({"accuracy": accuracy}, step=self.step)
            self.scheduler.step()

        wandb.finish()
        return loss, accuracy

    def train_loader(self) -> DataLoader:
        return DataLoader(pad_cat(self.train_list), batch_size=self.args.batch_size, shuffle=True, num_workers=0)

    def test_loader(self) -> DataLoader:
        return DataLoader(pad_cat(self.test_list), batch_size=self.args.batch_size, shuffle=False, num_workers=0)
