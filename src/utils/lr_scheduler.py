import math
import torch
from typing import List

class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.get_warmup_lr()
        else:
            lr = self.get_cosine_lr()
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_warmup_lr(self):
        return self.optimizer.param_groups[0]['lr'] * (self.current_step / self.warmup_steps)
        
    def get_cosine_lr(self):
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.min_lr + (self.optimizer.param_groups[0]['lr'] - self.min_lr) * \
               0.5 * (1 + math.cos(math.pi * progress))

    def get_lr(self) -> float:
        """Get current learning rate"""
        if self.current_step < self.warmup_steps:
            return self.get_warmup_lr()
        return self.get_cosine_lr()

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate"""
        return [self.get_lr()] 