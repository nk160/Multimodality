import torch
from tqdm import tqdm
import wandb
from typing import Dict, Any

class TrainingLogger:
    def __init__(self, config):
        self.config = config
        self.step = 0
        
    def log_batch(self, metrics: Dict[str, float], desc: str = "Training"):
        """Log batch metrics with progress bar"""
        if wandb.run is not None:
            wandb.log(metrics, step=self.step)
        self.step += 1
        
    def log_epoch(self, metrics: Dict[str, float]):
        """Log epoch-level metrics"""
        if wandb.run is not None:
            wandb.log(metrics, step=self.step)

def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move batch data to specified device"""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()} 