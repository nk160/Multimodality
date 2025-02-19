import torch
import torch.nn as nn
from typing import Dict, Any, Union, List
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import config

def setup_device() -> torch.device:
    """Set up and return the appropriate device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Set cuda device if multiple GPUs
        if torch.cuda.device_count() > 1:
            torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    return device

def count_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def clip_gradients(model: nn.Module, max_norm: float = 1.0):
    """Clip gradients to prevent exploding gradients"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Union[str, Path]
):
    """Save model checkpoint with metadata"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, path)

def load_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Union[str, Path]
) -> Dict[str, Any]:
    """Load model checkpoint and return metadata"""
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss']
    }

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move all tensors in batch to specified device"""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()}

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy for generated captions"""
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).float()
    return correct.mean().item()

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
