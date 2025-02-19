import wandb
import torch
from src.model.transformer import ImageCaptioningTransformer
from src.data.preprocessing import DataPreprocessor
from src.utils.helpers import setup_device
from src.config import config

def test_wandb_connection():
    print("Testing W&B connection...")
    
    # Initialize W&B
    wandb.init(
        project="Multimodality",
        config={
            "learning_rate": 1e-4,
            "architecture": "CLIP-GPT",
            "dataset": "Flickr30k",
            "test_run": True
        }
    )
    
    # Log a few test metrics
    for i in range(5):
        wandb.log({
            "test_loss": 1.0 - i * 0.1,
            "step": i
        })
    
    print("Successfully logged test metrics to W&B")
    wandb.finish()

if __name__ == "__main__":
    test_wandb_connection() 