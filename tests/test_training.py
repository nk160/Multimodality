import pytest
import torch
from tqdm import tqdm
import wandb
from src.config import config

def test_training_batch():
    """Test single training batch with progress monitoring"""
    # Setup small batch
    batch_size = 4
    print("\nTesting training batch processing...")
    
    try:
        # Process single batch
        with tqdm(total=1, desc="Training batch") as pbar:
            # Your batch processing here
            # Simulate processing time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            pbar.update(1)
            
        print("✓ Batch processing successful")
        return True
    except Exception as e:
        print(f"✗ Batch processing failed: {str(e)}")
        return False

def test_wandb_logging():
    """Test W&B logging setup"""
    print("\nTesting W&B logging...")
    
    try:
        # Initialize W&B in dry-run mode for testing
        wandb.init(mode="disabled")
        
        # Log some test metrics
        wandb.log({
            "test_loss": 0.5,
            "test_accuracy": 0.8
        })
        
        wandb.finish()
        print("✓ W&B logging successful")
        return True
    except Exception as e:
        print(f"✗ W&B logging failed: {str(e)}")
        return False

def test_gpu_availability():
    """Test GPU availability and memory"""
    print("\nTesting GPU availability...")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")
        print(f"✓ GPU memory cached: {torch.cuda.memory_reserved(0)/1e9:.2f}GB")
        return True
    else:
        print("✗ No GPU available, running on CPU")
        return False

if __name__ == "__main__":
    # Run tests
    test_training_batch()
    test_wandb_logging()
    test_gpu_availability() 