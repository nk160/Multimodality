import torch
import wandb
from src.model.transformer import ImageCaptioningTransformer
from src.data.preprocessing import DataPreprocessor
from src.data.dataloader import get_dataloader
from src.utils.helpers import setup_device
from src.config import config

def test_setup():
    print("Testing setup...")
    
    # Initialize wandb in disabled mode for testing
    wandb.init(mode="disabled")
    
    # Test device
    device = setup_device()
    print(f"Using device: {device}")
    
    # Test preprocessor
    preprocessor = DataPreprocessor()
    print(f"Vocab size: {preprocessor.get_vocab_size()}")
    
    # Test dataloader
    train_loader = get_dataloader("test")
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Input ids shape: {batch['input_ids'].shape}")
    
    # Test model
    model = ImageCaptioningTransformer(
        vocab_size=preprocessor.get_vocab_size(),
        hidden_size=config.model.hidden_size
    ).to(device)
    print(f"Model created successfully")
    
    # Test forward pass
    with torch.no_grad():
        outputs = model(
            images=batch['image'].to(device),
            text_tokens=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device)
        )
    print(f"Forward pass successful")
    print(f"Output keys: {outputs.keys()}")
    
    print("All tests passed!")
    
    # Cleanup
    wandb.finish()

if __name__ == "__main__":
    test_setup() 