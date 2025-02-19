import torch
from train import Trainer
from src.model.transformer import ImageCaptioningTransformer
from src.utils.lr_scheduler import WarmupCosineScheduler
from src.config import config
from src.utils.helpers import set_seed
from torch.optim import AdamW
import wandb

def test_training_pipeline():
    """Test the full training pipeline with a small number of epochs"""
    # Override config for testing
    config.training.num_epochs = 2
    config.data.train_batch_size = 8
    config.data.eval_batch_size = 8
    
    # For testing purposes, we'll use the test split
    original_dataset = config.data.dataset_name
    config.data.dataset_name = "nlphuji/flickr30k"
    
    print("Starting test with smaller dataset...")
    
    # Initialize everything
    set_seed(42)
    trainer = Trainer()
    trainer.setup_wandb()
    
    print("Creating model...")
    model = ImageCaptioningTransformer(
        vocab_size=trainer.preprocessor.get_vocab_size(),
        hidden_size=config.model.hidden_size,
        decoder_layers=config.model.decoder_layers,
        decoder_attention_heads=config.model.decoder_attention_heads,
        dropout=config.model.dropout
    ).to(trainer.device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=10,  # Smaller number for testing
        total_steps=20    # Smaller number for testing
    )
    
    print("\nStarting test training...")
    try:
        # Use test split for both training and validation
        train_loss = trainer.train_epoch(model, optimizer, scheduler, split="test")
        print(f"Training loss: {train_loss:.4f}")
        
        print("\nRunning validation...")
        val_metrics = trainer.validate(model, split="test")
        print("Validation metrics:", val_metrics)
        
        print("\nTest completed successfully!")
    finally:
        # Restore original dataset config
        config.data.dataset_name = original_dataset
        wandb.finish()

if __name__ == "__main__":
    test_training_pipeline() 