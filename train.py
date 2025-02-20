import torch
from torch.optim import AdamW
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, List
import evaluate
import math
import os

from src.model.transformer import ImageCaptioningTransformer
from src.data.preprocessing import DataPreprocessor
from src.data.dataloader import get_dataloader
from src.utils.helpers import (
    setup_device, 
    move_batch_to_device,
    clip_gradients,
    set_seed
)
from src.utils.lr_scheduler import WarmupCosineScheduler
from src.config import config

# Create checkpoint directory if it doesn't exist
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_model_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint with full state"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

class Trainer:
    def __init__(self):
        self.device = setup_device()
        self.preprocessor = DataPreprocessor()
        
        # Initialize metrics
        self.bleu = evaluate.load('bleu')
        self.rouge = evaluate.load('rouge')
        
        # Track best model
        self.best_val_loss = float('inf')
        self.patience = config.training.patience
        self.patience_counter = 0
        
    def setup_wandb(self):
        """Initialize W&B run"""
        wandb.init(
            project="Multimodality",
            config={
                "learning_rate": config.training.learning_rate,
                "architecture": "CLIP-GPT",
                "dataset": "Flickr30k",
                "epochs": config.training.num_epochs,
                "train_batch_size": config.data.train_batch_size,
                "eval_batch_size": config.data.eval_batch_size,
                "hidden_size": config.model.hidden_size,
                "decoder_layers": config.model.decoder_layers,
                "attention_heads": config.model.decoder_attention_heads,
                "dropout": config.model.dropout,
                "weight_decay": config.training.weight_decay,
                "warmup_steps": config.training.warmup_steps,
            }
        )
        
    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute BLEU and ROUGE scores"""
        # Skip empty predictions
        if not any(predictions):  # If all predictions are empty
            print("\nWarning: Empty predictions detected!")
            return {
                "bleu": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0
            }
        
        bleu_score = self.bleu.compute(predictions=predictions, references=references)
        rouge_score = self.rouge.compute(predictions=predictions, references=references)
        
        return {
            "bleu": bleu_score["bleu"],
            "rouge1": rouge_score["rouge1"],
            "rouge2": rouge_score["rouge2"],
            "rougeL": rouge_score["rougeL"],
        }

    def train_epoch(self, model: ImageCaptioningTransformer, 
                    optimizer: torch.optim.Optimizer,
                    scheduler: WarmupCosineScheduler,
                    split: str = "train") -> float:
        """Train for one epoch"""
        model.train()
        train_loader = get_dataloader(split)
        epoch_loss = 0
        
        # Progress bar
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = move_batch_to_device(batch, self.device)
            
            # Forward pass
            outputs = model(
                images=batch['image'],
                text_tokens=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            loss = outputs['loss']
            epoch_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Log batch metrics
            wandb.log({
                "batch_loss": loss.item(),
                "learning_rate": scheduler.get_lr()
            })
        
        return epoch_loss / len(train_loader)

    def validate(self, model: ImageCaptioningTransformer, 
                 split: str = "validation",
                 max_samples: int = 1000,
                 full_validation: bool = False) -> Dict[str, float]:
        """Optimized validation with async metrics"""
        model.eval()
        val_loader = get_dataloader(split)
        val_loss = 0
        predictions = []
        references = []
        samples_processed = 0
        
        print("\n=== Sample Predictions vs Ground Truth ===\n")
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc="Validation")):
                if max_samples and samples_processed >= max_samples:
                    break
                    
                # Debug print
                print("\nDecoding tokens...")
                
                # Generate captions
                generated_ids = model.generate(
                    images=batch['image'].to(self.device),
                    max_length=config.data.max_length
                )
                
                # Debug print
                print(f"Generated IDs shape: {generated_ids.shape}")
                print(f"Sample IDs: {generated_ids[0]}")
                
                batch_predictions = self.preprocessor.decode(generated_ids)
                
                # Debug print
                print(f"Decoded prediction: {batch_predictions[0]}")
                print(f"Ground truth: {batch['caption'][0]}")
                
                # Compute validation loss
                outputs = model(
                    images=batch['image'].to(self.device),
                    text_tokens=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device)
                )
                val_loss += outputs['loss'].item()
                
                # Collect predictions and references
                predictions.extend(batch_predictions)
                references.extend(batch['caption'])
                samples_processed += batch['image'].size(0)
        
        # Compute metrics
        metrics = {
            'val_loss': val_loss / (samples_processed / batch['image'].size(0))
        }
        
        if predictions:
            metrics.update(self.compute_metrics(predictions[:100], references[:100]))
        
        return metrics

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_lr(self):
        # Linear warmup
        if self.current_step < self.warmup_steps:
            return self.current_step / self.warmup_steps
            
        # Cosine decay
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return self.min_lr + (0.5 * (1 + math.cos(math.pi * progress)))

def train():
    """Main training function"""
    # Set random seed
    set_seed(42)
    
    # Initialize trainer
    trainer = Trainer()
    trainer.setup_wandb()
    
    # Create model
    model = ImageCaptioningTransformer(
        vocab_size=trainer.preprocessor.get_vocab_size(),
        hidden_size=config.model.hidden_size,
        decoder_layers=config.model.decoder_layers,
        decoder_attention_heads=config.model.decoder_attention_heads,
        dropout=config.model.dropout,
        tokenizer=trainer.preprocessor.tokenizer
    ).to(trainer.device)
    
    # Create optimizer with lower initial learning rate
    optimizer = AdamW(
        model.parameters(),
        lr=1e-7,  # Start very small
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Calculate steps
    num_training_steps = config.training.num_epochs * len(get_dataloader("train"))
    warmup_steps = num_training_steps // 10  # 10% warmup
    
    # Create scheduler with warmup
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=num_training_steps,
        min_lr=1e-7
    )
    
    # Training loop
    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.training.num_epochs}")
        
        # Training phase - use train split
        train_loss = trainer.train_epoch(model, optimizer, scheduler, split="train")
        
        # Quick validation - use validation split
        val_metrics = trainer.validate(model, split="validation", max_samples=500)
        
        # Full validation every N epochs
        if epoch % config.training.full_validate_every == 0:
            print("\nRunning full validation...")
            full_metrics = trainer.validate(model, split="validation", full_validation=True)
            # Log with different names to distinguish in W&B
            wandb.log({f"full_{k}": v for k, v in full_metrics.items()})
        
        # Log regular metrics
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val_metrics
        }
        wandb.log(metrics)
        
        # Print metrics summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        if 'bleu' in val_metrics:
            print(f"BLEU Score: {val_metrics['bleu']:.4f}")
            print(f"ROUGE-1: {val_metrics['rouge1']:.4f}")
            print(f"ROUGE-2: {val_metrics['rouge2']:.4f}")
            print(f"ROUGE-L: {val_metrics['rougeL']:.4f}")
        
        # Early stopping check
        if val_metrics['val_loss'] < trainer.best_val_loss:
            trainer.best_val_loss = val_metrics['val_loss']
            trainer.patience_counter = 0
            
            # Save best model
            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_metrics['val_loss'],
                path=Path(CHECKPOINT_DIR) / "best_model.pt"
            )
        else:
            trainer.patience_counter += 1
            if trainer.patience_counter >= trainer.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Regular checkpoint
        if (epoch + 1) % config.training.save_every == 0:
            save_model_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_metrics['val_loss'],
                path=Path(CHECKPOINT_DIR) / f"checkpoint_epoch_{epoch+1}.pt"
            )
    
    wandb.finish()

if __name__ == "__main__":
    train() 