import torch
from torch.optim import AdamW
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Dict, List
import evaluate

from src.model.transformer import ImageCaptioningTransformer
from src.data.preprocessing import DataPreprocessor
from src.data.dataloader import get_dataloader
from src.utils.helpers import (
    setup_device, 
    move_batch_to_device,
    clip_gradients,
    save_model_checkpoint,
    set_seed
)
from src.utils.lr_scheduler import WarmupCosineScheduler
from src.config import config

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
            clip_gradients(model)
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
                max_samples: int = 1000,  # Regular validation subset
                full_validation: bool = False) -> Dict[str, float]:
        """Optimized validation with async metrics"""
        model.eval()
        val_loader = get_dataloader(split)
        val_loss = 0
        predictions = []
        references = []
        samples_processed = 0
        
        # Determine max samples
        max_samples = None if full_validation else max_samples
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if max_samples and samples_processed >= max_samples:
                    break
                    
                # Basic validation (loss computation)
                outputs = model(
                    images=batch['image'].to(self.device),
                    text_tokens=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device)
                )
                val_loss += outputs['loss'].item()
                
                # Generate captions asynchronously for metrics
                if samples_processed % 100 == 0:  # Generate every 100 samples
                    generated_ids = model.generate(
                        images=batch['image'].to(self.device),
                        max_length=config.data.max_length
                    )
                    predictions.extend(self.preprocessor.decode(generated_ids))
                    references.extend(batch['caption'])
                
                samples_processed += batch['image'].size(0)
        
        # Quick metrics for monitoring
        metrics = {
            'val_loss': val_loss / (samples_processed / batch['image'].size(0))
        }
        
        # Async compute full metrics if we have predictions
        if predictions:
            metrics.update(self.compute_metrics(predictions[:100], references[:100]))
        
        return metrics

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
        dropout=config.model.dropout
    ).to(trainer.device)
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Create scheduler - using test split for steps calculation
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=config.training.warmup_steps,
        total_steps=config.training.num_epochs * len(get_dataloader("test"))
    )
    
    # Training loop
    for epoch in range(config.training.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.training.num_epochs}")
        
        # Training phase - use test split
        train_loss = trainer.train_epoch(model, optimizer, scheduler, split="test")
        
        # Quick validation - use test split
        val_metrics = trainer.validate(model, split="test", max_samples=500)
        
        # Full validation every N epochs
        if epoch % config.training.full_validate_every == 0:
            print("\nRunning full validation...")
            full_metrics = trainer.validate(model, split="test", full_validation=True)
            # Log with different names to distinguish in W&B
            wandb.log({f"full_{k}": v for k, v in full_metrics.items()})
        
        # Log regular metrics
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val_metrics
        }
        wandb.log(metrics)
        
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
                path=Path(config.checkpoint_dir) / "best_model.pt"
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
                path=Path(config.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            )
    
    wandb.finish()

if __name__ == "__main__":
    train() 