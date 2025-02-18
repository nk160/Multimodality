from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class ModelConfig:
    # CLIP configurations
    clip_model_name: str = "openai/clip-vit-base-patch32"
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    
    # Decoder configurations
    decoder_layers: int = 6
    decoder_attention_heads: int = 8
    decoder_ffn_dim: int = 2048
    max_position_embeddings: int = 512
    dropout: float = 0.1

@dataclass
class DataConfig:
    # Dataset configurations
    dataset_name: str = "nlphuji/flickr30k"
    train_batch_size: int = 32
    eval_batch_size: int = 32
    num_workers: int = 4
    
    # Image configurations
    image_size: int = 224
    
    # Text configurations
    max_length: int = 128
    pad_token_id: int = 0
    
@dataclass
class TrainingConfig:
    # Training parameters
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    wandb_project: str = "multimodality"
    wandb_entity: Optional[str] = None
    log_every_n_steps: int = 100

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    
config = Config()
