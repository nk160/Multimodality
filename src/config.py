from dataclasses import dataclass, field
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
    decoder_layers: int = 8
    decoder_attention_heads: int = 12
    decoder_ffn_dim: int = 2048
    max_position_embeddings: int = 512
    dropout: float = 0.1

@dataclass
class DataConfig:
    # Dataset configurations
    dataset_name: str = "nlphuji/flickr30k"
    train_batch_size: int = 64
    eval_batch_size: int = 64
    num_workers: int = 8
    
    # Image configurations
    image_size: int = 224
    
    # Text configurations
    max_length: int = 128
    pad_token_id: int = 0
    
@dataclass
class TrainingConfig:
    # Training parameters
    num_epochs: int = 6
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    save_every: int = 2
    patience: int = 1
    clip_gradient: float = 1.0
    full_validate_every: int = 6
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    wandb_project: str = "multimodality"
    wandb_entity: Optional[str] = None
    log_every_n_steps: int = 100

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"

config = Config()
