import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Optional, Union, Tuple

# Add src to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import config

class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention and cross-attention"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Self attention for text
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross attention to image features
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, config.model.intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.model.intermediate_size, hidden_size)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, image_features: torch.Tensor, 
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for a single decoder layer
        Args:
            x: Text features [batch_size, seq_len, hidden_size]
            image_features: Image features from CLIP [batch_size, num_patches, hidden_size]
            attention_mask: Mask for self-attention [batch_size, seq_len, seq_len]
        """
        # 1. Self Attention with residual connection
        attn_output, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attention_mask,
            need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Cross Attention with residual connection
        cross_output, _ = self.cross_attn(
            query=x,
            key=image_features,
            value=image_features,
            need_weights=False
        )
        x = self.norm2(x + self.dropout(cross_output))
        
        # 3. Feed Forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class Decoder(nn.Module):
    """
    Full decoder for image captioning
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = config.model.hidden_size,
        num_layers: int = config.model.decoder_layers,
        num_heads: int = config.model.decoder_attention_heads,
        max_length: int = config.data.max_length,
        dropout: float = config.model.dropout
    ):
        super().__init__()
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Save config
        self.max_length = max_length
        self.hidden_size = hidden_size
        
        # Add special token IDs
        self.start_token_id = 0  # We'll need to get this from tokenizer
        self.end_token_id = 1    # We'll need to get this from tokenizer

    def forward(
        self,
        image_features: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_probs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the entire decoder
        Args:
            image_features: CLIP image features [batch_size, num_patches, hidden_size]
            text_tokens: Input token ids [batch_size, seq_len]
            attention_mask: Mask for self-attention [batch_size, seq_len, seq_len]
            return_probs: Whether to return probability distribution over vocabulary
        Returns:
            logits or (logits, probs) if return_probs is True
        """
        # Add error checking
        if image_features.size(-1) != self.hidden_size:
            raise ValueError(f"Image feature size {image_features.size(-1)} does not match hidden size {self.hidden_size}")
        
        if torch.max(text_tokens) >= self.token_embedding.num_embeddings:
            raise ValueError("Text tokens contain ids larger than vocabulary size")
        
        # Get sequence length
        seq_length = text_tokens.size(1)
        
        # Create position ids
        position_ids = torch.arange(seq_length, device=text_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(text_tokens)
        
        # Get embeddings
        token_embeddings = self.token_embedding(text_tokens)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = token_embeddings + position_embeddings
        x = self.dropout(x)
        
        # Pass through each decoder layer
        for layer in self.layers:
            x = layer(x, image_features, attention_mask)
        
        # Get logits
        logits = self.output_projection(x)
        
        if return_probs:
            # Apply softmax for probability distribution
            probs = F.softmax(logits, dim=-1)
            return logits, probs
        return logits

    def generate(
        self,
        image_features: torch.Tensor,
        max_length: int = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate caption for image features
        Args:
            image_features: Image features from CLIP
            max_length: Maximum length of generated caption
            temperature: Sampling temperature (higher = more random)
        """
        # Add error checking
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        
        if max_length is not None and max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        
        if max_length is None:
            max_length = self.max_length
            
        # Start with batch of start tokens
        batch_size = image_features.shape[0]
        device = image_features.device
        
        # Initialize with start token
        current_tokens = torch.full(
            (batch_size, 1),
            fill_value=self.start_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Generate tokens
        for _ in range(max_length - 1):
            # Get logits for next token
            logits, probs = self.forward(
                image_features,
                current_tokens,
                return_probs=True
            )
            
            # Get probabilities for next token only
            next_token_probs = probs[:, -1, :] / temperature
            
            # Sample from probability distribution
            next_token = torch.multinomial(next_token_probs, 1)
            
            # Append to sequence
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
            # Stop if all sequences have end token
            if (next_token == self.end_token_id).all():
                break
                
        return current_tokens
