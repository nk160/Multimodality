import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import torch.nn.functional as F
from src.config import config

from .encoder import CLIPEncoder
import wandb

class ImageCaptioningTransformer(nn.Module):
    """
    Complete transformer model for image captioning,
    combining CLIP encoder and custom decoder
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: Optional[int] = None,
        decoder_layers: Optional[int] = 6,
        decoder_attention_heads: Optional[int] = 8,
        dropout: Optional[float] = 0.1,
        tokenizer=None
    ):
        super().__init__()
        
        # Initialize encoder
        self.encoder = CLIPEncoder()
        self.tokenizer = tokenizer
        
        # Use encoder's hidden size if not specified
        if hidden_size is None:
            hidden_size = self.encoder.hidden_size
            
        # Initialize decoder with default values if None
        self.decoder = Decoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=decoder_layers or config.model.decoder_layers,
            num_heads=decoder_attention_heads or config.model.decoder_attention_heads,
            dropout=dropout or config.model.dropout
        )
        
    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the entire model
        Args:
            images: [batch_size, 3, H, W]
            text_tokens: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_loss: Whether to compute and return loss
        Returns:
            Dict containing model outputs and loss if requested
        """
        # Get image features from CLIP encoder
        image_features = self.encoder(images)
        
        # Create attention mask
        if attention_mask is not None:
            seq_length = text_tokens.size(1)
            # Create causal mask (upper triangular)
            causal_mask = torch.triu(
                torch.ones((seq_length, seq_length), dtype=torch.bool, device=attention_mask.device),
                diagonal=1
            )
            # Convert padding mask to boolean where True means to mask
            key_padding_mask = ~attention_mask.bool()
        else:
            causal_mask = None
            key_padding_mask = None
        
        # Get decoder outputs
        logits = self.decoder(
            image_features=image_features,
            text_tokens=text_tokens,
            attention_mask=causal_mask,
            key_padding_mask=key_padding_mask
        )
        
        outputs = {"logits": logits}
        
        # Compute loss if requested
        if return_loss:
            # Shift labels for causal language modeling
            labels = text_tokens[:, 1:].contiguous()
            shift_logits = logits[:, :-1, :].contiguous()
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                labels.view(-1)
            )
            
            outputs["loss"] = loss
            
            # Log to W&B
            wandb.log({"train_loss": loss.item()})
            
        return outputs
    
    def generate(self, images, max_length=128):
        """Generate captions for images"""
        # Get encoder states for images
        encoder_states = self.encode_images(images)
        
        # Generate with controlled parameters
        return self.decoder.generate(
            encoder_hidden_states=encoder_states,
            max_length=max_length,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            early_stopping=True
        )
    
    def encode_images(self, images):
        """Encode images using CLIP encoder"""
        return self.encoder(images)
    
    def save_pretrained(self, path: str):
        """Save model to path"""
        torch.save(self.state_dict(), path)
        
    def from_pretrained(self, path: str):
        """Load model from path"""
        self.load_state_dict(torch.load(path))
        return self

class DecoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        # Self attention
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.self_attn_norm = nn.LayerNorm(hidden_size)
        
        # Cross attention
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.cross_attn_norm = nn.LayerNorm(hidden_size)
        
        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, image_features, attention_mask=None, key_padding_mask=None):
        # Transpose for attention operations
        x = x.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        image_features = image_features.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        
        # Self attention
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attn_out)
        x = self.self_attn_norm(x)
        
        # Cross attention with image features
        cross_out, _ = self.cross_attn(
            x, image_features, image_features
        )
        x = x + self.dropout(cross_out)
        x = self.cross_attn_norm(x)
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)
        x = self.ff_norm(x)
        
        # Transpose back
        x = x.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, dropout):
        super().__init__()
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(config.model.max_position_embeddings, hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def generate(
        self,
        encoder_hidden_states,
        max_length,
        bos_token_id,
        eos_token_id,
        pad_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        no_repeat_ngram_size=3,
        num_return_sequences=1,
        early_stopping=True
    ):
        """Generate text tokens given image features"""
        batch_size = encoder_hidden_states.size(0)
        device = encoder_hidden_states.device
        
        # Initialize sequence with start token
        curr_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Get next token predictions
            outputs = self(
                image_features=encoder_hidden_states,
                text_tokens=curr_ids
            )
            next_token_logits = outputs[:, -1, :] / temperature
            
            if do_sample:
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_probs = F.softmax(top_k_logits, dim=-1)
                next_tokens = top_k_indices[torch.arange(batch_size), torch.multinomial(next_token_probs, 1).squeeze()]
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Add predicted tokens
            curr_ids = torch.cat([curr_ids, next_tokens.unsqueeze(1)], dim=1)
            
            # Early stopping if all sequences have end token
            if early_stopping and (curr_ids == eos_token_id).any(dim=1).all():
                break
        
        return curr_ids

    def forward(
        self,
        image_features,
        text_tokens,
        attention_mask=None,
        key_padding_mask=None
    ):
        """Forward pass through decoder"""
        # Get embeddings
        text_embeds = self.token_embedding(text_tokens)
        
        # Add positional encoding
        position_ids = torch.arange(text_tokens.size(1), device=text_tokens.device)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = text_embeds + position_embeds
        
        # Apply transformer blocks
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                image_features,
                attention_mask=attention_mask,
                key_padding_mask=key_padding_mask
            )
            
        # Get logits
        logits = self.output_layer(hidden_states)
        
        return logits
