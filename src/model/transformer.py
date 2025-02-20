import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import torch.nn.functional as F
from src.config import config
import math

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
        
        # Ensure encoder states have correct shape
        # CLIP encoder outputs [batch_size, hidden_size]
        # Need [batch_size, 1, hidden_size] for attention
        if len(encoder_states.shape) == 2:
            encoder_states = encoder_states.unsqueeze(1)  # Add sequence length dimension
        
        # Generate with controlled parameters
        return self.decoder.generate(
            encoder_hidden_states=encoder_states,
            max_length=max_length,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.9,  # Increase temperature for more diversity
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
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Calculate attention scaling
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5  # Standard transformer scaling
        
        # Initialize attention layers with proper scaling
        std = (2.0 / (5 * hidden_size)) ** 0.5  # Xavier/Glorot initialization
        
        # Self attention layers
        self.self_attn_query = nn.Linear(hidden_size, hidden_size)
        self.self_attn_key = nn.Linear(hidden_size, hidden_size)
        self.self_attn_value = nn.Linear(hidden_size, hidden_size)
        self.self_attn_out = nn.Linear(hidden_size, hidden_size)
        self.self_attn_norm = nn.LayerNorm(hidden_size)
        
        # Cross attention layers
        self.cross_attn_query = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_key = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_value = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_out = nn.Linear(hidden_size, hidden_size)
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
        
        # Initialize self attention weights
        nn.init.normal_(self.self_attn_query.weight, std=std)
        nn.init.normal_(self.self_attn_key.weight, std=std)
        nn.init.normal_(self.self_attn_value.weight, std=std)
        nn.init.normal_(self.self_attn_out.weight, std=std)
        
        # Initialize cross attention weights
        nn.init.normal_(self.cross_attn_query.weight, std=std)
        nn.init.normal_(self.cross_attn_key.weight, std=std)
        nn.init.normal_(self.cross_attn_value.weight, std=std)
        nn.init.normal_(self.cross_attn_out.weight, std=std)
        
        # Initialize feed forward weights
        nn.init.normal_(self.feed_forward[0].weight, std=std)  # First linear layer
        nn.init.normal_(self.feed_forward[3].weight, std=std)  # Second linear layer
        
        # Initialize all biases to zero
        nn.init.zeros_(self.self_attn_query.bias)
        nn.init.zeros_(self.self_attn_key.bias)
        nn.init.zeros_(self.self_attn_value.bias)
        nn.init.zeros_(self.self_attn_out.bias)
        nn.init.zeros_(self.cross_attn_query.bias)
        nn.init.zeros_(self.cross_attn_key.bias)
        nn.init.zeros_(self.cross_attn_value.bias)
        nn.init.zeros_(self.cross_attn_out.bias)
        
        # Add Pre-LN layers
        self.pre_self_norm = nn.LayerNorm(hidden_size)
        self.pre_cross_norm = nn.LayerNorm(hidden_size)
        self.pre_ff_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, image_features, attention_mask=None, key_padding_mask=None):
        batch_size = x.shape[0]
        
        # Scale inputs to prevent explosion
        x = x * 0.1
        image_features = image_features * 0.1
        
        # Pre-LN for self attention
        normed_x = self.pre_self_norm(x)
        
        # Self attention with numerical stability
        self_q = self.self_attn_query(normed_x)
        self_k = self.self_attn_key(normed_x)
        self_v = self.self_attn_value(normed_x)
        
        # Reshape for multi-head self attention
        self_q = self_q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        self_k = self_k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        self_v = self_v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Self attention with stable softmax
        self_attn = torch.matmul(self_q, self_k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            self_attn = self_attn.masked_fill(attention_mask == 0, -1e4)  # Use finite value
        self_attn = torch.softmax(self_attn, dim=-1)
        self_attn = torch.nan_to_num(self_attn, 0.0)  # Replace NaNs with 0
        self_attn = self.dropout(self_attn)
        
        # Get self attention output
        self_output = torch.matmul(self_attn, self_v)
        self_output = self_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        self_output = self.self_attn_out(self_output)
        
        # Debug prints for self attention
        print("\nSelf Attention Stats:")
        print(f"Self attention weights mean: {self_attn.mean():.4f}")
        print(f"Self attention weights std: {self_attn.std():.4f}")
        print(f"Self attention output mean: {self_output.mean():.4f}")
        print(f"Self attention output std: {self_output.std():.4f}")
        
        # Add & Norm for self attention (residual only)
        x = x + self.dropout(self_output)
        
        # Pre-LN for cross attention
        normed_x = self.pre_cross_norm(x)
        
        # Cross attention with stability fixes
        q = self.cross_attn_query(normed_x)
        k = self.cross_attn_key(image_features)
        v = self.cross_attn_value(image_features)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with stability
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)  # Use torch.softmax
        attn_weights = torch.nan_to_num(attn_weights, 0.0)  # Replace NaNs
        attn_weights = self.dropout(attn_weights)
        
        # Get attention output
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        cross_out = self.cross_attn_out(attn_output)
        
        # Debug prints
        print("\nCross Attention Stats:")
        print(f"Cross attention weights mean: {attn_weights.mean():.4f}")
        print(f"Cross attention weights std: {attn_weights.std():.4f}")
        print(f"Cross attention output mean: {cross_out.mean():.4f}")
        print(f"Cross attention output std: {cross_out.std():.4f}")
        
        # Add & Norm (residual only)
        x = x + self.dropout(cross_out)
        
        # Pre-LN for feed forward
        normed_x = self.pre_ff_norm(x)
        ff_out = self.feed_forward(normed_x)
        x = x + self.dropout(ff_out)
        
        # Scale outputs back
        return x * 10.0

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, dropout):
        super().__init__()
        
        # Initialize embeddings with proper scale
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(config.model.max_position_embeddings, hidden_size)
        
        # Initialize with proper scale
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        # Add debug prints for embeddings
        print("\nToken Embedding Stats:")
        print(f"Shape: {self.token_embedding.weight.shape}")
        print(f"Mean: {self.token_embedding.weight.mean():.4f}")
        print(f"Std: {self.token_embedding.weight.std():.4f}")
        print(f"Min: {self.token_embedding.weight.min():.4f}")
        print(f"Max: {self.token_embedding.weight.max():.4f}")
        
        # Add debug prints for position embeddings
        print("\nPosition Embedding Stats:")
        print(f"Shape: {self.position_embedding.weight.shape}")
        print(f"Mean: {self.position_embedding.weight.mean():.4f}")
        print(f"Std: {self.position_embedding.weight.std():.4f}")
        print(f"Min: {self.position_embedding.weight.min():.4f}")
        print(f"Max: {self.position_embedding.weight.max():.4f}")
        
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
        
        # Initialize output layer with proper scaling
        output_std = (2.0 / (hidden_size + vocab_size)) ** 0.5  # Xavier/Glorot for output
        nn.init.normal_(self.output_layer.weight, std=output_std)
        nn.init.zeros_(self.output_layer.bias)
        
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
        temperature=0.9,
        top_k=50,
        no_repeat_ngram_size=3,
        num_return_sequences=1,
        early_stopping=True
    ):
        """Generate text tokens given image features"""
        batch_size = encoder_hidden_states.size(0)
        device = encoder_hidden_states.device
        
        # Debug prints
        print(f"Encoder states shape: {encoder_hidden_states.shape}")
        print(f"Start token: {bos_token_id}")
        
        # Initialize sequence with start token
        curr_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Get next token predictions
            outputs = self(
                image_features=encoder_hidden_states,
                text_tokens=curr_ids
            )
            print(f"Output logits shape: {outputs.shape}")
            next_token_logits = outputs[:, -1, :] / temperature
            
            if do_sample:
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_probs = F.softmax(top_k_logits, dim=-1)
                next_tokens = top_k_indices[torch.arange(batch_size), torch.multinomial(next_token_probs, 1).squeeze()]
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Debug: Print token values
            print(f"Generated token: {next_tokens.tolist()}")
            
            # Add predicted tokens
            curr_ids = torch.cat([curr_ids, next_tokens.unsqueeze(1)], dim=1)
            
            # Early stopping if all sequences have end token
            if early_stopping and (next_tokens == eos_token_id).any():
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
        # Ensure image features have correct shape for attention
        if len(image_features.shape) == 2:
            image_features = image_features.unsqueeze(1)  # [batch, 1, hidden]
        
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
