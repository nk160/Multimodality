import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .encoder import CLIPEncoder
from .decoder import Decoder
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
    def __init__(self, config):
        super().__init__()
        # ... other layers ...
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.decoder_ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_ffn_dim, config.hidden_size)
        )
