import torch
import torch.nn as nn
from transformers import CLIPModel
import sys
import os

# Add src to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import config

class CLIPEncoder(nn.Module):
    """Wrapper for CLIP's vision encoder"""
    
    def __init__(self):
        """Initialize the CLIP encoder"""
        super().__init__()
        
        # Load pretrained CLIP model
        self.clip = CLIPModel.from_pretrained(config.model.clip_model_name)
        
        # Freeze CLIP parameters (since we're using it as a feature extractor)
        for param in self.clip.parameters():
            param.requires_grad = False
            
        # Store relevant dimensions
        self.hidden_size = config.model.hidden_size
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using CLIP's vision encoder
        Args:
            images (torch.Tensor): Batch of images [batch_size, 3, H, W]
        Returns:
            torch.Tensor: Image features [batch_size, sequence_length, hidden_size]
        """
        # Get vision model outputs
        vision_outputs = self.clip.vision_model(
            pixel_values=images,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        
        # Get sequence of image features
        image_features = vision_outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
        
        return image_features
    
    def get_cls_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get CLS token features from images
        Args:
            images (torch.Tensor): Batch of images [batch_size, 3, H, W]
        Returns:
            torch.Tensor: CLS token features [batch_size, hidden_size]
        """
        # Get full sequence of features
        features = self.forward(images)
        
        # Return CLS token (first token)
        return features[:, 0, :]
        
    def get_pooled_features(self, images: torch.Tensor, pool_type: str = 'mean') -> torch.Tensor:
        """
        Get pooled image features
        Args:
            images (torch.Tensor): Batch of images
            pool_type (str): Type of pooling ('mean', 'max', or 'cls')
        Returns:
            torch.Tensor: Pooled features [batch_size, hidden_size]
        """
        features = self.forward(images)
        if pool_type == 'mean':
            return features.mean(dim=1)
        elif pool_type == 'max':
            return features.max(dim=1)[0]
        else:  # 'cls'
            return features[:, 0, :]
            
    def get_attention_maps(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get attention maps for visualization
        Args:
            images (torch.Tensor): Batch of images [batch_size, 3, H, W]
        Returns:
            torch.Tensor: Attention maps from vision transformer
        """
        vision_outputs = self.clip.vision_model(
            pixel_values=images,
            output_attentions=True,
            return_dict=True
        )
        return vision_outputs.attentions
