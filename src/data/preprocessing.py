import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPTokenizer
from typing import Dict, Union, List
import sys
import os

# Add src to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import config

class DataPreprocessor:
    """Handles all data preprocessing for images and text"""
    
    def __init__(self):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Debug tokenizer configuration
        print("\nTokenizer Configuration:")
        print("Special tokens:", self.tokenizer.special_tokens_map)
        print("Vocabulary size:", self.tokenizer.vocab_size)
        print("Start token:", self.tokenizer.bos_token_id)
        print("End token:", self.tokenizer.eos_token_id)
        print("Pad token:", self.tokenizer.pad_token_id)
        
        # Image preprocessing pipeline
        self.image_transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocess a single image
        Args:
            image: PIL Image or path to image
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        return self.image_transform(image)
    
    def preprocess_text(
        self,
        text: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess text data
        Args:
            text: Input text or list of texts
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
        Returns:
            Dict with input_ids and attention_mask
        """
        # Handle single string input
        if isinstance(text, str):
            text = [text]
            
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding='max_length' if padding else False,
            max_length=config.data.max_length if truncation else None,
            truncation=truncation,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded.input_ids,
            'attention_mask': encoded.attention_mask
        }
    
    def get_vocab_size(self) -> int:
        """Get size of tokenizer vocabulary"""
        return self.tokenizer.vocab_size
    
    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """
        Decode token ids to text
        Args:
            token_ids: Tensor of token ids [batch_size, seq_len]
        Returns:
            List of decoded texts
        """
        return self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True
        )
    
    def get_start_token_id(self) -> int:
        """Get start token ID for generation"""
        return self.tokenizer.bos_token_id
    
    def get_end_token_id(self) -> int:
        """Get end token ID for generation"""
        return self.tokenizer.eos_token_id
