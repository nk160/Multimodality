import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPTokenizer
from typing import Dict, List, Tuple
import sys
import os

# Add src to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.config import config

class Flickr30kDataset(Dataset):
    """Dataset class for Flickr30k"""
    
    def __init__(self, split: str = "test"):
        """
        Initialize the dataset
        Args:
            split (str): Dataset split (currently only 'test' is available)
        """
        if split != "test":
            raise ValueError(f"Only 'test' split is available for {config.data.dataset_name}")
        
        self.dataset = load_dataset(config.data.dataset_name, split=split)
        self.tokenizer = CLIPTokenizer.from_pretrained(config.model.clip_model_name)
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((config.data.image_size, config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset
        Args:
            idx (int): Index of the item
        Returns:
            Dict containing image and tokenized caption
        """
        item = self.dataset[idx]
        
        # Process image
        image = item['image']  # This is already a PIL Image
        image = self.image_transform(image)
        
        # Get caption (key might be 'caption' instead of 'captions')
        caption = item['caption'] if 'caption' in item else item['text']
        
        # Handle if caption is a list
        if isinstance(caption, list):
            caption = caption[0]
        
        tokenized = self.tokenizer(
            caption,
            padding='max_length',
            max_length=config.data.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': tokenized.input_ids.squeeze(0),
            'attention_mask': tokenized.attention_mask.squeeze(0),
            'caption': caption  # Keep original caption for reference
        }

def get_dataloader(split: str = "test") -> DataLoader:
    """
    Create a DataLoader for the specified split
    Args:
        split (str): Dataset split (currently only 'test' is available)
    Returns:
        DataLoader for the specified split
    """
    dataset = Flickr30kDataset(split=split)
    batch_size = (config.data.train_batch_size if split == "train" 
                 else config.data.eval_batch_size)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=config.data.num_workers,
        pin_memory=True
    )
