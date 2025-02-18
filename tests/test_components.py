import sys
import os
import torch
import pytest
from PIL import Image
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config
from src.data.dataloader import Flickr30kDataset, get_dataloader
from src.model.encoder import CLIPEncoder

def test_dataloader():
    """Test the Flickr30k dataset and dataloader"""
    # Initialize dataset
    dataset = Flickr30kDataset(split="test")
    
    # Test single item
    item = dataset[0]
    assert isinstance(item['image'], torch.Tensor)
    assert item['image'].shape == (3, config.data.image_size, config.data.image_size)
    assert isinstance(item['caption'], str)
    assert isinstance(item['input_ids'], torch.Tensor)
    assert isinstance(item['attention_mask'], torch.Tensor)
    
    # Test dataloader
    dataloader = get_dataloader(split="test")
    batch = next(iter(dataloader))
    assert batch['image'].shape == (config.data.train_batch_size, 3, 
                                  config.data.image_size, config.data.image_size)

def test_encoder():
    """Test the CLIP encoder"""
    # Create dummy image
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, config.data.image_size, config.data.image_size)
    
    # Initialize encoder
    encoder = CLIPEncoder()
    
    # Test forward pass
    features = encoder(dummy_image)
    assert features.shape[0] == batch_size
    assert features.shape[-1] == config.model.hidden_size
    
    # Test CLS features
    cls_features = encoder.get_cls_features(dummy_image)
    assert cls_features.shape == (batch_size, config.model.hidden_size)
    
    # Test pooled features
    for pool_type in ['mean', 'max', 'cls']:
        pooled = encoder.get_pooled_features(dummy_image, pool_type=pool_type)
        assert pooled.shape == (batch_size, config.model.hidden_size)
    
    # Test attention maps
    attention_maps = encoder.get_attention_maps(dummy_image)
    assert isinstance(attention_maps, tuple)
    assert len(attention_maps) > 0

if __name__ == "__main__":
    # Run tests
    test_dataloader()
    test_encoder()
    print("All tests passed!") 