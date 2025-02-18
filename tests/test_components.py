import sys
import os
import torch
import pytest
from PIL import Image
import numpy as np
import torch.nn.functional as F

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config
from src.data.dataloader import Flickr30kDataset, get_dataloader
from src.model.encoder import CLIPEncoder
from src.model.decoder import Decoder

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

def test_decoder():
    """Test the decoder's functionality"""
    # Create dummy data
    batch_size = 2
    seq_length = 10
    vocab_size = 1000
    
    # Dummy image features from CLIP
    image_features = torch.randn(batch_size, 196, config.model.hidden_size)
    
    # Dummy text tokens
    text_tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Initialize decoder
    decoder = Decoder(vocab_size=vocab_size)
    
    # Test forward pass (logits only)
    logits = decoder(image_features, text_tokens)
    assert logits.shape == (batch_size, seq_length, vocab_size)
    
    # Test forward pass with probabilities
    logits, probs = decoder(image_features, text_tokens, return_probs=True)
    assert logits.shape == (batch_size, seq_length, vocab_size)
    assert probs.shape == (batch_size, seq_length, vocab_size)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, seq_length))
    
    # Test generation
    generated = decoder.generate(image_features)
    assert generated.shape[0] == batch_size
    assert generated.shape[1] <= config.data.max_length
    
    # Test error handling
    with pytest.raises(ValueError):
        # Test with mismatched hidden sizes
        wrong_features = torch.randn(batch_size, 196, config.model.hidden_size + 1)
        decoder(wrong_features, text_tokens)
    
    with pytest.raises(ValueError):
        # Test with invalid text tokens
        invalid_tokens = torch.randint(vocab_size, vocab_size + 10, (batch_size, seq_length))
        decoder(image_features, invalid_tokens)
    
    with pytest.raises(ValueError):
        # Test with invalid temperature
        decoder.generate(image_features, temperature=-1.0)
    
    with pytest.raises(ValueError):
        # Test with invalid max_length
        decoder.generate(image_features, max_length=0)

if __name__ == "__main__":
    # Run tests
    test_dataloader()
    test_encoder()
    test_decoder()
    print("All tests passed!") 