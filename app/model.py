import wandb
import torch
import os
from PIL import Image
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from typing import Union, Optional
from src.model.decoder import Decoder
from src.config import config  # Add this import

class ImageDescriptionModel:
    def __init__(self, clip_model, decoder_model):
        self.clip = clip_model
        self.decoder = decoder_model
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def process_image(self, image: Union[str, Image.Image]):
        if isinstance(image, str):
            # Load image from URL
            response = requests.get(image)
            image = Image.open(BytesIO(response.content))
        
        # Process image through CLIP
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs.to(self.device)
    
    def generate_description(self, image_inputs, reference_text: Optional[str] = None):
        with torch.no_grad():
            # Get image features from CLIP
            image_features = self.clip.get_image_features(**image_inputs)
            
            # Process reference text if provided
            text_features = None
            if reference_text:
                text_inputs = self.processor(text=reference_text, return_tensors="pt", padding=True)
                text_features = self.clip.get_text_features(**text_inputs.to(self.device))
            
            # Generate description using your decoder
            output = self.decoder(
                image_features=image_features,
                text_features=text_features
            )
            
            return output

def load_model():
    # Configure wandb
    wandb.login(key=os.environ["WANDB_API_KEY"])
    
    # Initialize CLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    try:
        # Try to load checkpoint
        CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        decoder_model = Decoder(
            vocab_size=config.model.vocab_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.decoder_layers,
            num_heads=config.model.decoder_attention_heads,
            max_length=config.data.max_length,
            dropout=config.model.dropout
        )
        
        decoder_model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded checkpoint from", checkpoint_path)
    except FileNotFoundError:
        print("No checkpoint found, initializing new decoder")
        decoder_model = Decoder(
            vocab_size=config.model.vocab_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.decoder_layers,
            num_heads=config.model.decoder_attention_heads,
            max_length=config.data.max_length,
            dropout=config.model.dropout
        )
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    
    decoder_model.to(device)
    return ImageDescriptionModel(clip_model, decoder_model)
    
    pass  # Replace with actual implementation 