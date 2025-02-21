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
from datasets import load_dataset

class ImageDescriptionModel:
    def __init__(self, clip_model, decoder_model):
        self.clip = clip_model
        self.decoder = decoder_model
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __call__(self, image_url=None, reference_text=None, image=None):
        """Make the model callable"""
        try:
            # Process image
            if image_url:
                image_inputs = self.process_image(image_url)
            elif image:
                image_inputs = self.process_image(image)
            else:
                raise ValueError("Either image_url or image must be provided")
            
            # Generate description
            with torch.no_grad():
                # Get image features
                image_features = self.clip.get_image_features(**image_inputs)
                
                # Generate text
                output = self.decoder.generate(
                    image_features=image_features,
                    max_length=config.data.max_length
                )
                
                # Convert output to text
                generated_text = "Generated description here"  # We'll need to implement text decoding
                
                return {
                    "generated_text": generated_text,
                    "status": "success"
                }
                
        except Exception as e:
            print(f"Error generating description: {str(e)}")
            return {
                "generated_text": "Error generating description",
                "status": "error",
                "error": str(e)
            }
    
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
        # Initialize wandb
        run = wandb.init(
            entity="nigelkiernan-lpt-advisory",
            project="Multimodality"
        )
        
        # Download the specific artifact
        artifact = run.use_artifact('model-weights:latest')
        artifact_dir = artifact.download()
        
        # Load the model weights
        checkpoint_path = os.path.join(artifact_dir, "best_model.pt")
        print(f"Loading weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Print state dict keys for debugging
        print("State dict keys:", checkpoint['model_state_dict'].keys())
        
        # Get only decoder weights
        decoder_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('decoder.'):
                # Convert blocks to layers in the key names
                new_key = k[8:]  # Remove 'decoder.' prefix
                new_key = new_key.replace('blocks', 'layers')
                new_key = new_key.replace('self_attn_query', 'self_attn.q_proj')
                new_key = new_key.replace('self_attn_key', 'self_attn.k_proj')
                new_key = new_key.replace('self_attn_value', 'self_attn.v_proj')
                new_key = new_key.replace('self_attn_out', 'self_attn.out_proj')
                new_key = new_key.replace('cross_attn_query', 'cross_attn.q_proj')
                new_key = new_key.replace('cross_attn_key', 'cross_attn.k_proj')
                new_key = new_key.replace('cross_attn_value', 'cross_attn.v_proj')
                new_key = new_key.replace('cross_attn_out', 'cross_attn.out_proj')
                decoder_state_dict[new_key] = v
        
        decoder_model = Decoder(
            vocab_size=config.model.vocab_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.decoder_layers,
            num_heads=config.model.decoder_attention_heads,
            max_length=config.model.max_position_embeddings,
            dropout=config.model.dropout
        )
        
        # Load the mapped state dict
        decoder_model.load_state_dict(decoder_state_dict)
        print("Loaded decoder weights from checkpoint")
        
    except Exception as e:
        print(f"Error loading model from W&B: {str(e)}")
        raise
    
    decoder_model.to(device)
    return ImageDescriptionModel(clip_model, decoder_model)
    
    pass  # Replace with actual implementation 

def get_test_example():
    """Get a test image and caption from Flickr30k"""
    try:
        # Load just one example from the validation set
        dataset = load_dataset("nlphuji/flickr30k", split="validation[:1]")
        
        # Get the first example
        example = dataset[0]
        
        return {
            "image_url": example["image_url"],
            "ground_truth": example["caption"][0]  # Get first caption
        }
    except Exception as e:
        print(f"Error loading test example: {str(e)}")
        # Fallback to a known working Flickr image
        return {
            "image_url": "http://farm4.staticflickr.com/3712/9413787559_87a8a9db17.jpg",
            "ground_truth": "A man in a blue shirt is standing on a ladder cleaning windows."
        } 