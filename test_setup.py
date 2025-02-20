import torch
from transformers import CLIPTokenizer, CLIPModel

def test_setup():
    # Check GPU
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Test CLIP model loading
    model_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    print("CLIP model and tokenizer loaded successfully")

if __name__ == "__main__":
    test_setup() 