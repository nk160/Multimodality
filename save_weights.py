import wandb

def save_to_wandb():
    wandb.init(project="Multimodality", id="qoh9ac7v")  # Use the same run ID from your training
    
    # Log the model weights
    artifact = wandb.Artifact('model-weights', type='model')
    artifact.add_file('checkpoints/best_model.pt')
    wandb.log_artifact(artifact)
    
    wandb.finish()

if __name__ == "__main__":
    save_to_wandb() 