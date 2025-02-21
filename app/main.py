from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
import torch
from PIL import Image
import io
from .model import load_model

app = FastAPI(title="Image Description Generator API")

# Load the model at startup
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = load_model()

class PredictionInput(BaseModel):
    image_url: Optional[str] = None
    ground_truth_description: Optional[str] = None

class PredictionOutput(BaseModel):
    generated_description: str
    reference_description: Optional[str] = None

@app.post("/generate-description", response_model=PredictionOutput)
async def generate_description(input_data: PredictionInput):
    try:
        # Process the image and generate description
        result = model.generate_description(  # Call the method directly instead of calling the model
            image_url=input_data.image_url,
            reference_text=input_data.ground_truth_description
        )
        
        return PredictionOutput(
            generated_description=result['generated_text'],
            reference_description=input_data.ground_truth_description
        )
    except Exception as e:
        print(f"Error in generate_description: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-image-and-describe")
async def upload_image_and_describe(
    file: UploadFile = File(...),
    ground_truth: Optional[str] = None
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        result = model(
            image=image,
            reference_text=ground_truth
        )
        
        return PredictionOutput(
            generated_description=result['generated_text'],
            reference_description=ground_truth
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 