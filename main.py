from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version
import shutil
import os

app = FastAPI()

class PredictionOut(BaseModel):
    diagnosis: str

@app.get("/")
def home():
    return {"health_check": "ok", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
async def predict(file: UploadFile = File(...)):
    try:
        
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_pipeline(file_path)


        os.remove(file_path)

        return {"diagnosis": result}

    except Exception as e:
        return {"error": str(e)}
