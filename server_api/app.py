from http import HTTPStatus

import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch


from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from Data.image_transforms import Image_Transforms
from Tokenizer.Tokenizer import token_to_strings



app = FastAPI(
    title="Image to Latex Convert",
    desription="Convert an image of math equation into LaTex code.",
)


@app.on_event("startup")
async def load_model():
    global lit_model
    global transform
    lit_model = torch.jit.load("Models_Parameters_Log/scripted_model1.pt")
    transform = Image_Transforms.test_transform_with_padding


@app.get("/", tags=["General"])
def read_root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict/", tags=["Prediction"])
def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert('RGB')
    image_tensor = transform(image=np.array(image))['image'][:1]



    pred = lit_model(image_tensor.unsqueeze(0))

    latex_code = token_to_strings(tokens=pred)

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"pred": latex_code},
    }
    return response