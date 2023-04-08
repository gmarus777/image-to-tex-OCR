from http import HTTPStatus
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from albumentations.augmentations.geometric.resize import Resize
import torch.nn.functional as F


from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from server_api.transform import Image_Transforms
from Tokenizer.Tokenizer import token_to_strings



app = FastAPI(
    title="Convert Image to Latex ",
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
    MAX_RATIO =8
    GOAL_HEIGHT =128
    max_H = 128
    max_W = 1024


    image = Image.open(file.file).convert('RGB')
    image = np.asarray(image)
    h, w, c = image.shape
    ratio = w / h
    if ratio == 0:
        ratio = 1
    if ratio > MAX_RATIO:
        ratio = MAX_RATIO

    new_h = GOAL_HEIGHT
    new_w = int(new_h * ratio)
    image = Resize(interpolation=cv2.INTER_LINEAR, height=new_h, width=new_w, always_apply=True)(image=image)['image']

    image_tensor = transform(image=np.array(image))['image']#[:1]


    image_tensor = F.pad(image_tensor, (0, max_W - new_w, 0, max_H - new_h), value=0)

    pred = lit_model(image_tensor.unsqueeze(0))

    latex_code = token_to_strings(tokens=pred)

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"pred": latex_code},
    }
    return response

