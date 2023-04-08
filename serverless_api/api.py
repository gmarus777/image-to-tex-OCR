
"""AWS Lambda function"""

import smart_open
import cv2
import numpy as np
from PIL import Image
import torch
from albumentations.augmentations.geometric.resize import Resize
import torch.nn.functional as F
#from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from server_api.transform import Image_Transforms
from Tokenizer.Tokenizer import token_to_strings

MAX_RATIO =8
GOAL_HEIGHT =128
max_H = 128
max_W = 1024

model = torch.jit.load("Models_Parameters_Log/scripted_model1.pt")


def handler(event, _context):
    """Provide main prediction API"""
    image_tensor = _load_image(event)
    pred = model(image_tensor.unsqueeze(0))
    latex_code = token_to_strings(tokens=pred)
    return {"pred": latex_code}


def _load_image(event):
    image_url = event.get("image_url")
    if image_url is None:
        return "no image_url provided in event"
    print("INFO url {}".format(image_url))
    image_file = smart_open.open(image_url, "rb")
    image = Image.open(image_file).convert('RGB')
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

    image_tensor = Image_Transforms.test_transform_with_padding(image=np.array(image))['image']

    image_tensor = F.pad(image_tensor, (0, max_W - new_w, 0, max_H - new_h), value=0)

    return image_tensor


