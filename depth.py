import torch
import urllib.request
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

img = cv2.imread("data\imgdata\\1_18_1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_tensor = transform(img)
print(input_tensor.shape)
with torch.no_grad():
    prediction = midas(input_tensor)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).squeeze()
print(prediction.shape)