import cv2
import numpy as np
import torch
from utils.datasets import letterbox
from models.maskhead import MaskHead

from utils.general import non_max_suppression
from utils.data_loader import resize_image

from PIL import Image

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/test')

model = MaskHead(torch.device("cuda")).to("cuda")

img_names = [
    "input.jpg"
]
images = []
for img_name in img_names:
    image = cv2.imread(img_name)
    image = resize_image(image, side_length=416)
    image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    image = image / 255.
    images.append(image)
images = torch.Tensor(images).to("cuda")
model.eval()
import time

t1 = time.time()
mask_outputs, roi_align_boxes_input = model(images)
t2 = time.time()
print("Time taken: ", t2-t1)

print(mask_outputs.shape)
print(roi_align_boxes_input)

# writer.add_graph(model, image)
# writer.close()
