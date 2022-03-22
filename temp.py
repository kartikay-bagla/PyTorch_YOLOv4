import cv2
import numpy as np
import torch
from utils.datasets import letterbox
from models.maskhead import MaskHead

from utils.general import non_max_suppression

from PIL import Image

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/test')

def resize_image(img, bboxes=None, side_length=416):
    # get height and width of image
    h, w = img.shape[:2]

    # if it has 2 dimensions only i.e mask, create 2 dim array
    if len(img.shape) == 2:
        new_img = np.zeros((side_length, side_length), dtype=img.dtype)
    # otherwise create a 3 dim array with 3 channels
    else:
        new_img = np.zeros((side_length, side_length, 3), dtype=img.dtype)

    # convert to PIL image
    pil_img = Image.fromarray(img)
    # image.thumbnail converts the image to 416, x or x, 416 based on the whichever dimension is bigger
    pil_img.thumbnail((side_length, side_length))
    # get the new dimensions
    new_w, new_h = pil_img.size

    # calculate scaling factor
    scale_factor = side_length / h if h > w else side_length / w

    # since we center the image, need to calculate the offset for the start of the actual image
    h_start = 0 if h > w else side_length // 2 - new_h // 2
    w_start = 0 if w > h else side_length // 2 - new_w // 2

    # copy over the resized image into the square image
    new_img[h_start:h_start+new_h, w_start:w_start+new_w] = np.array(pil_img)

    # update bounding boxes
    if bboxes:
        bboxes = [[i*scale_factor for i in bbox] for bbox in bboxes]
        for bbox in bboxes:
            bbox[0] = bbox[0] + w_start
            bbox[1] = bbox[1] + h_start

    # return values
    if bboxes:
        return new_img, bboxes
    else:
        return new_img

model = MaskHead()

img_names = [
    "test1.jpg",
    "test2.jpg"
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
