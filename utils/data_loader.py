from pycocotools.coco import COCO
from functools import reduce
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
import pandas as pd
from torchvision.io import read_image
import numpy as np

import cv2
from PIL import Image

USED_CATEGORIES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]
USED_CATEGORIES_TO_IDS = {cat: i for i, cat in enumerate(USED_CATEGORIES)}

COCO_CAT_IDS_TO_USED_IDS = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 
    18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 
    34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 
    48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 
    61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 
    78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
}


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
    return (new_img, bboxes) if bboxes else new_img


def plot_image_and_box(img, box=(0, 0, 0, 0)):
    x,y,w,h = box
    plt.imshow(img)
    plt.plot([x, x], [y, y+h], linewidth=5, color="blue")
    plt.plot([x, x+w], [y, y], linewidth=5, color="blue")
    plt.plot([x, x+w], [y+h, y+h], linewidth=5, color="blue")
    plt.plot([x+w, x+w], [y, y+h], linewidth=5, color="blue")
    plt.show()

class CustomImageDataset(Dataset):
    def __init__(self, annotation_file, image_folder, batch_size=32, side_length=416, mask_side=52):
        self.batch_size = batch_size
        self.side_length = side_length
        self.mask_side = mask_side
        self.image_folder = image_folder

        self.coco = COCO(annotation_file=annotation_file)
        # get coco ids for categories
        coco_ids_for_used_categories = self.coco.getCatIds(catNms=USED_CATEGORIES)
        # get all image ids per category and filter out duplicates
        self.img_ids = list(reduce(lambda a, b: a.union(b), [set(self.coco.getImgIds(catIds=[i])) for i in coco_ids_for_used_categories]))


    def __len__(self):
        return len(self.img_ids) // self.batch_size

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # filter out relevant rows based on index
        relevant_img_ids = self.img_ids[idx*self.batch_size: (idx+1)*self.batch_size]

        # load annotations
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=relevant_img_ids))

        all_imgs, all_bboxes, all_class_ids, all_masks = [], [], [], []
        for coco_img in self.coco.loadImgs(relevant_img_ids):
            # filter annotations for this image
            anns = [ann for ann in annotations if ann['image_id'] == coco_img['id']]
            # temp storage of these values
            bboxes, class_ids, batch_masks = [], [], []
            for ann in anns:
                try:
                    x1, y1, w, h = ann['bbox']
                    x2, y2 = x1+w, y1+h
                    x1, y1, x2, y2 = map(lambda x: int(round(x)), (x1, y1, x2, y2))
                    
                    # since mask is on entire image, extract roi from mask and resize
                    mask = self.coco.annToMask(ann)
                    mask = mask[y1:y2,x1:x2]
                    mask = resize_image(mask, side_length=self.mask_side)
                    batch_masks.append(mask)
                    # map coco ids (91) to our ids (80)
                    class_ids.append(COCO_CAT_IDS_TO_USED_IDS[ann['category_id']])
                    bboxes.append([x1, y1, x2, y2])
                except:
                    pass

            all_bboxes.append(bboxes)
            all_class_ids.append(class_ids)
            all_masks.append(batch_masks)

            img = cv2.imread(self.image_folder + coco_img['file_name'])
            # resize image and bboxes
            img, bboxes = resize_image(img, bboxes, self.side_length)
            # bgr to rgb, channels first, scaling
            img = img[:,:,::-1].transpose(2,0,1) / 255.

            all_imgs.append(img)

        # output tensors
        images = torch.zeros((self.batch_size, 3, self.side_length, self.side_length))
        total_anns = sum(len(i) for i in all_bboxes)
        targets = torch.zeros((total_anns, 7))
        masks = torch.zeros((total_anns, self.mask_side, self.mask_side))

        # used to keep track of current annotation index
        ann_counter = 0
        for batch_index, (img, bboxes, class_ids, batch_masks) in enumerate(zip(all_imgs, all_bboxes, all_class_ids, all_masks)):
            # update tensors
            images[batch_index] = torch.tensor(img)
            targets[ann_counter:ann_counter+len(bboxes), 0] = batch_index
            targets[ann_counter:ann_counter+len(bboxes), 1:5] = torch.tensor(bboxes)
            targets[ann_counter:ann_counter+len(bboxes), 5] = 1
            targets[ann_counter:ann_counter+len(bboxes), 6] = torch.tensor(class_ids)

            # convert to np array first since torch conversion of list of numpy arrays is slower than this
            batch_masks = torch.tensor(np.array(batch_masks))
            masks[ann_counter:ann_counter+len(bboxes)] = batch_masks

            # update counter
            ann_counter += len(bboxes)

        return images, targets, masks
