import torch
from torch import nn
from torchvision.ops import roi_align, batched_nms
from models.models import Darknet, load_darknet_weights
from utils.general import non_max_suppression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

YOLO_OUTPUT_FILTERS_PER_LAYER = (256, 512, 1024)
YOLO_OUTPUT_FILTERS = sum(YOLO_OUTPUT_FILTERS_PER_LAYER)
MASK_CONV_FILTERS = 256
NUM_CLASSES = 80
MAX_BOXES_PER_IMAGE = 300
NMS_IOU_THRESHOLD = 0.5
ROI_ALIGN_OUTPUT_SIDE = 14
ROI_ALIGN_OUTPUT_SCALES = (8, 16, 32)


def get_mask_conv_layer(in_channels, out_channels, activation=None):
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    ]
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)

class MaskHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.yolo_model = Darknet("cfg/yolov4.cfg").to(device)
        load_darknet_weights(self.yolo_model, "weights/yolov4.weights")
        for param in self.yolo_model.parameters():
            param.requires_grad = False

        self.mask_head = nn.Sequential(
            get_mask_conv_layer(
                in_channels=YOLO_OUTPUT_FILTERS,
                out_channels=MASK_CONV_FILTERS,
                activation=nn.ReLU()
            ),
            get_mask_conv_layer(
                in_channels=MASK_CONV_FILTERS,
                out_channels=MASK_CONV_FILTERS,
                activation=nn.ReLU()
            ),
            get_mask_conv_layer(
                in_channels=MASK_CONV_FILTERS,
                out_channels=MASK_CONV_FILTERS,
                activation=nn.ReLU()
            ),
            nn.ConvTranspose2d(
                in_channels=MASK_CONV_FILTERS,
                out_channels=MASK_CONV_FILTERS,
                kernel_size=2,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(MASK_CONV_FILTERS),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=MASK_CONV_FILTERS,
                out_channels=NUM_CLASSES,
                kernel_size=1,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(NUM_CLASSES),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        if self.training:
            yolo_output, yolo_features = self.yolo_model(x)
        else:
            yolo_reshaped_output, p, yolo_features, yolo_output = self.yolo_model(x)

        yolo_output, _ = zip(*yolo_output)
        yolo_output = torch.cat(yolo_output, dim=1)
        
        nms_preds = non_max_suppression(yolo_output, 0.4, 0.5)
        total_boxes = sum(len(preds) for preds in nms_preds)
        roi_align_boxes_input = torch.zeros((
            total_boxes, 5
        )).to(device)
        counter = 0
        for i, preds in enumerate(nms_preds):
            roi_align_boxes_input[
                counter:counter + len(preds), 1:
            ] = preds[:, :4]
            roi_align_boxes_input[
                counter:counter + len(preds), 0
            ] = i


        final_tensor = torch.zeros((
            total_boxes,
            YOLO_OUTPUT_FILTERS,
            ROI_ALIGN_OUTPUT_SIDE,
            ROI_ALIGN_OUTPUT_SIDE
        )).to(device)
        yolo_feature_filter_counter = 0
        for i, yolo_feature_filters in enumerate(YOLO_OUTPUT_FILTERS_PER_LAYER):
            final_tensor[
                :,
                yolo_feature_filter_counter:yolo_feature_filter_counter + yolo_feature_filters,
            ] = roi_align(
                input=yolo_features[i],
                boxes=roi_align_boxes_input,
                output_size=ROI_ALIGN_OUTPUT_SIDE,
                spatial_scale=1/ROI_ALIGN_OUTPUT_SCALES[i],
                sampling_ratio=0,
                aligned=True
            )
            yolo_feature_filter_counter += yolo_feature_filters
            

        mask_outputs = self.mask_head(final_tensor)
        return yolo_output, mask_outputs, nms_preds