import torch
from torch import nn
from torchvision.ops import roi_align, batched_nms
from models.models import Darknet, load_darknet_weights
from utils.general import non_max_suppression

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
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device

        self.yolo_model = Darknet("cfg/yolov4.cfg").to(self.device)
        load_darknet_weights(self.yolo_model, "yolov4.weights")
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
            nn.BatchNorm2d(NUM_CLASSES)
        ).to(self.device)

    def forward(self, x, targets=None):
        # targets should be [N, 7] where N is the sum of the
        # number of boxes in all images and the columns are
        # (batch_index, x1, y1, x2, y2, conf=1, class_id)
        if self.training:
            # if training, targets is already provided
            _, yolo_features = self.yolo_model(x)
        else:
            # if not training, need to get targets from yolo output
            yolo_reshaped_output, p, yolo_features, yolo_output = self.yolo_model(x)

            reshaped_yolo_output, _ = zip(*yolo_output)
            reshaped_yolo_output = torch.cat(reshaped_yolo_output, dim=1)

            nms_targets = non_max_suppression(reshaped_yolo_output, 0.4, 0.5)
            total_targets = sum(len(t) for t in nms_targets)
            targets = torch.zeros((total_targets, 7), dtype=torch.float32).to(self.device)
            counter = 0
            for i, batch in enumerate(nms_targets):
                targets[counter:counter + len(batch), 0] = i
                targets[counter:counter+len(batch), 1:] = batch
                counter += len(batch)

        final_tensor = torch.zeros((
            targets.shape[0],
            YOLO_OUTPUT_FILTERS,
            ROI_ALIGN_OUTPUT_SIDE,
            ROI_ALIGN_OUTPUT_SIDE
        )).to(self.device)
        yolo_feature_filter_counter = 0
        for i, yolo_feature_filters in enumerate(YOLO_OUTPUT_FILTERS_PER_LAYER):
            final_tensor[
                :,
                yolo_feature_filter_counter:yolo_feature_filter_counter + yolo_feature_filters,
            ] = roi_align(
                input=yolo_features[i],
                boxes=targets[:, :5], # we only need batch_index and box coordinates
                output_size=ROI_ALIGN_OUTPUT_SIDE,
                spatial_scale=1/ROI_ALIGN_OUTPUT_SCALES[i],
                sampling_ratio=0,
                aligned=True
            )
            yolo_feature_filter_counter += yolo_feature_filters

        mask_outputs = self.mask_head(final_tensor)

        if self.training:
            # [N] contains the batch index of each mask
            # [N, 80, 28, 28], [N]
            return mask_outputs, targets[:, 0] 
        else:
            # [N, 80, 28, 28], [N, 7] where N is the total number of boxes
            # and the columns are (batch_index, x1, y1, x2, y2, conf, class_id)
            return mask_outputs, targets
