import math
import time
import cv2
import numpy as np
import random
from .tensorrt_utils import TrtModel

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = (50,250,50)

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]
    return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
    # Compute IoU
    iou = intersection_area / union_area
    return iou

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FastSAM:
    def __init__(self, model_path='FastSAM.trt', conf_thres=0.4, iou_thres=0.9, num_masks=32, model_height=512, model_width=512, img_height=512, img_width=512):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks
        self.input_height = model_height
        self.input_width = model_width
        self.img_height = img_height
        self.img_width = img_width

        # Initialize model
        self.initialize_model(model_path)

    def infer(self, image):
        self.image = image.copy()
        return self.segment_objects(image)

    def initialize_model(self, path):
        self.session = TrtModel(engine_path=path, model_height=512, model_width=512)
        self.model_input_shape = self.session.input_shape
        self.model_output_shape_masks, self.model_output_shape_boxes = self.session.output_shape

    def segment_objects(self, image):
        input_tensor = self.prepare_input(image)
        # Perform inference on the image
        outputs = self.inference(input_tensor)
        outputs[0] = outputs[0].reshape(self.model_output_shape_masks)
        outputs[1] = outputs[1].reshape(self.model_output_shape_boxes)
        self.boxes, self.mask_maps, self.scores = self.process_output(outputs)
        return self.boxes, self.mask_maps, self.scores

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        # Resize input image
        input_img = cv2.resize(image, (self.input_width, self.input_height))
        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def inference(self, input_tensor):
        outputs = self.session(input_tensor)
        return outputs

    def process_output(self, outputs):
        ############## BBOXES ##############
        box_output = outputs[1]
        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        if len(scores) == 0:
            return [], [], [], np.array([])
        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]
        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)
        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)
        det_boxes = boxes[indices]
        mask_predictions = mask_predictions[indices]
        det_scores = scores[indices]

        ############## MASKS ##############
        mask_output = outputs[0]
        if mask_predictions.shape[0] == 0:
            return []
        mask_output = np.squeeze(mask_output)
        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))
        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(det_boxes, (self.img_height, self.img_width), (mask_height, mask_width))
        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i in range(len(scale_boxes)):
            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))
            x1 = int(math.floor(det_boxes[i][0]))
            y1 = int(math.floor(det_boxes[i][1]))
            x2 = int(math.ceil(det_boxes[i][2]))
            y2 = int(math.ceil(det_boxes[i][3]))
            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
            crop_mask = cv2.blur(crop_mask, blur_size)
            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        self.det_boxes = det_boxes
        self.mask_maps = mask_maps
        self.scores = det_scores
        return self.det_boxes, self.mask_maps, self.scores

    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]
        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes, (self.input_height, self.input_width), (self.img_height, self.img_width))
        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)
        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)
        return boxes

    def draw_detections(self, mask_alpha=0.3):
        size = min([self.img_height, self.img_width]) * 0.0006
        thickness = int(min([self.img_height, self.img_width]) * 0.001)
        mask_img = self.image.copy()
        # Draw bounding boxes and labels of detections
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box.astype(int)
            # Draw fill mask image
            if self.mask_maps is None:
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            else:
                crop_mask = self.mask_maps[i][y1:y2, x1:x2, np.newaxis]
                crop_mask_img = mask_img[y1:y2, x1:x2]
                crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * colors
                mask_img[y1:y2, x1:x2] = crop_mask_img
        mask_img = cv2.addWeighted(mask_img, mask_alpha, self.image, 1 - mask_alpha, 0)
        # Draw bounding boxes and labels of detections
        for box in self.boxes:
            x1, y1, x2, y2 = box.astype(int)
            # Draw rectangle
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), colors, 2)
            cv2.rectangle(mask_img, (x1, y1), (x1 + thickness, y1 - thickness), colors, -1)
        return mask_img

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])
        return boxes