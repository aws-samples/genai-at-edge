import numpy as np
from PIL import Image
import cv2
import os
import clip
from .tensorrt_utils import TrtModel

mean = np.array([0.48145466, 0.4578275, 0.40821073])
std = np.array([0.26862954, 0.26130258, 0.27577711])

class VIT:
    def __init__(self, image_model_path='ViT-B-32-IMAGE.trt', text_model_path='ViT-B-32-TEXT.trt'):
        self.vit_visual = TrtModel(image_model_path,dtype=np.float32,max_batch_size=1,model_height=224,model_width=224)
        self.vit_textual = TrtModel(text_model_path,dtype=np.int32,max_batch_size=1,model_height=77,model_width=1)
        self.image = np.zeros([224, 224, 3], np.uint8)

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def format_results(self, boxes, masks, scores, filter=0):
        annotations = []
        n = len(masks)
        for i in range(n):
            annotation = {}
            mask = masks[i] == 1.0
            if np.sum(mask) < filter:
                continue
            annotation["id"] = i
            annotation["segmentation"] = mask
            annotation["bbox"] = boxes[i]
            annotation["score"] = scores[i]
            annotation["area"] = annotation["segmentation"].sum()
            annotations.append(annotation)
        return annotations

    def text_prompt(self, annotations, text_prompt):
        cropped_images, cropped_boxes, not_crop, filter_id, annotations = self.crop_image(annotations)
        scores = self.retriev(cropped_images, text_prompt)
        max_idx = scores.argsort()
        max_idx = max_idx[-1]
        max_idx += sum(np.array(filter_id) <= int(max_idx))
        return annotations[max_idx]["segmentation"], max_idx

    def point_prompt(self, masks, points, pointlabel):
        h = masks[0]["segmentation"].shape[0]
        w = masks[0]["segmentation"].shape[1]
        onemask = np.zeros((h, w))
        for i, annotation in enumerate(masks):
            if type(annotation) == dict:
                mask = annotation["segmentation"]
            else:
                mask = annotation
            for i, point in enumerate(points):
                if mask[point[1], point[0]] == 1 and pointlabel[i] == 1:
                    onemask += mask
                if mask[point[1], point[0]] == 1 and pointlabel[i] == 0:
                    onemask -= mask
        onemask = onemask >= 1
        return onemask, 0

    def box_prompt(masks, bbox):
        h = masks.shape[1]
        w = masks.shape[2]
        bbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
        bbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
        bbox[2] = round(bbox[2]) if round(bbox[2]) < w else w
        bbox[3] = round(bbox[3]) if round(bbox[3]) < h else h
        bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        masks_area = torch.sum(masks[:, bbox[1] : bbox[3], bbox[0] : bbox[2]], dim=(1, 2))
        orig_masks_area = torch.sum(masks, dim=(1, 2))
        union = bbox_area + orig_masks_area - masks_area
        IoUs = masks_area / union
        max_iou_index = torch.argmax(IoUs)
        return masks[max_iou_index].cpu().numpy(), max_iou_index

    def prompt(self, results, box_prompt=[[0,0,0,0]], point_prompt=[[0,0]], point_label=[0], text_prompt="", box=None, point=None, text=None):
        ori_h, orig_w = self.image[:2]
        if box:
            mask, idx = self.box_prompt(results, convert_box_xywh_to_xyxy(box_prompt))
        elif point:
            mask, idx = self.point_prompt(results, point_prompt, point_label)
        elif text:
            mask, idx = self.text_prompt(results, text_prompt)
        else:
            return None
        return mask

    def preprocess(self, image_in):
        image_in = cv2.resize(image_in, (224,224))
        image_in = image_in.astype(np.float32)
        image_in = image_in / 255.
        image_in = (image_in - mean) / std
        image_in = image_in.transpose(2, 0, 1)
        image_in = image_in[np.newaxis, :, :, :]
        return image_in

    def retriev(self, elements, search_text):
        preprocessed_images = [self.preprocess(image) for image in elements]
        tokenized_text = clip.tokenize([search_text]).numpy()
        image_features = np.asarray([self.vit_visual(ft)[0][0].copy() for ft in preprocessed_images])
        text_features = np.asarray([self.vit_textual(ft)[0][0].copy() for ft in tokenized_text])
        image_features /= np.linalg.norm(image_features, axis=-1, keepdims=True)
        text_features /= np.linalg.norm(text_features, axis=-1, keepdims=True)
        probs = 100.0 * image_features @ text_features.T
        return self.softmax(probs[:, 0])

    def crop_image(self, annotations):
        ori_h, ori_w = self.image.shape[:2]
        mask_h, mask_w = annotations[0]["segmentation"].shape
        if ori_w != mask_w or ori_h != mask_h:
            image = self.image.resize((mask_w, mask_h))
        else:
            image = self.image.copy()
        cropped_boxes = []
        cropped_images = []
        not_crop = []
        filter_id = []
        for _, mask in enumerate(annotations):
            if np.sum(mask["segmentation"]) <= 100:
                filter_id.append(_)
                continue
            bbox = self.get_bbox_from_mask(mask["segmentation"])
            cropped_images.append(self.segment_image(bbox, image))
            cropped_boxes.append(bbox)
        return cropped_images, cropped_boxes, not_crop, filter_id, annotations

    def get_bbox_from_mask(self, mask):
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x1, y1, w, h = cv2.boundingRect(contours[0])
        x2, y2 = x1 + w, y1 + h
        if len(contours) > 1:
            for b in contours:
                x_t, y_t, w_t, h_t = cv2.boundingRect(b)
                x1 = min(x1, x_t)
                y1 = min(y1, y_t)
                x2 = max(x2, x_t + w_t)
                y2 = max(y2, y_t + h_t)
            h = y2 - y1
            w = x2 - x1
        return [x1, y1, x2, y2]

    def segment_image(self, bbox, image):
        image_array = np.array(image)
        segmented_image_array = np.zeros_like(image_array)
        x1, y1, x2, y2 = bbox
        segmented_image_array[y1:y2, x1:x2] = image_array[y1:y2, x1:x2]
        black_image = np.ones(image.shape)*255
        transparency_mask = np.zeros(image_array.shape[:3], dtype=np.uint8)
        transparency_mask[y1:y2, x1:x2] = 255
        segmented_image = cv2.bitwise_and(segmented_image_array, transparency_mask)
        black_image = black_image*(1-transparency_mask/255) + segmented_image
        return black_image
