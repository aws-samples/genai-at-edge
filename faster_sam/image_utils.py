import numpy as np
import cv2

class ImageUtils:
    def __init__(self):
        self.orig_height, self.orig_width = 0, 0
        self.resized_height, self.resized_width = 0, 0
        self.scale_x, self.scale_y, self.pad_x, self.pad_y = 0.0, 0.0, 0.0, 0.0
        return

    def read_image_path(self, path, rgb=True):
        self.orig_image = cv2.imread(path)
        if rgb: self.orig_image_RGB = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2RGB)
        else: self.orig_image_RGB = self.orig_image
        image_in = self.resize_with_padding(self.orig_image)
        if rgb: image_in_RGB = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
        else: image_in_RGB = image_in
        return self.orig_image, self.orig_image_RGB, image_in, image_in_RGB

    def read_image_camera(self, cap, rgb=True):
        self.orig_image = cap.read()[1]
        if rgb: self.orig_image_RGB = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2RGB)
        else: self.orig_image_RGB = self.orig_image
        image_in = self.resize_with_padding(self.orig_image)
        if rgb: image_in_RGB = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
        else: image_in_RGB = image_in
        return self.orig_image, self.orig_image_RGB, image_in, image_in_RGB

    def resize_with_padding(self, image, size=512):
        height, width = image.shape[:2]
        scale = size / max(height, width)
        if height > width: resized_image = cv2.resize(image, (int(width * scale), size))
        else: resized_image = cv2.resize(image, (size, int(height * scale)))
        new_height, new_width = resized_image.shape[:2]
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        x_start = (size - new_width) // 2
        y_start = (size - new_height) // 2
        canvas[y_start:y_start+new_height, x_start:x_start+new_width] = resized_image
        self.scale_x = new_width/width
        self.scale_y = new_height/height
        self.pad_x = x_start
        self.pad_y = y_start
        return canvas

    def remove_padding(self, image):
        height, width, _ = image.shape
        return image[self.pad_y:height-self.pad_y, self.pad_x:width-self.pad_x]

    def scale_box_remove_padding(self, box):
        new_box = box - np.array([self.pad_y, self.pad_x, self.pad_y, self.pad_x])
        new_box = new_box / np.array([self.scale_y, self.scale_x, self.scale_y, self.scale_x])
        return new_box

    def scale_mask_remove_padding(self, mask):
        height, width = mask.shape
        new_mask = mask[self.pad_y:height-self.pad_y, self.pad_x:width-self.pad_x]
        new_mask = cv2.resize(new_mask, None, fx=1./self.scale_x, fy=1./self.scale_y)
        return new_mask

    def fast_process(self, annotations, bbox=None, points=None, edges=False):
        if isinstance(annotations[0], dict):
            annotations = [annotation["segmentation"] for annotation in annotations]
        image = self.orig_image_RGB.copy()
        original_h = image.shape[0]
        original_w = image.shape[1]
        scaled_annotations = [[]]*len(annotations)
        for i, mask in enumerate(annotations):
            mask = mask.astype(np.uint8)
            mask = self.scale_mask_remove_padding(mask)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            scaled_annotations[i] = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))
        annotations = np.array(scaled_annotations)
        mask = self.fast_show_mask(annotations, bbox=bbox, points=points, target_height=original_h, target_width=original_w,)
        image = cv2.addWeighted(np.uint8(mask[:,:,:3]*255), 0.5, image, 1, 0)
        contour_all = []
        temp = np.zeros((original_h, original_w, 1))
        for i, mask in enumerate(annotations):
            if type(mask) == dict:
                mask = mask["segmentation"]
            annotation = mask.astype(np.uint8)
            contours, hierarchy = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_all.append(contour)
        cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2)
        color = np.array([0 / 255, 0 / 255, 255 / 255, 0.8])
        contour_mask = temp/255 * color.reshape(1, 1, -1)
        image = cv2.addWeighted(np.uint8(cv2.merge([temp,temp,temp])), 0.2, image, 1, 0)
        return image

    def fast_show_mask(self, annotation, bbox=None, points=None, target_height=960, target_width=960,):
        mask_sum = annotation.shape[0]
        height = annotation.shape[1]
        weight = annotation.shape[2]
        areas = np.sum(annotation, axis=(1, 2))
        sorted_indices = np.argsort(areas)
        annotation = annotation[sorted_indices]
        index = (annotation != 0).argmax(axis=0)
        color = np.random.random((mask_sum, 1, 1, 3))
        transparency = np.ones((mask_sum, 1, 1, 1)) * 0.6
        visual = np.concatenate([color, transparency], axis=-1)
        mask_image = np.expand_dims(annotation, -1) * visual
        show = np.zeros((height, weight, 4))
        h_indices, w_indices = np.meshgrid(np.arange(height), np.arange(weight), indexing="ij")
        indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))
        show[h_indices, w_indices, :] = mask_image[indices]
        return show