import os
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from transformers import SamProcessor
from PIL import Image

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, processor: SamProcessor, no_prompt=False):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.no_prompt = no_prompt

        # Filter out image_ids without any annotations
        self.image_ids = [
            image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0
        ]
        self.processor = processor

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        masks = []
        points = []
        labels = []

        for ann in anns:
            if ann['iscrowd']:
                continue 
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2]
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        if not bboxes or not masks:
            # Skip images without valid annotations
            return self.__getitem__((idx + 1) % len(self))
        
        if self.no_prompt:
            # Skip prompt generation
            inputs = self.processor(
                images=image,
                segmentation_maps=masks[0],  # Use the first mask
                input_points=None,
                input_labels=None,
                input_boxes=None,
                return_tensors="pt",
            )
            return inputs

        # Select one mask and bbox for simplicity (you can modify to handle multiple)
        mask = masks[0]
        bbox = bboxes[0]

        # Generate a point inside the mask
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0:
            # Handle empty masks (skip this sample)
            return self.__getitem__((idx + 1) % len(self))
        idx_point = np.random.choice(len(y_indices))
        y_point, x_point = y_indices[idx_point], x_indices[idx_point]
        points.append([float(x_point), float(y_point)])
        labels.append(1) 

        use_bbox = len(bboxes) > 0

        if use_bbox:
            # Use the first bounding box
            prompt_boxes = [bboxes[0]]
            prompt_points = []  # No points
            prompt_labels = []  # No labels
        else:
            # Use the first point
            prompt_boxes = []  # No boxes
            prompt_points = [points[0]]
            prompt_labels = [labels[0]]

        # Prepare inputs using the processor
        inputs = self.processor(
            images=image,
            segmentation_maps=masks[0] if masks else None,  # Use the first mask
            input_points=[[prompt_points]] if prompt_points else None,
            input_labels=[[prompt_labels]] if prompt_labels else None,
            input_boxes=[[prompt_boxes]] if prompt_boxes else None,
            return_tensors="pt",
        )

        return inputs
    
