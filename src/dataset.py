import os
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from transformers import SamProcessor
from PIL import Image

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, processor: SamProcessor, no_prompt=False, dtype=torch.float32):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.no_prompt = no_prompt
        self.processor = processor
        self.dtype = dtype

        # Filter out image_ids without any annotations
        self.image_ids = [
            image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0
        ]

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

        # Create segmentation_map by combining masks with label 1
        segmentation_map = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            segmentation_map[mask == 1] = 1  # Binary labels (1 for object)

        # Initialize fields
        input_points = None
        input_labels = None
        input_boxes = None

        if not self.no_prompt:
            use_bbox = len(bboxes) > 0

            if use_bbox:
                # Use bounding boxes as prompts
                labels_prompt = [1] * len(bboxes)  # Labels set to 1 for all boxes
                # Prepare inputs using the processor
                processed = self.processor(
                    images=image,
                    segmentation_maps=masks,
                    input_points=None,
                    input_labels=[labels_prompt],    
                    input_boxes=[bboxes],           
                    return_tensors="pt",
                )
            else:
                # No bounding boxes, generate point prompts
                points = []
                labels_prompt = []
                for mask in masks:
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) == 0:
                        continue  # Skip empty mask
                    idx_point = np.random.choice(len(y_indices))
                    y_point, x_point = y_indices[idx_point], x_indices[idx_point]
                    points.append([float(x_point), float(y_point)])
                    labels_prompt.append(1)  # Label '1' for positive points

                if not points:
                    # No valid points found, skip sample
                    return self.__getitem__((idx + 1) % len(self))

                # Prepare inputs using the processor
                processed = self.processor(
                    images=image,
                    segmentation_maps=masks,
                    input_points=[points],      # List of points
                    input_labels=[labels_prompt],  # List of labels
                    input_boxes=None,
                    return_tensors="pt",
                )
        else:
            # Skip prompt generation
            processed = self.processor(
                images=image,
                segmentation_maps=segmentation_map,
                input_points=None,
                input_labels=None,
                input_boxes=None,
                return_tensors="pt",
            )

        for key in processed:
            if isinstance(processed[key], torch.Tensor):
                processed[key] = processed[key].to(self.dtype)

        return processed