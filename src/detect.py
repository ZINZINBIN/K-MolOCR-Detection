from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as FT
from typing import Tuple, List
from src.utils import rev_label_map, label_color_map
from src.models.FasterRCNN.FasterRCNN import FasterRCNN
from src.models.SSD300.model import SSD300
import torch.nn as nn
import torch

def transform(image, resize : Tuple = (300, 300), mean : List = [0.485, 0.456, 0.406], std : List = [0.229, 0.224, 0.225]):
    new_image = FT.resize(image, resize)
    new_image = FT.to_tensor(new_image)
    new_image = FT.normalize(new_image, mean, std) 
    return new_image

def detect(original_image, model:nn.Module, device:str, min_score, max_overlap, top_k, suppress=None):
    image = transform(original_image)
    model.eval()
    
    is_success = False
    
    if type(model) == FasterRCNN:
        det_boxes, det_labels, det_scores = model.predict(image.unsqueeze(0).to(device))
    
    elif type(model) == SSD300:    
        predicted_locs, predicted_scores = model(image.unsqueeze(0).to(device))
        det_boxes, det_labels, det_scores = model.predict(predicted_locs, predicted_scores, min_score, max_overlap, top_k)
    
    det_boxes = det_boxes[0].cpu()
    original_dims = torch.FloatTensor([original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    
    det_boxes = det_boxes * original_dims
    det_labels = [rev_label_map[l] for l in det_labels[0].cpu().tolist()]
    
    # no object detected
    if det_labels == ['background']:
        return original_image
    
    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]]) 
                
        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw
    
    is_success = True
    
    return annotated_image, is_success