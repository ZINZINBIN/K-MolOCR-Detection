import torch
from torch.utils.data import DataLoader
from typing import Literal, Optional, List, Dict
from src.utils import calculate_mAP

def evaluate(
    dataloader : DataLoader, 
    model:torch.nn.Module,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    min_score:float = 0.5,
    max_overlap:float = 0.5,
    top_k:int = 8,
    ):
    
    model.eval()
    model.to(device)
    
    test_loss = 0
    total_size = 0
    
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        
        with torch.no_grad():
                            
            predicted_locs, predicted_scores = model(data.to(device))
            loss = loss_fn(predicted_locs, predicted_scores, target['boxes'], target['classes'])
            
            total_size += predicted_locs.size()[0]
            test_loss += loss.item()
            det_boxes_batch, det_labels_batch, det_scores_batch = model.predict(predicted_locs, predicted_scores, min_score, max_overlap, top_k)
            
            boxes = [b.to(device) for b in target['boxes']]
            labels = [l.to(device) for l in target['classes']]
            
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            
            true_boxes.extend(boxes)
            true_labels.extend(labels)
    
    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, max_overlap, device)
    
    if total_size >0:
    	test_loss /= total_size
    else:
    	test_loss = 0
     
    return test_loss, APs, mAP