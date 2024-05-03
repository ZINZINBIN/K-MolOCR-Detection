import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from src.models.SSD300.model import SSD300

from itertools import product as product
from src.utils import gcxgcy_to_cxcy, cxcy_to_xy, find_jaccard_overlap
from copy import deepcopy

class SoftNMSWrapper:
    def __init__(self, model : Union[nn.Module, SSD300], device : str, sigma : float):
        self.device =device
        self.model = model
        self.n_classes = model.n_classes
        self.sigma = sigma
        
    def __call__(self, image:torch.Tensor):
        predicted_locs, predicted_scores = self.model(image)
        return predicted_locs, predicted_scores
    
    def eval(self):
        self.model.eval()
        
    def predict(self, predicted_locs:torch.Tensor, predicted_scores:torch.Tensor, min_score:float, max_overlap:float, top_k:int):
        device = self.device
        batch_size = predicted_locs.size(0)
        n_priors = self.model.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()
        
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(predicted_locs[i], self.model.priors_cxcy.to(device)))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()
  
            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = deepcopy(predicted_scores[i][:, c].detach())  # (8732)
                class_decoded_locs = deepcopy(decoded_locs.detach())
                            
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                
                if n_above_min_score == 0:
                    continue
                
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = class_decoded_locs[score_above_min_score]  # (n_qualified, 4)
                
                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)
                
                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)
                
                # Non-Maximum Suppression (NMS)
                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue
                    
                    decay = torch.exp(-overlap[box] ** 2 / self.sigma)
                    class_scores *= decay

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, class_scores < min_score)
                    
                    # The max operation retains previously suppressed boxes, like an 'OR' operation
                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size
