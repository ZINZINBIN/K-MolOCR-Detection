import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional
from torch.autograd import Variable
from src.utils import find_jaccard_overlap, cxcy_to_gcxgcy, xy_to_cxcy

class FocalLoss(nn.Module):
    def __init__(self, weight : Optional[torch.Tensor] = None, gamma : float = 2.0):
        super(FocalLoss, self).__init__()
        assert gamma >= 0, "gamma should be positive"
        self.gamma = gamma
        self.weight = weight
    
    def update_weight(self, weight : Optional[torch.Tensor] = None):
        self.weight = weight

    def compute_focal_loss(self, inputs:torch.Tensor, gamma:float, alpha : torch.Tensor):
        p = torch.exp(-inputs)
        loss = alpha * (1-p) ** gamma * inputs
        return loss

    def forward(self, input : torch.Tensor, target : torch.Tensor):
        weight = self.weight.to(input.device)
        alpha = weight.gather(0, target.data.view(-1))
        alpha = Variable(alpha)
        return self.compute_focal_loss(F.cross_entropy(input, target, reduce = False, weight = None), self.gamma, alpha)

class MultiBoxLoss(nn.Module):
    def __init__(self, priors_cxcy, threshold : float = 0.5, neg_pos_ratio : float = 3.0, alpha : float = 1.0, use_focal_loss : bool = False):
        super().__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = self.cxcy_to_xy(priors_cxcy)
        
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        
        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce = False) if not use_focal_loss else FocalLoss(torch.Tensor([0.2,0.2,0.2,0.2,0.2]), 2.0)
        
    def cxcy_to_xy(self, cxcy:Union[torch.Tensor, np.ndarray]):
        return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2), cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1) 
    
    def forward(self, predicted_locs:torch.Tensor, predicted_scores:torch.Tensor, boxes:torch.Tensor, labels:torch.Tensor):
        
        batch_size = predicted_locs.size()[0]
        n_priors = self.priors_cxcy.size()[0]
        n_classes = predicted_scores.size()[2]
        
        device = predicted_locs.device
        
        assert n_priors == predicted_locs.size()[1] == predicted_scores.size()[1]
        
        true_locs = torch.zeros((batch_size, n_priors, 4), dtype = torch.float32).to(device)
        true_classes = torch.zeros((batch_size, n_priors), dtype = torch.long).to(device)
        
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i].to(device), self.priors_xy.to(device))  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i].to(device)[object_for_each_prior]  # (8732)
            
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i].to(device)[object_for_each_prior]), self.priors_cxcy.to(device))
            
        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs.to(device)[positive_priors])  # (), scalar

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        return conf_loss + self.alpha * loc_loss