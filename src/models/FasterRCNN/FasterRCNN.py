''' 
    FasterRCNN code
    Pytorch based implementation of faster rcnn framework. This code is based on the paper;
    "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks by Shaoqing Ren"
    
    The main characteristics
    - Vgg16 is used as an image encoder for faster rcnn.
    - 
    
    Reference
    - Paper : https://arxiv.org/pdf/1506.01497.pdf
    - Code : https://github.com/AlphaJia/pytorch-faster-rcnn
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import transform as sktsf
from torchvision.ops import nms
from typing import Optional, List, Literal, Tuple
from src.models.FasterRCNN.RPN import RPN
from src.models.FasterRCNN.backbone import VGG
from itertools import product as product
from src.utils import gcxgcy_to_cxcy, cxcy_to_xy, find_jaccard_overlap
import warnings

class FasterRCNN(nn.Module):
    def __init__(self, extractor : nn.Module, rpn : RPN, head : nn.Module, loc_normalize_mean = (0,0,0,0), loc_normalize_std = (0.1, 0.1, 0.2, 0.2)):
        super.__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate')
        
        # priors_cxcy : needed for detection
        self.priors_cxcy = self.create_prior_boxes()
                
    @property
    def n_class(self):
        return self.head.n_class
        
    def use_preset(self, preset : Literal['evaluate','visualize']):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
            
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        
    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l,self.nms_thresh)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
            
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score
        
    def forward(self, x : torch.Tensor, scale : float = 1.0, is_return_all : bool = False):
        img_size = x.shape[2:]
        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale)
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        
        if is_return_all:
            return roi_cls_locs, roi_scores, rois, roi_indices
        else:
            return roi_cls_locs, roi_scores
    
    def preprocess(img, min_size : int = 600, max_size : int = 1000):
        C, H, W = img.shape
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        img = img / 255.
        img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
        # both the longer and shorter should be less than
        # max_size and min_size
        if opt.caffe_pretrain:
            normalize = caffe_normalize
        else:
            normalize = pytorch_normalze
        return normalize(img)
    
    def predict(self, imgs : np.ndarray, visualize : bool =False):
        self.eval()
        sizes = list()
        for img in imgs:
            size = img.shape[1:]
            sizes.append(size)
        
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            for img in imgs:
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
        else:
             prepared_imgs = imgs 
             
        bboxes = list()
        labels = list()
        scores = list()
        
        for img, size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale, is_return_all = True)
            
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = at.totensor(rois) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = (F.softmax(at.totensor(roi_score), dim=1))

            bbox, label, score = self._suppress(cls_bbox, prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()
        
        return bboxes, labels, scores