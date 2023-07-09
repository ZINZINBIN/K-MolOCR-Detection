import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
from torchvision.ops import nms

def loc2bbox(src_bbox:np.ndarray, loc:np.ndarray):
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.enp(dh) * src_height[:, np.newaxis]
    w = np.enp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox

def generate_anchor_base(base_size : int = 16, ratios : List = [0.5,1,2], anchor_scales = [8,16,32]):
    py = base_size / 2.
    px = base_size / 2.
    
    anchor_base = np.zeros((len(ratios)*len(anchor_scales), 4), dtype = np.float32)
    
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1./ratios[i])
            
            idx = i * len(anchor_scales) + j
            anchor_base[idx, 0] = py - h / 2.0
            anchor_base[idx, 1] = px - w / 2.0
            anchor_base[idx, 2] = py + h / 2.0
            anchor_base[idx, 3] = px + w / 2.0

    return anchor_base

class ProposalCreator:
    def __init__(
        self, 
        parent_model:nn.Module, 
        nms_thresh : float = 0.7,
        n_train_pre_nms:int=12000,
        n_train_post_nms:int=2000,
        n_test_pre_nms:int=6000,
        n_test_post_nms:int=300,
        min_size:int=16
        ):
        
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size
    
    def __call__(self, loc, score, anchor, img_size:int, scale : float = 1.0, device : str = "cpu"):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms    

        roi = loc2bbox(anchor, loc)
        
         # Clip predicted boxes to image.
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])

        # Remove predicted boxes with either height or width < threshold.
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).

        # unNOTE: somthing is wrong here!
        # TODO: remove cuda.to_gpu
        keep = nms(torch.from_numpy(roi).to(device), torch.from_numpy(score).to(device), self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]
        return roi

# Region Proposal network 
class RPN(nn.Module):
    def __init__(
        self, in_channels : int = 512, mid_channels : int = 512, ratios : List = [0.5, 1, 2],
        anchor_scales : List = [8, 16, 32], feat_stride : int = 16, proposal_creator_params : Dict = {}):
        super().__init__()
        
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios = ratios)
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3,1,1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        
        self._normal_init(self.conv1, 0, 0.01)
        self._normal_init(self.score, 0, 0.01)
        self._normal_init(self.loc, 0, 0.01)
        
    def _normal_init(self, m:nn.Conv2d, mean, stddev, truncated = False):
        
        if truncated:
            m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
            m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()
            
    def _enumerate_shifted_anchor(self, anchor_base:np.ndarray, feat_stride:int, height:int, width:int):
        # enumerate all shifted anchors
        # add A anchors (1,A,4) to
        # cell K shifts (K,1,4) to get
        # shift anchors (K,A,4)
        # reshape to (K*A, 4) shited anchors
        # return (K*A, 4)
        
        shift_y = np.arange(0, height * feat_stride, feat_stride)
        shift_x = np.arange(0, width * feat_stride, feat_stride)
        
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shift = np.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), axis = 1)
        
        A = anchor_base.shape[0]
        K = shift.shape[0]
        
        anchor = anchor_base.reshape((1,A,4)) + shift.reshape((1,K,4)).transpose((1,0,2))
        anchor = anchor.reshape((K*A,4)).astype(np.float32)
        return anchor
            
    def forward(self, x : torch.Tensor, img_size : int, scale : float = 1.0):
        n, _, hh, ww = x.size()
        anchor = self._enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, hh, ww)

        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(x))

        rpn_locs = self.loc(h)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale,
                device = x.device
                )
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        
        return rpn_locs, rpn_scores, rois, roi_indices, anchor
    
    