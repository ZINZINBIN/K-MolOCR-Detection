import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2, os
from PIL import Image

from src.models.SSD300.model import SSD300
from src.detect import transform
import argparse

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="GradCAM application for checking model capabilities")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "SSD_ddp")
    parser.add_argument("--input_file_dir", type = str, default = "./dataset/detection/folder_00/img_00001.jpg")
    parser.add_argument("--save_dir", type = str, default = "./results/gradcam")

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    
    # model architecture
    parser.add_argument("--module", type = str, default = "conv7")
    parser.add_argument("--layer", type = str, default = "30")
    
    # detection setup
    parser.add_argument("--min_score", type = float, default = 0.5)
    parser.add_argument("--max_overlap", type = float, default = 0.25)
    parser.add_argument("--top_k", type = int, default = 12)
    
    args = vars(parser.parse_args())
    
    return args

class GradCAM(nn.Module):
    def __init__(self, model : SSD300, module : str, layer : str):
        super().__init__()
        self.model = model
        self.module = module
        self.layer = layer
        
        self.forward_result = []
        self.backward_result = []
        
        self.register_hook()
    
    def register_hook(self):
        for module_name, module in self.model.base._modules.items():
            # if module_name == self.module:
            #     for layer_name, layer in module._modules.items():
            #         if layer_name == self.layer:    
            #             module.register_forward_hook(self.forward_hook)
            #             module.register_backward_hook(self.backward_hook)
            
            module.register_forward_hook(self.forward_hook)
            module.register_backward_hook(self.backward_hook)
                        
    def forward(self, input:torch.Tensor, target_index, min_score:float, max_overlap:float, top_k:int):
        locs, scores = self.model(input)
        det_boxes, det_labels, det_scores = model.predict(locs, scores, min_score, max_overlap, top_k)

        if target_index is None:
            target_index = torch.where(det_labels[0] == 3)
        
        det_scores[0][target_index].mean().backward(retain_graph=True)
        
        out = 0
        for forward_result, backward_result in zip(self.forward_result, reversed(self.backward_result)):
            a_k = torch.mean(backward_result, dim=(1, 2), keepdim=True)         # [512, 1, 1]
            dout = torch.sum(a_k * forward_result, dim=0).cpu()                  # [512, 7, 7] * [512, 1, 1]
            dout = torch.relu(dout) / torch.max(dout)
            dout = F.upsample_bilinear(dout.unsqueeze(0).unsqueeze(0), [300, 300])  # 4D로 바꿈
            out += dout
        
        return out.cpu().detach().squeeze().numpy()

    def forward_hook(self, _, input:torch.Tensor, output:torch.Tensor):
        self.forward_result.append(torch.squeeze(output))

    def backward_hook(self, _, grad_input:torch.Tensor, grad_output:torch.Tensor):
        self.backward_result.append(torch.squeeze(grad_output[0]))
        
def show_cam_on_image(img, mask, save_dir):

    # mask = (np.max(mask) - np.min(mask)) / (mask - np.min(mask))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    
    cv2.imwrite(os.path.join(save_dir, "gradcam.png"), np.uint8(cam * 255))
    cv2.imwrite(os.path.join(save_dir, "heatmap.png"), np.uint8(heatmap * 255))

if __name__ == "__main__":
    
    # parsing
    args = parsing()
    tag = args['tag']
    
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'
    
    device = 'cpu'
    save_best_dir = "./weights/{}_best.pt".format(tag)
    
    model = SSD300(5)
    model.to(device)
    model.load_state_dict(torch.load(save_best_dir, map_location = device))
    
    model.train()
    
    original_image = Image.open(args['input_file_dir'], mode='r')
    original_image = original_image.convert('RGB')
    
    image_tensor = transform(original_image).unsqueeze(0).to(device).requires_grad_(True)
    
    grad_cam = GradCAM(model=model, module=args['module'], layer=args['layer'])
    
    mask = grad_cam(image_tensor, None, args["min_score"], args["max_overlap"], args["top_k"])
    
    resize_img = image_tensor.detach().squeeze(0).permute(1,2,0).numpy()
    
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])
    
    show_cam_on_image(resize_img, mask, args['save_dir'])