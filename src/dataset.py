import pandas as pd
import numpy as np
import torch, cv2
from PIL import Image
import ast
from torch.utils.data import Dataset
from src.utils import transform
from typing import Literal

class DetectionDataset(Dataset):
    def __init__(self, df : pd.DataFrame, split : Literal['TRAIN', 'TEST'], transforms = transform):
        self.df = df
        self.paths = df['img'].values
        self.labels = df['label'].values
        self.classes = df['class'].values
        self.transforms = transforms
        self.split = split
        
    def __getitem__(self, idx:int):
        # data
        img = Image.open(self.paths[idx], mode = 'r')
        img = img.convert('RGB')
        
        # target
        boxes = torch.as_tensor(ast.literal_eval(self.labels[idx]), dtype = torch.float32)
        n_mols = torch.as_tensor(self.df['n_molecules'].values[idx], dtype = torch.int32)
        classes = torch.as_tensor(ast.literal_eval(self.classes[idx]), dtype = torch.int32)
        
        # preprocessing
        if self.transforms:
            img, boxes, classes = self.transforms(img, boxes, classes, self.split)
            
        target = {}
        target["boxes"] = boxes
        target["classes"] = classes
        target["n_molecules"] = n_mols
   
        return img, target
    
    def __len__(self):
        return len(self.paths)
    
    def collate_fn(self, samples):
        
        imgs = []
        boxes = []
        classes = []
        n_mols = []
        
        for data, target in samples:
            imgs.append(data)
            boxes.append(target['boxes'])
            classes.append(target['classes'])
            n_mols.append(target['n_molecules'])
            
        imgs = np.stack(imgs, axis = 0)
        imgs = torch.from_numpy(imgs)
        
        new_target = {}
        new_target["boxes"] = boxes
        new_target["classes"] = classes
        new_target["n_molecules"] = n_mols
        
        return imgs, new_target