from src.utils import PDF2Image
from src.detect import detect
import pandas as pd
import numpy as np
import torch, os
from src.models.SSD300.model import SSD300
from src.detect import detect, detect_chem_from_PDF
import argparse
import json

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="Detect the Moleculer structure image in PDF file")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "SSD")
    parser.add_argument("--pdf_filepath", type = str, default = "./dataset/sample_test_independent/AU2018379499A1.pdf")
    parser.add_argument("--save_filepath", type = str, default = "./results/AU2018379499A1")
    
    # detection setup
    parser.add_argument("--min_score", type = float, default = 0.5)
    parser.add_argument("--max_overlap", type = float, default = 0.25)
    parser.add_argument("--top_k", type = int, default = 12)

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 0)

    args = vars(parser.parse_args())
    return args

# torch device state
print("=============== device setup ===============")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

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
    save_best_dir = "./weights/{}_ddp_best.pt".format(tag)
    
    model = SSD300(5)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(save_best_dir, map_location = device))
    
    detect_chem_from_PDF(model = model, device = device, PDF_filepath = args['pdf_filepath'], save_filepath = args['save_filepath'], min_score = args['min_score'], max_overlap = args['max_overlap'], top_k = args['top_k'])