from src.utils import PDF2Image
from src.detect import detect
import pandas as pd
import numpy as np
import torch, os
from src.models.SSD300.model import SSD300
from src.detect import detect
import argparse
import json

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="Detect the Moleculer structure image in PDF file")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "SSD")
    parser.add_argument("--save_dir", type = str, default = "./results")
    
    # detection setup
    parser.add_argument("--min_score", type = float, default = 0.2)
    parser.add_argument("--max_overlap", type = float, default = 0.5)
    parser.add_argument("--top_k", type = int, default = 5)

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
    save_best_dir = "./weights/{}_best.pt".format(tag)
    
    model = SSD300(5)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(save_best_dir, map_location = device))
    
    paths = [
        "./dataset/sample_test/file001.pdf",
        "./dataset/sample_test/file002.pdf",
        "./dataset/sample_test/file003.pdf",
    ]
    
    if not os.path.exists("./results/sample_test/"):
        os.mkdir("./results/sample_test/")

    page_ids = 0
    sample_ids = 0

    image_ids = []
    classes = []
    positions = []
    
    num_pages = 0
    
    print("# Molecular detection proceeding..")
    
    for file_idx, path in enumerate(paths):

        imgs = PDF2Image(path, False, None)
        save_path = "./results/sample_test/file{:03d}".format(file_idx + 1)
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        for idx, img in enumerate(imgs): 
            annot, is_success, locs, labels = detect(img, model, device, min_score = args['min_score'], max_overlap = args['max_overlap'], top_k = args['top_k'], return_results=True)
            
            if not is_success:
                continue
            
            img_path = "./results/sample_test/file{:03d}/page{:03d}.jpg".format(file_idx + 1, idx + 1)
            annot.save(img_path)
            
            locs = np.array(locs)
            labels = np.array(labels)
            target_indx = np.where((labels == "molecule") | (labels == "table"))
            
            locs = locs[target_indx].tolist()
            labels = labels[target_indx].tolist()
            
            image_ids.extend([idx + 1 + num_pages for _ in range(len(locs))])
            positions.extend(locs)
            classes.extend(labels)
        
        num_pages += len(imgs)

    print("# Detection process complete")

    ids = [i for i in range(len(image_ids))]
    
    dict4json = {
        "id":ids,
        "image_id":image_ids,
        "class":classes,
        "position":positions
    }
    
    with open("./results/sample_test.json", 'w', encoding='utf-8') as file:
        json.dump(dict4json, file, indent="\t")
    
    print("# JSON file conversion complete")