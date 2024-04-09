from src.utils import PDF2Image
from src.detect import detect
import pandas as pd
import numpy as np
import torch, os
from src.models.SSD300.model import SSD300
from src.detect import detect
from src.utils import crop_mol_img
from config.API_key import API_KEY
from tqdm.auto import tqdm
import argparse, json, boto3, io, cv2

def transfer_bucket(file_path :str, s3_file_name:str):
    aws_access_key = API_KEY.aws_access_key
    aws_secret_key = API_KEY.aws_secret_key
    aws_default_region = API_KEY.aws_default_region

    bucket_name = API_KEY.bucket_name

    s3 = boto3.client(
        "s3", 
        aws_access_key_id=aws_access_key, 
        aws_secret_access_key=aws_secret_key
    )

    with open(file_path, "rb") as f:
        content = f.read()
        
    content_byte = io.BytesIO(content)

    # print(s3_file_name)
    s3.upload_fileobj(content_byte, bucket_name, s3_file_name, ExtraArgs={'ACL': 'public-read'})

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
    save_best_dir = "./weights/{}_ddp_best.pt".format(tag)
    
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
        
    if not os.path.exists("./results/sample_test/molecules/"):
        os.mkdir("./results/sample_test/molecules/")
    
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
        
        for idx, img in enumerate(tqdm(imgs, desc = "Detection process for file path: {}".format(path))): 
            origin_img = img
            annot, is_success, locs, labels = detect(img, model, device, min_score = args['min_score'], max_overlap = args['max_overlap'], top_k = args['top_k'], return_results=True, soft_nms = False)
            
            if not is_success:
                continue
            
            img_path = "./results/sample_test/file{:03d}/page{:03d}.jpg".format(file_idx + 1, idx + 1)
    
            annot.save(img_path)
            
            locs = np.array(locs)
            labels = np.array(labels)
            target_indx = np.where((labels == "molecule"))
            
            locs = locs[target_indx].tolist()
            labels = labels[target_indx].tolist()
            
            image_ids.extend([idx + 1 + num_pages for _ in range(len(locs))])
            positions.extend(locs)
            classes.extend(labels)
            
            # Crop image and transfer the image to AWS server
            for idx_mol, loc in enumerate(locs):
                xl,yl,w,h = loc
                
                img_mol = np.array(origin_img)[int(yl):int(yl+h), int(xl):int(xl+w)]
                
                tag = "file_{:03d}_page_{:03d}_mol_{:03d}".format(file_idx+1,idx+1,idx_mol+1)
                local_save_path = "./results/sample_test/molecules/{}.png".format(tag)
                
                # save to local directory
                cv2.imwrite(local_save_path, img_mol)
                
                # save to AWS server
                s3_save_path = "https://kmolocr.s3.ap-northeast-2.amazonaws.com/{}".format(tag)
                
                transfer_bucket(local_save_path, s3_save_path)
        
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