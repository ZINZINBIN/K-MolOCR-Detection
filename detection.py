import torch
from src.models.SSD300.model import SSD300
from src.detect import detect
from src.utils import calculate_mAP, transform
import argparse, ast
import pandas as pd
from PIL import Image

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="Detect the Moleculer structure image in PDF file")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "SSD")
    parser.add_argument("--save_dir", type = str, default = "./results")

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 3)
    
    # detection setup
    parser.add_argument("--min_score", type = float, default = 0.5)
    parser.add_argument("--max_overlap", type = float, default = 0.5)
    parser.add_argument("--top_k", type = int, default = 8)
    parser.add_argument("--image_index", type = int, default = 1000)

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
    
    # Select image from dataset
    df = pd.read_csv("./dataset/detection_data.csv")
    
    indx = args['image_index']
    img_path = df['img'].values[indx]

    true_boxes = torch.as_tensor(ast.literal_eval(df['label'].values[indx]), dtype = torch.float32)
    true_labels = torch.as_tensor(ast.literal_eval(df['class'].values[indx]), dtype = torch.int32)
    
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    
    # Detection process
    result = detect(original_image, model, device, min_score = args['min_score'], max_overlap = args['max_overlap'], top_k = args['top_k'])
    result[0].save("./results/detection_with_background.jpg")
    
    # Evaluation process    
    true_image, true_boxes, true_labels = transform(original_image, true_boxes, true_labels, 'TEST')
    
    predicted_locs, predicted_scores = model(true_image.unsqueeze(0).to(device))
    det_boxes, det_labels, det_scores = model.predict(predicted_locs, predicted_scores, args['min_score'], args['max_overlap'], args['top_k'])
    
    AP, mAP = calculate_mAP(det_boxes, det_labels, det_scores, [true_boxes], [true_labels], 0.5, device)
    
    for k,v in AP.items():
        print("{}:{}".format(k,v))
    
    print("mean average precision: {}".format(mAP))