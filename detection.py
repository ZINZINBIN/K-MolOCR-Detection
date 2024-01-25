import torch
from src.models.SSD300.model import SSD300
from src.detect import detect
import argparse
from PIL import Image

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="Detect the Moleculer structure image in PDF file")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "SSD")
    parser.add_argument("--save_dir", type = str, default = "./results")

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 3)

    # batch size / sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--num_epoch", type = int, default = 16)
    parser.add_argument("--verbose", type = int, default = 4)
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--pin_memory", type = bool, default = True)
    parser.add_argument("--train_test_ratio", type = float, default = 0.2)

    # optimizer : SGD, RMSProps, Adam, AdamW
    parser.add_argument("--optimizer", type = str, default = "AdamW", choices=["SGD", "RMSProps", "Adam", "AdamW"])
    
    # learning rate, step size and decay constant
    parser.add_argument("--lr", type = float, default = 2e-4)
    parser.add_argument("--use_scheduler", type = bool, default = True)
    parser.add_argument("--step_size", type = int, default = 4)
    parser.add_argument("--gamma", type = float, default = 0.95)
    parser.add_argument("--max_norm_grad", type = float, default = 1.0)

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
    
    img_path = './dataset/detection/img_00001.jpg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    
    annot = detect(original_image, model, device, min_score = 0.2, max_overlap = 0.5, top_k = 5)
    annot.save("./results/detection_with_background.jpg")