import pandas as pd
import torch, os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.models.SSD300.model import SSD300
from src.models.FasterRCNN.FasterRCNN import FasterRCNNVGG16
from src.dataset import DetectionDataset
from src.loss import MultiBoxLoss
from src.train import train
from src.evaluate import evaluate
import argparse

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="Training the Moleculer Object Detection model")
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "")
    parser.add_argument("--model", type = str, default = "SSD", choices = ["SSD", "FasterRCNN", "RCNN"])
    parser.add_argument("--save_dir", type = str, default = "./results")

    # gpu allocation
    parser.add_argument("--gpu_num", type = int, default = 3)

    # batch size / sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--num_epoch", type = int, default = 64)
    parser.add_argument("--verbose", type = int, default = 4)
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--pin_memory", type = bool, default = True)
    parser.add_argument("--train_test_ratio", type = float, default = 0.2)

    # optimizer : SGD, RMSProps, Adam, AdamW
    parser.add_argument("--optimizer", type = str, default = "AdamW", choices=["SGD", "RMSProps", "Adam", "AdamW"])
    
    # Loss function setup
    parser.add_argument("--threshold", type = float, default = 0.5)
    parser.add_argument("--neg_pos_ratio", type = float, default = 3.0)
    parser.add_argument("--alpha", type = float, default = 1.0)
    parser.add_argument("--use_focal_loss", type = bool, default = False)
    
    # learning rate, step size and decay constant
    parser.add_argument("--lr", type = float, default = 2e-4)
    parser.add_argument("--use_scheduler", type = bool, default = True)
    parser.add_argument("--step_size", type = int, default = 4)
    parser.add_argument("--gamma", type = float, default = 0.95)
    parser.add_argument("--max_norm_grad", type = float, default = 1.0)
    
    # detection setup
    parser.add_argument("--min_score", type = float, default = 0.5)
    parser.add_argument("--max_overlap", type = float, default = 0.5)
    parser.add_argument("--top_k", type = int, default = 8)

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
    
    tag = "{}".format(args['model'])
    
    if len(args['tag'])>0:
        tag = "{}_{}".format(tag, args['tag'])
    
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:" + str(args["gpu_num"])
    else:
        device = 'cpu'
        
    save_best_dir = "./weights/{}_best.pt".format(tag)
    save_last_dir = "./weights/{}_last.pt".format(tag)
    exp_dir = os.path.join("./runs/", "tensorboard_{}".format(tag))
    
    # directory check
    if not os.path.exists("./runs"):
        os.mkdir("./runs")
        
    if not os.path.exists("./results"):
        os.mkdir("./results")
        
    if not os.path.exists("./weights"):
        os.mkdir("./weights")
    
    # load data
    df = pd.read_csv("./dataset/detection_data.csv")
    
    df_train, df_test = train_test_split(df, test_size = args['train_test_ratio'], shuffle = True, random_state = 42)
    df_train, df_valid = train_test_split(df_train, test_size = args['train_test_ratio'], shuffle = True, random_state = 42)
 
    train_dataset = DetectionDataset(df_train, split = 'TRAIN')
    valid_dataset = DetectionDataset(df_valid, split = 'TEST')
    test_dataset = DetectionDataset(df_test, split = 'TEST')
    
    print("=============== Dataset info ===============")
    print("train data : {}".format(train_dataset.__len__()))
    print("valid data : {}".format(valid_dataset.__len__()))
    print("test data : {}".format(test_dataset.__len__()))

    train_loader = DataLoader(train_dataset, batch_size = args['batch_size'], shuffle = True, num_workers=8, pin_memory=True, drop_last = True, persistent_workers=True, collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size = args['batch_size'], shuffle = True, num_workers=8, pin_memory=True, drop_last = True, persistent_workers=True, collate_fn=valid_dataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = args['batch_size'], shuffle = True, num_workers=8, pin_memory=True, drop_last = True, persistent_workers=True, collate_fn=test_dataset.collate_fn)

    if args['model'] == 'SSD':
        model = SSD300(5)
    elif args['model'] == 'FasterRCNN':
        model = FasterRCNNVGG16(n_fg_class=5)
        
    loss_fn = MultiBoxLoss(
        model.priors_cxcy, 
        threshold = args['threshold'], 
        neg_pos_ratio = args['neg_pos_ratio'], 
        alpha = args['alpha'],
        use_focal_loss = args['use_focal_loss']
    )
    
    model.to(device)
    
    # optimizer
    if args["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = args['lr'])
    elif args["optimizer"] == "RMSProps":
        optimizer = torch.optim.RMSprop(model.parameters(), lr = args['lr'])
    elif args["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args['lr'])
    elif args["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr = args['lr'])
        
    # scheduler
    if args["use_scheduler"]:    
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args['step_size'], gamma=args['gamma'])
   
    print("=============== Training process ===============")
    train_loss, valid_loss = train(
        train_loader,
        valid_loader,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        args['num_epoch'],
        args['verbose'],
        save_best_dir = save_best_dir,
        save_last_dir = save_last_dir,
        exp_dir = exp_dir,
        max_norm_grad = 1.0,
    )
    
    print("=============== Evaluation process ===============")
    model.eval()
    model.load_state_dict(torch.load(save_best_dir, map_location = device))
    
    test_loss, APs, mAP = evaluate(
        dataloader = test_loader, 
        model = model,
        loss_fn = loss_fn,
        device = device,
        min_score = args['min_score'],
        max_overlap = args['max_overlap'],
        top_k = args['top_k'],
    )
    
    print("test loss : {:.3f}, APs : {:.3f}, mAPs:{:.3f}".format(test_loss, APs['molecule'], mAP))