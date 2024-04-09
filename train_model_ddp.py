import pandas as pd
import torch, os
from sklearn.model_selection import train_test_split
from src.models.SSD300.model import SSD300
from src.models.FasterRCNN.FasterRCNN import FasterRCNNVGG16
from src.dataset import DetectionDataset
from src.loss import MultiBoxLoss
from src.train_ddp import train
import argparse

# argument parser
def parsing():
    parser = argparse.ArgumentParser(description="Training the Moleculer Object Detection model through distributed data parallel")
    
    # random seed
    parser.add_argument("--random_seed", type = int, default = 42)
    
    # tag and result directory
    parser.add_argument("--tag", type = str, default = "ddp")
    parser.add_argument("--model", type = str, default = "SSD", choices = ["SSD", "FasterRCNN", "RCNN"])
    parser.add_argument("--save_dir", type = str, default = "./results")

    # batch size / sequence length / epochs / distance / num workers / pin memory use
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--num_epoch", type = int, default = 256)
    parser.add_argument("--verbose", type = int, default = 4)
    parser.add_argument("--num_workers", type = int, default = 4)
    parser.add_argument("--pin_memory", type = bool, default = True)
    parser.add_argument("--train_test_ratio", type = float, default = 0.2)
    parser.add_argument("--continue_training", type = bool, default = False)
    
    # optimizer : SGD, RMSProps, Adam, AdamW
    parser.add_argument("--optimizer", type = str, default = "AdamW", choices=["SGD", "RMSProps", "Adam", "AdamW"])
    
    # Loss function setup
    parser.add_argument("--threshold", type = float, default = 0.5)
    parser.add_argument("--neg_pos_ratio", type = float, default = 3.0)
    parser.add_argument("--alpha", type = float, default = 1.0)
    parser.add_argument("--use_focal_loss", type = bool, default = False)
    
    # detection setup
    parser.add_argument("--min_score", type = float, default = 0.5)
    parser.add_argument("--max_overlap", type = float, default = 0.5)
    parser.add_argument("--top_k", type = int, default = 12)
    
    # learning rate, step size and decay constant
    parser.add_argument("--lr", type = float, default = 2e-4)
    parser.add_argument("--max_norm_grad", type = float, default = 1.0)

    args = vars(parser.parse_args())
    return args

# torch cuda initialize and clear cache
torch.cuda.init()
torch.cuda.empty_cache()

if __name__ == "__main__":
    
    # initialize process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    # torch device state
    print("=============== device setup ===============")
    print("torch device avaliable : ", torch.cuda.is_available())
    print("torch current device : ", torch.cuda.current_device())
    print("torch device num : ", torch.cuda.device_count())
        
    # parsing
    args = parsing()
    
    tag = "{}".format(args['model'])
    
    if len(args['tag'])>0:
        tag = "{}_{}".format(tag, args['tag'])
        
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
    model.summary()
    
    if args['continue_training'] and os.path.exists(save_best_dir):
        print("Load previous best parameters for continuing training process")
        model.load_state_dict(torch.load(save_best_dir, map_location = "cpu"))

    print("=============== Training process ===============")
    train(
        batch_size = args['batch_size'],
        model = model,
        train_dataset = train_dataset,
        valid_dataset = valid_dataset,
        test_dataset = test_dataset,
        random_seed = args['random_seed'],
        resume = False,
        learning_rate  = args['lr'],
        loss_fn = loss_fn,
        max_norm_grad = args['max_norm_grad'],
        model_filepath = save_last_dir,
        num_epoch = args['num_epoch'],
        verbose = args['verbose'],
        save_best = save_best_dir,
        tensorboard_dir = exp_dir,
        min_score = args['min_score'],
        max_overlap = args['max_overlap'],
        top_k = args['top_k'],
        optimizer_type = args['optimizer']
    )