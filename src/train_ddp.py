import torch, os, random, warnings
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Literal, Optional, List, Dict
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# distributed data parallel package
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from src.evaluate import evaluate

# anomaly detection from training process : backward process
torch.autograd.set_detect_anomaly(True)

warnings.filterwarnings(action = "ignore")

def set_random_seeds(random_seed:int = 42):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_distributed_loader(train_dataset:Dataset, valid_dataset : Dataset, num_replicas : int, rank : int, num_workers : int, batch_size : int = 32):
    train_sampler = DistributedSampler(train_dataset, num_replicas=num_replicas, rank = rank, shuffle = True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=num_replicas, rank = rank, shuffle = True)

    train_loader = DataLoader(train_dataset, batch_size, sampler = train_sampler, num_workers = num_workers, pin_memory=True, drop_last = True, persistent_workers=True, collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size, sampler = valid_sampler, num_workers = num_workers, pin_memory=True, drop_last = True, persistent_workers=True, collate_fn=valid_dataset.collate_fn)

    return train_loader, valid_loader, train_sampler, valid_sampler

def train_per_epoch(
    ddp_model,
    train_loader,
    valid_loader,
    optimizer,
    scheduler,
    loss_fn : torch.nn.Module,
    max_norm_grad : Optional[float] = None,
    device:Optional[str] = "cpu"
    ):
    
    # training process
    train_loss = 0
    total_size = 0 
    ddp_model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        optimizer.zero_grad()
            
        output_locs, output_score = ddp_model(data.to(device))
        loss = loss_fn(output_locs, output_score, target['boxes'], target['classes'])
            
        if not torch.isfinite(loss):
            # logger added
            continue
        else:
            loss.backward()
        
        if max_norm_grad:
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm_grad)
        
        optimizer.step()
        
        # metric computation
        train_loss += loss.item()
        total_size += output_locs.size()[0]
        
    if scheduler:
        scheduler.step()

    if total_size >0:
        train_loss /= total_size
    else:
        train_loss = 0
        
    # validation process
    ddp_model.eval()
    valid_loss = 0
    total_size = 0

    for batch_idx, (data, target) in enumerate(valid_loader):
        with torch.no_grad():
            
            optimizer.zero_grad()
        
            output_locs, output_score = ddp_model(data.to(device))
            loss = loss_fn(output_locs, output_score, target['boxes'], target['classes'])
           
            if not torch.isfinite(loss):
                continue
            
            # metric computation
            valid_loss += loss.item()
            
            total_size += output_locs.size()[0]

    if total_size >0:
        valid_loss /= total_size
    else:
        valid_loss = 0
    
    return train_loss, valid_loss

def train_per_proc(
    rank : int, 
    world_size : int, 
    batch_size : Optional[int],
    model : torch.nn.Module,
    train_dataset : Dataset,
    valid_dataset : Dataset,
    test_dataset:Dataset,
    loss_fn : torch.nn.Module,
    max_norm_grad : Optional[float] = None,
    model_filepath : str = "./weights/distributed.pt",
    random_seed : int = 42,
    resume : bool = True,
    learning_rate : float = 1e-3,
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best : str = "./weights/best.pt",
    tensorboard_dir : Optional[str] = None,
    min_score : float = 0.5,
    max_overlap : float = 0.5,
    top_k :int = 8,
    optimizer_type:Literal['SGD', 'RMSProps', 'Adam', 'AdamW'] = 'SGD'
    ):
    
    dist.init_process_group("nccl", rank = rank, world_size = world_size)

    train_loss_list = []
    valid_loss_list = []

    best_epoch = 0
    best_loss = np.inf
    
    if not os.path.isdir(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    
    # tensorboard setting
    if dist.get_rank() == 0 and tensorboard_dir:
        writer = SummaryWriter(tensorboard_dir)
    else:
        writer = None
        
    device = torch.device("cuda:{}".format(rank))
    set_random_seeds(random_seed)

    torch.cuda.set_device(device)
    model.to(device)
    model.train()
    
    ddp_model = DDP(model, device_ids = [device], output_device=device)
    
    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr = learning_rate)
    elif optimizer_type == "RMSProps":
        optimizer = torch.optim.RMSprop(ddp_model.parameters(), lr = learning_rate)
    elif optimizer_type == "Adam":
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr = learning_rate)
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr = learning_rate)
    else:
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr = learning_rate)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 8, gamma=0.995)
    
    if not os.path.isfile(model_filepath) and dist.get_rank() == 0:
        torch.save(model.state_dict(), model_filepath)
        
    dist.barrier()
    
    # continue learning
    if resume == True:
        map_location = {"cuda:0":"cuda:{}".format(rank)}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location), strict = False)
        
    train_loader, valid_loader, train_sampler, valid_sampler = get_distributed_loader(train_dataset, valid_dataset, num_replicas=world_size, rank = rank, num_workers = 8, batch_size = batch_size)

    if rank == 0:
        test_loader = DataLoader(test_dataset, batch_size, num_workers = 8, pin_memory=True, drop_last = True, persistent_workers=True, collate_fn=test_dataset.collate_fn)
    else:
        test_loader = None
    
    for epoch in tqdm(range(num_epoch), desc = "Distributed training process", disable = False if rank == 0 else True):
        
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)
        
        train_loss, valid_loss = train_per_epoch(
            ddp_model,
            train_loader,
            valid_loader,
            optimizer,
            scheduler,
            loss_fn,
            max_norm_grad,
            device
        )
        
        dist.barrier()
        
        if dist.get_rank() == 0:
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            
            # tensorboard recording
            if writer is not None:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/valid', valid_loss, epoch)
                
            if verbose:
                if epoch % verbose == 0:
                    test_loss, APs, mAP = evaluate(test_loader, model, loss_fn, device, min_score, max_overlap, top_k)
                    
                    print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f}, test loss : {:.3f}, Precision : {:.3f}, Total precision : {:.3f}".format(
                        epoch+1, train_loss, valid_loss, test_loss, APs['molecule'], mAP
                    ))
          
            # save the best parameters
            if best_loss > valid_loss:
                best_loss = valid_loss
                best_epoch  = epoch
                torch.save(model.state_dict(), save_best)

            # save the last parameters
            torch.save(model.state_dict(), model_filepath)
            
        dist.barrier()
    
    if dist.get_rank() == 0:
        print("# training process finished, best loss : {:.3f}, best epoch : {}".format(best_loss, best_epoch))
    
    if writer is not None:
        writer.close()
        
    # clean up
    dist.destroy_process_group()

    return train_loss_list, valid_loss_list

def train(
    batch_size : Optional[int],
    model : torch.nn.Module,
    train_dataset : Dataset,
    valid_dataset : Dataset,
    test_dataset:Dataset,
    random_seed : int = 42,
    resume : bool = True,
    learning_rate : float = 1e-3,
    loss_fn = None,
    max_norm_grad : Optional[float] = None,
    model_filepath : str = "./weights/distributed.pt",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best : str = "./weights/distributed_best.pt",
    tensorboard_dir : Optional[str] = None,
    min_score : float = 0.5,
    max_overlap : float = 0.5,
    top_k :int = 8,
    optimizer_type:Literal['SGD', 'RMSProps', 'Adam', 'AdamW'] = 'SGD'
):
    
    world_size = torch.cuda.device_count()

    mp.spawn(
        train_per_proc,
        args = (
            world_size, batch_size, model, train_dataset, valid_dataset, test_dataset, loss_fn, max_norm_grad, 
            model_filepath, random_seed, resume, learning_rate, num_epoch, verbose, save_best, tensorboard_dir, 
            min_score, max_overlap, top_k, optimizer_type
            ),
        nprocs = world_size,
        join = True
    )
    
    print("# Distributed training process is complete")
    
def example(rank, world_size):
    dist.init_process_group("gloo", rank = rank, world_size=world_size)
    model = torch.nn.Linear(10,10).to(rank)
    ddp_model = DDP(model, device_ids = [rank])

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr = 1e-3)

    outputs = ddp_model(torch.randn(20,10).to(rank))
    labels = torch.randn(20,10).to(rank)
    
    optimizer.zero_grad()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    print("rank : {} process".format(rank))

def main():
    world_size = 4
    mp.spawn(
        example,
        args = (world_size,),
        nprocs = world_size,
        join = True
    )

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()