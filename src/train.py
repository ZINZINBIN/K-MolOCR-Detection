import torch, os
from torch.utils.data import DataLoader
from typing import Literal, Optional, List, Dict
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# anomaly detection from training process : backward process
torch.autograd.set_detect_anomaly(True)

def train_per_epoch(
    dataloader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    max_norm_grad : Optional[float] = None,
    ):
    
    model.train()
    model.to(device)
    
    train_loss = 0
    total_size = 0
    
    # other metrics
    
    for batch_idx, (data, target) in enumerate(dataloader):
        
        optimizer.zero_grad()
        
        for param in model.parameters():
            param.grad = None
            
        output_locs, output_score = model(data.to(device))
        loss = loss_fn(output_locs, output_score, target['boxes'], target['classes'])
            
        if not torch.isfinite(loss):
            # logger added
            continue
        else:
            loss.backward()
        
        if max_norm_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_grad)
        
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
    
    return train_loss

def valid_per_epoch(
    dataloader : DataLoader, 
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    ):
    
    model.eval()
    model.to(device)
    
    valid_loss = 0
    total_size = 0
    
    # other metrics
    
    for batch_idx, (data, target) in enumerate(dataloader):
        with torch.no_grad():
            
            optimizer.zero_grad()
        
            for param in model.parameters():
                param.grad = None
            
            output_locs, output_score = model(data.to(device))
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
    
    return valid_loss

def train(
    train_loader : DataLoader, 
    valid_loader : DataLoader,
    model : torch.nn.Module,
    optimizer : torch.optim.Optimizer,
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler],
    loss_fn : torch.nn.Module,
    device : str = "cpu",
    num_epoch : int = 64,
    verbose : Optional[int] = 8,
    save_best_dir : str = "./weights/best.pt",
    save_last_dir : str = "./weights/last.pt",
    exp_dir : Optional[str] = None,
    max_norm_grad : Optional[float] = None,
    ):

    train_loss_list = []
    valid_loss_list = []
    
    best_epoch = 0
    best_loss = torch.inf
    
    if not os.path.isdir(exp_dir):
        os.mkdir(exp_dir)
    
    # tensorboard
    if exp_dir:
        writer = SummaryWriter(exp_dir)
    else:
        writer = None

    for epoch in tqdm(range(num_epoch), desc = "training process"):
    
        train_loss = train_per_epoch(
            train_loader, 
            model,
            optimizer,
            scheduler,
            loss_fn,
            device,
            max_norm_grad,
        )

        valid_loss = valid_per_epoch(
            valid_loader, 
            model,
            optimizer,
            loss_fn,
            device,
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        
        # tensorboard recording : loss and score
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)
        
        if verbose:
            if epoch % verbose == 0:
                print("epoch : {}, train loss : {:.3f}, valid loss : {:.3f}".format(epoch+1, train_loss, valid_loss))
                    
        # save the last parameters
        torch.save(model.state_dict(), save_last_dir)

        # save the best parameters
        if  best_loss > valid_loss:
            best_loss = valid_loss
            best_epoch  = epoch
            torch.save(model.state_dict(), save_best_dir)

    print("training process finished, best loss : {:.3f}, best epoch : {}".format(best_loss, best_epoch))
    
    if writer:
        writer.close()

    return  train_loss_list, valid_loss_list
