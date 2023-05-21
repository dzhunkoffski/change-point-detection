import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm.notebook import tqdm
import torch
import wandb

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
plt.rcParams.update({'font.size': 15})

def train_epoch(model, optimizer, loss_fn, data_loader, tqdm_desc, device, calc_metric):
    model.train()

    train_loss = 0.0
    train_roc_auc = 0.0
    train_ap = 0.0

    for timeseries, labels in tqdm(data_loader, desc=tqdm_desc):
        timeseries = timeseries.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(timeseries)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(timeseries)
        if calc_metric:
            train_roc_auc += roc_auc_score(labels.cpu().detach().numpy(), logits.cpu().detach().numpy()) * len(timeseries)
            train_ap += average_precision_score(labels.cpu().detach().numpy(), logits.cpu().detach().numpy()) * len(timeseries)
    
    train_loss /= len(data_loader.dataset)
    train_roc_auc /= len(data_loader.dataset)
    train_ap /= len(data_loader.dataset)

    return train_loss, train_roc_auc, train_ap

@torch.no_grad()
def validate_epoch(model, loss_fn, data_loader, tqdm_desc, device, calc_metric):
    model.eval()

    val_loss = 0.0
    val_roc_auc = 0.0
    val_ap = 0.0

    for timeseries, labels in tqdm(data_loader, desc=tqdm_desc):
        timeseries = timeseries.to(device)
        labels = labels.to(device)

        logits = model(timeseries)
        loss = loss_fn(logits, labels)

        val_loss += loss.item() * len(timeseries)
        if calc_metric:
            val_roc_auc += roc_auc_score(labels.cpu().detach().numpy(), logits.cpu().detach().numpy()) * len(timeseries)
            val_ap += average_precision_score(labels.cpu().detach().numpy(), logits.cpu().detach().numpy()) * len(timeseries)
    
    val_loss /= len(data_loader.dataset)
    val_roc_auc /= len(data_loader.dataset)
    val_ap /= len(data_loader.dataset)

    return val_loss, val_roc_auc, val_ap

def train(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, device, scheduler = None, use_wandb = False, calc_metric = False):
    train_losses = []
    train_roc_aucs = []
    train_aps = []

    val_losses = []
    val_roc_aucs = []
    val_aps = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_roc_auc, train_ap = train_epoch(model, optimizer, loss_fn, train_loader, f'Epoch {epoch} [Train]', device, calc_metric)
        val_loss, val_roc_auc, val_ap = validate_epoch(model, loss_fn, val_loader, f'Epoch {epoch} [Val]', device, calc_metric)

        if scheduler is not None:
            scheduler.step()
        
        train_losses.append(train_loss)
        train_roc_aucs.append(train_roc_auc)
        train_aps.append(train_ap)

        val_losses.append(val_loss)
        val_roc_aucs.append(val_roc_auc)
        val_aps.append(val_ap)

        # WANDB
        if use_wandb:
            if calc_metric:
                wandb.log({
                    'train_loss': train_loss,
                    'train_roc_auc': train_roc_auc,
                    'train_ap': train_ap,
                    'val_loss': val_loss,
                    'val_roc_auc': val_roc_auc,
                    'val_ap': val_ap
                })
            else:
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                })
        clear_output()
    
    return {
        'TRAIN': {
            'loss': train_losses,
            'roc_auc': train_roc_aucs,
            'ap': train_aps
        },
        'VAL': {
            'loss': val_losses,
            'roc_auc': val_roc_aucs,
            'ap': val_aps
        }
    }
