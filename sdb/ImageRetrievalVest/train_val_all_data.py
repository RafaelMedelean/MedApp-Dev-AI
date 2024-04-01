from data_loaders_auto_split import train_loader,val_loader
from sklearn.metrics import f1_score, recall_score
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import sigmoid_focal_loss
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from tqdm import tqdm
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os
from torch.optim import AdamW 
import time
import random

SEED = 1337
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

#FocalLoss
class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
#Computing Optimal F1
def compute_optimal_thresholds_and_f1(labels, outputs):
 
    thresholds = np.arange(0, 1.05, 0.05)
    f1_scores = [f1_score(labels, outputs > thresh) for thresh in thresholds]
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    optimal_f1_score = f1_score(labels, outputs > optimal_threshold)
    
    return optimal_f1_score, optimal_threshold

##################################################################################################################
#Epochs, Patience, Names for tensor logging and saving models
total_epochs = 60
patience = 12 #EarlyStop Patience

writer_name = 'Densenet169_bigger_smaller_first_training_Loss(0.8,2)_optim(0.0001,0.001)_StepLR(50,0.8)'
save_output_directory = f"/sdb/ImageRetrievalVest/saving_models/{writer_name}"

if not os.path.exists(save_output_directory):
    os.makedirs(save_output_directory)
##################################################################################################################

writer = SummaryWriter(f'/sdb/ImageRetrievalVest/tensorboard/{writer_name}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################################################################################################
#Models
from models import generate_densenet

#DenseNet Encoder - NO PRETRAIN
model_depth = 169 # [121,169,201,264]
model = generate_densenet(model_depth, n_input_channels=1, num_classes=2)
model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
model.to(device)

epochs_without_improvement = 0
best_f1 = -float('inf')
best_model_state = None

optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
criterion = SigmoidFocalLoss(alpha = 0.8 , gamma = 2)
scheduler = StepLR(optimizer, step_size=50, gamma=0.8)

#scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=4, verbose=True)

# Train for each epoch
for epoch in range(total_epochs):
    epoch_start_time = time.time()
    print(f'epoch = {epoch}')
    train_loss = 0.0

    model.train()
    for batch_ndx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
        x, y = x.float().to(device), y.float().to(device)
        #print(f"Training batch image dimensions: {x.shape}")

        pred = model(x).squeeze()
        print(pred)
        #print(f'expected = {y} , pred = {pred}')
        loss = criterion(pred, y)
        #print(loss)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
    
    train_loss = train_loss / len(train_loader)
    val_loss = 0.0

    # Validate on the current fold's validation set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
            x, y = x.float().to(device), y.float().to(device)
            #print(f"Validation batch image dimensions: {x.shape}")

            pred = model(x).squeeze()
            #print(f'expected = {y} , pred = {pred}')
            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())
            loss = criterion(pred, y)
            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)

    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_preds_numpy = all_preds_tensor.cpu().numpy()

    all_labels_tensor = torch.cat(all_labels, dim=0)
    all_labels_numpy = all_labels_tensor.cpu().numpy()

    # Compute metrics
    optimal_f1_score, optimal_threshold = compute_optimal_thresholds_and_f1(all_labels_numpy, all_preds_numpy)

    # Log metrics to TensorBoard
    writer.add_scalar(f'Validation_Loss', val_loss, epoch)
    writer.add_scalar(f'Train_Loss', train_loss, epoch)
    writer.add_scalar(f'Optimal_F1', optimal_f1_score, epoch)
    
    if optimal_f1_score > best_f1:
        best_f1 = optimal_f1_score
        best_model_state = model.state_dict()
        save_checkpoint = {
            'epoch': epoch,
            'model_state_dict': best_model_state,
            'best_f1': best_f1
        }
        if(epoch > 30):
            save_path = f"{save_output_directory}/best_model_f1={best_f1}_epoch={epoch}.pth"
            torch.save(save_checkpoint, save_path)
            print(f"Model at epoch {epoch} with F1 score: {best_f1}")

    scheduler.step()
    epoch_end_time = time.time()
    writer.add_scalar(f'Epoch_Time', (epoch_end_time - epoch_start_time)//60, epoch)

writer.close()