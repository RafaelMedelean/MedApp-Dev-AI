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

writer_name = 'Densenet169_bigger_smaller_csv_just_validating'
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

path1 = '/sdb/ImageRetrievalVest/saving_models/Densenet169_bigger_smaller_first_training_Loss(0.8,2)_optim(0.0001,0.001)_StepLR(50,0.8)/best_model_f1=0.884437596302003_epoch=44.pth'
state_dict1 = torch.load(path1, map_location=device)
model.load_state_dict(state_dict1['model_state_dict'])

epochs_without_improvement = 0
val_loss = 0.0
best_f1 = -float('inf')
best_model_state = None

criterion = SigmoidFocalLoss(alpha = 0.8 , gamma = 2)

# Validate on the current fold's validation set
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for x, y in tqdm(val_loader, desc=f"Validation"):
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
print(f'Validation_Loss {val_loss}')
print(f'Optimal_F1 {optimal_f1_score}')

writer.close()