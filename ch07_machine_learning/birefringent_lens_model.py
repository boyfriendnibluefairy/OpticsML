"""
    file: birefringent_lens_model.py

    The training pipeline is divided into three steps:
    Step 1. Data Preparation
    Step 2. Model Configuration
    Step 3. Model Training

    The BirefringentLensModel handles the model training part which includes
    1. Computing gradients
    2. Updating parameters
    3. Mini-batch handling
"""
import numpy
import datetime

import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
