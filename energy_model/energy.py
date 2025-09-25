__author__ = 'HPC'
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import weibull_min
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties
import random
from sklearn.metrics import roc_auc_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim * 2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(input_dim * 2, input_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(input_dim, int(input_dim / 2))
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(int(input_dim / 2), num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x
    
def energy_ft_loss(outputs, outputs_in, outputs_out, labels):
    T = 10
    Ec_out = -T * torch.logsumexp(outputs_out / T, dim=1)
    Ec_in = -T * torch.logsumexp(outputs_in / T, dim=1)
    cross_entropy = nn.CrossEntropyLoss()
    m_k = -10
    m_u = -5
    loss = cross_entropy(outputs, labels)
    relu = nn.ReLU()
    loss += 0.1 * (torch.pow(relu(Ec_in - m_k), 2).mean() + torch.pow(relu(m_u - Ec_out), 2).mean())
    return loss
        
def calculate_energy(logits, T = 10):
    energy = -T * torch.logsumexp(logits / T, dim=1)
    return energy
