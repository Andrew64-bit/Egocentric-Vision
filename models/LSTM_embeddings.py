import torch
from torch import nn
import torch.nn.functional as F
from utils.logger import logger
import torch.optim as optim
import torch.utils.data as data
import numpy as np


class LSTM_Emb_Classifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_class, hidden_dim = 128, num_layers = 1, dropout = 0.5):
        super(LSTM_Emb_Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_class)
    
    def forward(self, x):
        #logger.info(f"Input Shape: {x.shape}")
        #logger.info(f"x.size(1) = {x.size(1)}")
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Prendere l'output del ultimo timestep
        out = self.dropout(out)
        embeddings = self.fc1(out)
        out = self.fc2(embeddings)
        return out, embeddings
    
