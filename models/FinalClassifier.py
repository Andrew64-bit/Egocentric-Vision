import torch
from torch import nn
import torch.nn.functional as F
from utils.logger import logger
import torch.optim as optim
import torch.utils.data as data
import numpy as np



class MLP(nn.Module):
    def __init__(self, clip_feature_dim, num_class):
        super(MLP, self).__init__()
        self.num_class = num_class
        self.clip_feature_dim = clip_feature_dim
        self.classifier = self.fc_fusion()
        
    def fc_fusion(self):
        num_bottleneck = 512
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.clip_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck,self.num_class),
                )
        return classifier
    def forward(self, input):
        output = self.classifier(input)
        return output
'''    
    def __init__(self, num_classes):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

    def forward(self, x):
        return self.classifier(x), {}'''


class MLPWithDropout(nn.Module):
    def __init__(self, clip_feature_dim, num_class):
        super(MLPWithDropout, self).__init__()
        self.num_class = num_class
        self.clip_feature_dim = clip_feature_dim
        self.classifier = self.fc_fusion()
        
    def fc_fusion(self):
        num_bottleneck = 512
        num_bottleneck1 = 256
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.clip_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(num_bottleneck, num_bottleneck1),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(num_bottleneck1, self.num_class),
                )
        return classifier
    
    def forward(self, input):
        output = self.classifier(input)
        return output

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_class, hidden_dim = 128, num_layers = 2):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_class)
    
    def forward(self, x):
        #x = x.unsqueeze(1)  # Add a dimension for the sequence length
        #logger.info(f"Input Shape: {x.shape}")
        #logger.info(f"x.size(1) = {x.size(1)}")
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Prendere l'output del ultimo timestep
        out = self.fc(out)
        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, num_classes, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x.shape: (batch_size, num_clips, d_model)
        x = self.positional_encoding(x)
        
        # Permuta per adattare alla forma attesa dal Transformer Encoder
        x = x.permute(1, 0, 2)  # (num_clips, batch_size, d_model)
        
        # Passa attraverso il Transformer Encoder
        transformer_out = self.transformer_encoder(x)
        
        # Media le uscite del Transformer per ogni clip
        transformer_out = transformer_out.mean(dim=0)  # (batch_size, d_model)
        
        # Passa attraverso il livello finale di classificazione
        output = self.fc(self.dropout(transformer_out))  # (batch_size, num_classes)
        
        return output
