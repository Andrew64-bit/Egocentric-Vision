import torch
from torch import nn
import torch.nn.functional as F
from utils.logger import logger
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import math



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
    def __init__(self, clip_feature_dim, num_class, num_bottleneck = 512, num_bottleneck1 = 256):
        super(MLPWithDropout, self).__init__()
        self.num_class = num_class
        self.clip_feature_dim = clip_feature_dim
        self.num_bottleneck =num_bottleneck
        self.num_bottleneck1 = num_bottleneck1
        self.classifier = self.fc_fusion()

        
    def fc_fusion(self):
        classifier = nn.Sequential(
                nn.Linear(self.clip_feature_dim, self.num_bottleneck),
                nn.BatchNorm1d(self.num_bottleneck),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.num_bottleneck, self.num_bottleneck1),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.num_bottleneck1, self.num_class),
                )
        return classifier
    
    def forward(self, input):
        # input.shape: (batch_size, num_clips, clip_feature_dim)
        # Apply average pooling over the clips
        pooled_input = torch.mean(input, dim=1)  # shape: (batch_size, clip_feature_dim)
        output = self.classifier(pooled_input)
        return output

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_class, hidden_dim = 128, num_layers = 1, dropout = 0.5):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_class)
    
    def forward(self, x):
        #logger.info(f"Input Shape: {x.shape}")
        #logger.info(f"x.size(1) = {x.size(1)}")
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Prendere l'output del ultimo timestep
        out = self.dropout(out)
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
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, num_classes,num_bottleneck = 512, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, num_bottleneck)
        self.fc2 = nn.Linear(num_bottleneck, num_classes)
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

        output = self.fc2(output)
        
        return output
    
class LSTMTransformerClassifier(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, num_classes,hidden_size = 256, dropout=0.1):
        super(LSTMTransformerClassifier, self).__init__()
        self.d_model = d_model
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=hidden_size, num_layers=1, batch_first=True)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x.shape: (batch_size, num_clips, d_model)
        x = self.positional_encoding(x)
        
        # Passa attraverso LSTM
        lstm_out, _ = self.lstm(x)
        
        # Permuta per adattare alla forma attesa dal Transformer Encoder
        lstm_out = lstm_out.permute(1, 0, 2)  # (num_clips, batch_size, d_model*2)
        
        # Passa attraverso il Transformer Encoder
        transformer_out = self.transformer_encoder(lstm_out)
        
        # Media le uscite del Transformer per ogni clip
        transformer_out = transformer_out.mean(dim=0)  # (batch_size, d_model*2)
        
        # Passa attraverso il livello finale di classificazione
        output = self.fc(self.dropout(transformer_out))  # (batch_size, num_classes)
        
        return output



class TRNClassifier(nn.Module):
    def __init__(self, num_bottleneck = 256, clip_feature_dim = 1024, num_clips = 5, num_class = 8, dropout = 0.5):
        super(TRNClassifier, self).__init__()
        self.subsample_num = 3
        self.clip_feature_dim = clip_feature_dim
        self.num_clips = num_clips
        self.scales = [i for i in range(num_clips, 1, -1)]  # Multi-scale frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_clips, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale)))

        self.num_class = num_class
        self.fc_fusion_scales = nn.ModuleList()
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                nn.ReLU(),
                nn.Linear(scale * self.clip_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_bottleneck, self.num_class),
            )

            self.fc_fusion_scales.append(fc_fusion)

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-clip relation' % i for i in self.scales])

    def forward(self, input):
        act_all = input[:, self.relations_scales[0][0], :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.clip_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.clip_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def return_relationset(self, num_clips, num_clips_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_clips)], num_clips_relation))