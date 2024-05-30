import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.clip_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(num_bottleneck, num_bottleneck),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(num_bottleneck, self.num_class),
                )
        return classifier
    
    def forward(self, input):
        output = self.classifier(input)
        return output
    

class TransformerClassifier(nn.Module):
    def __init__(self, clip_feature_dim, num_class, num_heads=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.num_class = num_class
        self.clip_feature_dim = clip_feature_dim

        self.embedding = nn.Linear(clip_feature_dim, dim_feedforward)
        self.positional_encoding = PositionalEncoding(dim_feedforward, dropout)
        
        encoder_layers = TransformerEncoderLayer(dim_feedforward, num_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_feedforward, num_class)
        )
        
    def forward(self, input):
        # input shape should be (batch_size, num_clips, clip_feature_dim)
        input = self.embedding(input)  # embedding the input
        input = self.positional_encoding(input)  # adding positional encoding
        transformer_out = self.transformer_encoder(input)  # transformer encoder
        output = self.classifier(transformer_out.mean(dim=1))  # average pooling across the sequence dimension
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

