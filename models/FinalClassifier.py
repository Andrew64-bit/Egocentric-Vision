import torch
from torch import nn
import torch.nn.functional as F
from utils.logger import logger
import torch.optim as optim
import torch.utils.data as data



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
        x = x.unsqueeze(1)  # Add a dimension for the sequence length
        #logger.info(f"Input Shape: {x.shape}")
        #logger.info(f"x.size(1) = {x.size(1)}")
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # Prendere l'output del ultimo timestep
        out = self.fc(out)
        return out
    
