from torch import nn


class Classifier(nn.Module):
    def __init__(self, clip_feature_dim, num_class):
        super(Classifier, self).__init__()
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
