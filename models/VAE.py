import torch
import torch.nn as nn
from torch.autograd import Variable

class FC_VAE(nn.Module):
    """Fully connected variational Autoencoder"""
    def __init__(self, n_input, nz, n_hidden=1024):
        super(FC_VAE, self).__init__()
        self.nz = nz
        self.n_input = n_input
        self.n_hidden = n_hidden

        self.encoder = nn.Sequential(nn.Linear(n_input, n_hidden),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(n_hidden),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                )

        self.fc1 = nn.Linear(n_hidden, nz)
        self.fc2 = nn.Linear(n_hidden, nz)

        self.decoder = nn.Sequential(nn.Linear(nz, n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, n_input),
                                    )
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        return self.fc1(h), self.fc2(h)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        return self.decoder(z)

    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z
 
    def generate(self, z):
        res = self.decode(z)
        return res