import torch
import torch.nn as nn
from torch.autograd import Variable

class FC_VAE(nn.Module):
    """Fully connected variational Autoencoder"""
    # dim_input (Input size), nz (Latent size), n_hidden (Hidden layer size
    def __init__(self, dim_input, nz, n_hidden=1024, device='mps',  dim_output=0):
        super(FC_VAE, self).__init__()
        self.device = device
        self.nz = nz
        self.dim_input = dim_input
        self.n_hidden = n_hidden
        self.dim_output = dim_output
        if dim_output == 0:
            self.dim_output = dim_input

        self.encoder = nn.Sequential(nn.Linear(dim_input, n_hidden),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(n_hidden),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                #nn.Linear(n_hidden, n_hidden),
                                #nn.BatchNorm1d(n_hidden),
                                #nn.ReLU(inplace=True),
                                #nn.Linear(n_hidden, n_hidden),
                                #nn.BatchNorm1d(n_hidden),
                                #nn.ReLU(inplace=True),
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
                                     #nn.Linear(n_hidden, n_hidden),
                                     #nn.BatchNorm1d(n_hidden),
                                     #nn.ReLU(inplace=True),
                                     #nn.Linear(n_hidden, n_hidden),
                                     #nn.BatchNorm1d(n_hidden),
                                     #nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, self.dim_output),
                                    )
    def forward(self, x):
        # print(f'Forward x: {x.device}')
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)

        return res, z, mu, logvar

    def encode(self, x):
        # print(f'Encode x: {x.device}')
        h = self.encoder(x)
        return self.fc1(h), self.fc2(h)

    def reparametrize(self, mu, logvar):
        # print(f'Reparametrize mu, logvar: {mu.device} {logvar.device}')
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps).to(self.device)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        # print(f'Decode z: {z.device}')
        return self.decoder(z)

    def get_latent_var(self, x):
        # print(f'Get latent var x: {x.device}')
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z
 
    def generate(self, z):
        # print(f'generate z: {z.device}')
        z = z.to(self.device)
        res = self.decode(z)
        return res