from models.VAE import VAE
from torch.optim import Adam
import torch
import torch.nn as nn
from utils.logger import logger
from tqdm import tqdm
from utils.args import args

def train(model, optimizer, epochs, device, train_loader, batch_size, x_dim, loss_function):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in tqdm(enumerate(train_loader)):
            x = x.view(batch_size, x_dim).to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        logger.info("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
    return overall_loss

if __name__ == '__main__':

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        DEVICE = 'mps'
        logger.info("------ USING APPLE SILICON GPU ------")

    BATCH_SIZE = 32
    EPOCHS = 50
    model = VAE().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # The loss function in VAE consists of reproduction loss and the Kullback–Leibler (KL) divergence.
    # The KL divergence is a metric used to measure the distance between two probability distributions.
    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD
    
    train(model, optimizer, epochs=EPOCHS, device=DEVICE)

    
