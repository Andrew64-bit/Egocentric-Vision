from models.VAE import VAE
from torch.optim import Adam
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.logger import logger
from tqdm import tqdm
from utils.args import args

def loss_function(recon_x, x, mu, logvar):
    lamb = 0.0000001
    MSE = nn.MSELoss()
    lloss = MSE(recon_x,x)

    if lamb>0:
        KL_loss = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        lloss = lloss + lamb*KL_loss

    return lloss

def train(model, optimizer, epochs, device, train_loader_rgb, train_loader_emg, batch_size, scheduler):
    model.train()

    for epoch in range(epochs):
        overall_loss = 0

        for (rgb_batch_idx, (rgb_x, _)), (emg_batch_idx, (emg_x, _)) in tqdm(zip(enumerate(train_loader_rgb), enumerate(train_loader_emg))):
            
            inputs = Variable(rgb_x)
            inputs.to(device)

            targets = Variable(emg_x)
            targets.to(device)

            optimizer.zero_grad()

            reconstructed_target, latents, mu, logvar = model(inputs)
            loss = loss_function(reconstructed_target, targets, mu, logvar)
            train_loss += loss.data.item() * inputs.size(0)
            
            
            loss.backward()
            optimizer.step()

        scheduler.step()
        logger.info("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(rgb_batch_idx*batch_size))


if __name__ == '__main__':

    # --------DEVICE SETUP--------
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        DEVICE = 'mps'
        logger.info("------ USING APPLE SILICON GPU ------")

    # -------HYPERPARAMETERS------
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    STEP_SIZE = 10
    GAMMA = 0.1
    
    # Define the DataLoaders
    # geti item return a single clip embedding 1024
    # ------TO DO !!!!!-----------------------------
    train_loader_rgb = []
    train_loader_emg = []

    model = VAE().to(DEVICE)

    # Create Optimizer & Scheduler objects
    optimizer = Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # The loss function in VAE consists of reproduction loss and the Kullback–Leibler (KL) divergence.
    # The KL divergence is a metric used to measure the distance between two probability distributions.
    
    train(model, optimizer, EPOCHS, DEVICE, train_loader_rgb, train_loader_emg, BATCH_SIZE, scheduler)

    # TO DO (N.B)--> inside for epoch loop --> train function "train()"
    #if (epoch+1) % 10 == 0:
            #train_accuracy = evaluate(model, train_loader, DEVICE)
            #val_accuracy = evaluate(model, val_loader, DEVICE)
            #logger.info(f'[EPOCH {epoch+1}] Train Accuracy: {train_accuracy}')
            #logger.info(f'[EPOCH {epoch+1}] Val Accuracy: {val_accuracy}')
    torch.save(model.state_dict(), f'./saved_models/{args.model}/final_{args.model}_epoch_{EPOCHS}.pth')
    

    
