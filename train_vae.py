from models.VAE import FC_VAE
from torch.optim import Adam
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.logger import setup_logger
from tqdm import tqdm
from utils.args import args
import numpy as np

def loss_function(recon_x, x, mu, logvar):

    lamb = 0.0000001
    MSE = nn.MSELoss()
    mse = MSE(recon_x,x)
    norm_original_data = torch.norm(x)

    # Calcola l'errore quadratico medio normalizzato
    nmse = mse / (norm_original_data ** 2)
    lloss = nmse * 100

    if lamb>0:
        KL_loss = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        lloss = lloss + lamb*KL_loss

    return lloss

def train(model, optimizer, epochs, device, train_loader_rgb, train_loader_emg, batch_size, scheduler):
    logger = setup_logger("LOG", "00_log_training_VAE")
    
    model.train()
    # reset parameter gradients
    model.zero_grad()

    for epoch in range(epochs):
        overall_loss = 0

        for (rgb_batch_idx, (rgb_x, _)), (emg_batch_idx, (emg_x, _)) in tqdm(zip(enumerate(train_loader_rgb), enumerate(train_loader_emg))):
            # print(f"Input: {rgb_x}")
            inputs = Variable(rgb_x)
            inputs = inputs.to(device)
            # print(f'DEVICE used: {device}')
            # print(f'Input: {inputs.device}')
            if not torch.equal(rgb_x, emg_x):
                raise ValueError(f"RGB and EMG inputs are not aligned!\n{rgb_x}\n{emg_x}")
            targets = Variable(emg_x)
            targets = targets.to(device)

            optimizer.zero_grad()

            reconstructed_target, latents, mu, logvar = model(inputs)
            loss = loss_function(reconstructed_target, targets, mu, logvar)
            overall_loss += loss.data.item() * inputs.size(0)
            
            
            loss.backward()
            optimizer.step()

        scheduler.step()
        logger.info(f"\tEpoch, {epoch + 1}, \tAverage Loss: , {overall_loss/(rgb_batch_idx*batch_size)}")


# Funzione di valutazione
def evaluate(model, device, test_loader_rgb, test_loader_emg):
    logger = setup_logger("LOG", "00_log_evaluation_VAE")
    model.eval()
    test_loss = 0
    all_reconstructed = []
    all_original = []
    
    with torch.no_grad():
        for (rgb_batch_idx, (rgb_x, _)), (emg_batch_idx, (emg_x, _)) in tqdm(zip(enumerate(test_loader_rgb), enumerate(test_loader_emg))):
            
            inputs = Variable(rgb_x).to(device)
            reconstructed, z, mu, logvar = model(inputs)
            
            loss = loss_function(reconstructed, inputs, mu, logvar)
            test_loss += loss.item()
            
            all_reconstructed.append(reconstructed.cpu().numpy())
            all_original.append(inputs.cpu().numpy())
    
    test_loss /= len(test_loader_rgb.dataset)
    print(f'Test Loss: {test_loss:.4f}')
    
    all_reconstructed = np.concatenate(all_reconstructed, axis=0)
    all_original = np.concatenate(all_original, axis=0)
    
    return all_reconstructed, all_original

def train_emg(model, optimizer, epochs, device, train_loader_rgb, train_loader_emg, batch_size, scheduler):
    logger = setup_logger("LOG", "00_log_training_VAE")

    model.train()
    # reset parameter gradients
    model.zero_grad()

    for epoch in range(epochs):
        overall_loss = 0

        for (rgb_batch_idx, sample_rgb), (emg_batch_idx, sample_emg) in tqdm(zip(enumerate(train_loader_rgb), enumerate(train_loader_emg))):
            # print(f"Input: {rgb_x}")
            inputs = Variable(sample_rgb["features"])
            inputs = inputs.to(device)
            # print(f'DEVICE used: {device}')
            # print(f'Input: {inputs.device}')

            targets = Variable(sample_emg["features"])
            targets = targets.to(device)

            optimizer.zero_grad()

            reconstructed_target, latents, mu, logvar = model(inputs)
            loss = loss_function(reconstructed_target, targets, mu, logvar)
            overall_loss += loss.data.item() * inputs.size(0)
            
            
            loss.backward()
            optimizer.step()

        scheduler.step()
        logger.info(f"\tEpoch, {epoch + 1}, \tAverage Loss: , {overall_loss/(rgb_batch_idx*batch_size)}")


# Funzione di valutazione
def evaluate_emg(model, device, test_loader_rgb, test_loader_emg):
    logger = setup_logger("LOG", "00_log_evaluation_VAE")
    model.eval()
    test_loss = 0
    all_reconstructed = []
    all_original = []
    
    with torch.no_grad():
        for (rgb_batch_idx, sample_rgb), (emg_batch_idx, sample_emg) in tqdm(zip(enumerate(test_loader_rgb), enumerate(test_loader_emg))):

            inputs = Variable(sample_rgb["features"]).to(device)
            reconstructed, z, mu, logvar = model(inputs)
            
            loss = loss_function(reconstructed, inputs, mu, logvar)
            test_loss += loss.item()
            
            all_reconstructed.append(reconstructed.cpu().numpy())
            all_original.append(inputs.cpu().numpy())
    
    test_loss /= len(test_loader_rgb.dataset)
    print(f'Test Loss: {test_loss:.4f}')
    
    all_reconstructed = np.concatenate(all_reconstructed, axis=0)
    all_original = np.concatenate(all_original, axis=0)
    
    return all_reconstructed, all_original

def train_tuning(model, optimizer, epochs, device, train_loader_rgb, train_loader_emg, batch_size, scheduler):
    logger = setup_logger("LOG", "00_log_training_VAE")

    model.train()
    # reset parameter gradients
    model.zero_grad()

    for epoch in range(epochs):
            overall_loss = 0

            for (rgb_batch_idx, rgb_input), (emg_batch_idx, emg_output) in zip(enumerate(train_loader_rgb), enumerate(train_loader_emg)):
                # print(f"Input: {rgb_x}")
                inputs = Variable(rgb_input)
                inputs = inputs.to(device)
                # print(f'DEVICE used: {device}')
                # print(f'Input: {inputs.device}')

                targets = Variable(emg_output)
                targets = targets.to(device)

                optimizer.zero_grad()

                reconstructed_target, latents, mu, logvar = model(inputs)
                # print("Input shape: ", inputs.shape)
                #print("Reconstructed target shape: ", reconstructed_target.shape)
                #print("Targets shape: ", targets.shape)
                loss = loss_function(reconstructed_target, targets, mu, logvar)
                overall_loss += loss.data.item() * inputs.size(0)
                
                
                loss.backward()
                optimizer.step()

            scheduler.step()
            logger.info(f"\tEpoch, {epoch + 1}, \tAverage Loss: , {overall_loss/(rgb_batch_idx*batch_size)}")


# Funzione di valutazione
def evaluate_tuning(model, device, test_loader_rgb, test_loader_emg):
    logger = setup_logger("LOG", "00_log_evaluation_VAE")
    model.eval()
    test_loss = 0
    all_reconstructed = []
    all_original = []
    
    with torch.no_grad():
        for (rgb_batch_idx, rgb_input), (emg_batch_idx, emg_output) in zip(enumerate(test_loader_rgb), enumerate(test_loader_emg)):

            inputs = Variable(rgb_input).to(device)
            reconstructed, z, mu, logvar = model(inputs)
            
            loss = loss_function(reconstructed, emg_output, mu, logvar)
            test_loss += loss.item()
            
            all_reconstructed.append(reconstructed.cpu().numpy())
            all_original.append(inputs.cpu().numpy())
    
    test_loss /= len(test_loader_rgb.dataset)
    print(f'Test Loss: {test_loss:.4f}')
    
    all_reconstructed = np.concatenate(all_reconstructed, axis=0)
    all_original = np.concatenate(all_original, axis=0)
    
    return all_reconstructed, all_original

if __name__ == '__main__':

    # --------DEVICE SETUP--------
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        DEVICE = 'mps'
        #logger.info("------ USING APPLE SILICON GPU ------")

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

    model = FC_VAE(dim_input=1024, nz=64).to(DEVICE)

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
    

    
