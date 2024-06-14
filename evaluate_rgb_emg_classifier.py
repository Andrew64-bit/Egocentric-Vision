from models.FinalClassifier import MLPWithDropout, LSTMClassifier, TRNClassifier
from utils.loaders import FeaturesDataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch import nn
from torchmetrics import Accuracy
from tqdm import tqdm
from utils.logger import logger
from utils.args import args

def evaluate(model_emg,model_rgb, data_loader_emg, data_loader_rgb,p, device):
    model_emg.eval()
    model_rgb.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ((emg, y), (rgb, y1)) in zip(data_loader_emg,data_loader_rgb):
            emg, y, rgb, y1 = emg.to(device), y.to(device), rgb.to(device), y1.to(device)

            if y != y1:
                raise ValueError(f"Invalid labels: {y} -> {y1}")


            outputs_emg = model_emg(emg)
            outputs_rgb = model_rgb(rgb)
            #outputs = forward_pass(outputs_emg,outputs_rgb,p)
            outputs = outputs_rgb * p + outputs_emg * (1-p)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = correct / total
    return accuracy

def forward_pass(outputs_emg,outputs_rgb,p):
    softmax = nn.Softmax(dim=1)
    outputs_emg = softmax(outputs_emg)
    outputs_rgb = softmax(outputs_rgb)
    outputs = outputs_rgb * p + outputs_emg * (1-p)
    return outputs



if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        DEVICE = 'mps'
        logger.info("------ USING APPLE SILICON GPU ------")


    val_dataset_rgb = FeaturesDataset(args.features_file_rgb,'test', False) 
    val_dataset_emg = FeaturesDataset(args.features_file_emg,'test', True) 

    # Define the DataLoaders
    loader_rgb = DataLoader(val_dataset_rgb, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
    loader_emg = DataLoader(val_dataset_emg, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
    #logger.info(f"Train Dataset Size: {len(train_dataset)}")

    #### ARCHITECTURE SETUP
    # Create the Network Architecture object
    # Combine different model emg rgb
    if hasattr(args, 'model_emg'):
        if args.model_emg == 'MLPWithDropout':
            d_model = 1024
            num_bottleneck = 512
            num_bottleneck1 = 256
            dropout = 0.5
            model_emg = MLPWithDropout(d_model,8, num_bottleneck, num_bottleneck1, dropout)
        elif args.model_emg == 'LSTMClassifier':

            d_model = 1024
            dropout = 0.5
            num_layers = 1
            hidden_dim = 128
            model_emg = LSTMClassifier(d_model,8,hidden_dim,num_layers, dropout)
        
        elif args.model_emg == 'TRNClassifier':
            model_emg = TRNClassifier()
        else:
            raise ValueError(f"Invalid model: {args.model_emg}")
        
        if args.model_rgb == 'MLPWithDropout':
            d_model = 1024
            num_bottleneck = 512
            num_bottleneck1 = 256
            dropout = 0.5
            model_rgb = MLPWithDropout(d_model,8, num_bottleneck, num_bottleneck1, dropout)
        elif args.model_rgb == 'LSTMClassifier':

            d_model = 1024
            dropout = 0.5
            num_layers = 1
            hidden_dim = 128
            model_rgb = LSTMClassifier(d_model,8,hidden_dim,num_layers, dropout)
        
        elif args.model_rgb == 'TRNClassifier':
            model_rgb = TRNClassifier()
        else:
            raise ValueError(f"Invalid model: {args.model_rgb}")

    # Combine same model emg rgb
    else:
        if args.model == 'MLPWithDropout':
            d_model = 1024
            num_bottleneck = 512
            num_bottleneck1 = 256
            dropout = 0.5
            model_emg = MLPWithDropout(d_model,8, num_bottleneck, num_bottleneck1, dropout)


            d_model = 1024
            num_bottleneck = 512
            num_bottleneck1 = 256
            dropout = 0.5
            model_rgb = MLPWithDropout(d_model,8, num_bottleneck, num_bottleneck1, dropout)
        elif args.model == 'LSTMClassifier':

            d_model = 1024
            dropout = 0.5
            num_layers = 1
            hidden_dim = 128
            model_emg = LSTMClassifier(d_model,8,hidden_dim,num_layers, dropout)

            d_model = 1024
            dropout = 0.5
            num_layers = 1
            hidden_dim = 128
            model_rgb = LSTMClassifier(d_model,8,hidden_dim,num_layers, dropout)
        
        elif args.model == 'TRNClassifier':
            model_emg = TRNClassifier()

            model_rgb = TRNClassifier()
        else:
        
            raise ValueError(f"Invalid model: {args.model}")
            
    #logger.info(f"Model EMG: {model_emg}")
    #logger.info(f"len train_dataset: {len(val_dataset_emg)}")
    #logger.info(f"len train_loader: {len(loader_emg)}")

    #logger.info(f"Model RGB: {model_rgb}")
    #logger.info(f"len train_dataset: {len(val_dataset_rgb)}")
    #logger.info(f"len train_loader: {len(loader_rgb)}")
    logger.info(f"prob: {args.prob}")
    model_emg.load_state_dict(torch.load(args.path_emg))
    model_rgb.load_state_dict(torch.load(args.path_rgb))
    #### TRAINING SETUP
    # Move model to device before passing it to the optimizer
    model_emg = model_emg.to(DEVICE)
    model_rgb = model_rgb.to(DEVICE)



    accuracy = evaluate(model_emg, model_rgb,loader_emg,loader_rgb,float(args.prob),DEVICE)
    logger.info(f"\n\nAccuracy: {accuracy}")

