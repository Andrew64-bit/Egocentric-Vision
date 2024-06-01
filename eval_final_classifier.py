from models.FinalClassifier import MLP, MLPWithDropout, LSTMClassifier, TRN_classifier
from utils.loaders import FeaturesDataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import AlexNet_Weights
import torch.nn.functional as F
from torchmetrics import Accuracy
from tqdm import tqdm
from utils.logger import logger
from utils.args import args
from transformers import ViTConfig, ViTForImageClassification

# Evaluate the model
def evaluate(model, test_loader, device):
    model.eval()
    accuracy_metric = Accuracy(task="multiclass", num_classes=8).to(device)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)
            if args.model == 'Transformer':
                # Reshape il tensore in [batch_size, num_channels, height, width]
                # Ogni vettore di 1024 elementi viene trasformato in una matrice 32x32
                x = x.view(1, 1, 32, 32)  # batch_size=32, num_channels=1, height=32, width=32

            outputs = model(x)
            #logger.info(f"Outputs: {outputs.size()}")
            #logger.info(f"Y: {y.size()}")
            if args.model == 'Transformer':
                outputs = outputs.logits

            loss = F.cross_entropy(outputs, y.long())
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == y).sum().item()

            # Update accuracy metric
            accuracy_metric.update(preds, y)

    avg_loss = total_loss / total_samples
    avg_accuracy = accuracy_metric.compute().item()

    return avg_loss, avg_accuracy

if __name__ == '__main__':
    BATCH_SIZE = 32
    LR = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    STEP_SIZE = 30
    GAMMA = 0.1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        DEVICE = 'mps'
    NUM_EPOCHS = 50
    
    #### ARCHITECTURE SETUP
    # Create the Network Architecture object
    if args.model == 'MLP':
        model = MLP(1024,8)
    elif args.model == 'TRN':
        model = TRN_classifier()
    elif args.model == 'MLPWithDropout':
        model = MLPWithDropout(1024,8)
    elif args.model == 'Transformer':
        # Internal configuration of the ViT model
        configuration = ViTConfig()
        configuration.num_channels = 1
        configuration.image_size = 32
        configuration.num_labels = 8
        configuration.hidden_size = 256
        configuration.intermediate_size = 512
        configuration.num_attention_heads = 4
        configuration.num_hidden_layers = 4
        model = ViTForImageClassification(configuration)

    elif args.model == 'LSTMClassifier':
        model = LSTMClassifier(1024,8)
    else:
        raise ValueError(f"Invalid model: {args.model}")
    
    logger.info(f"Model: {model}")

    #### TRAINING SETUP
    # Move model to device before passing it to the optimizer
    model = model.to(DEVICE)

    # Load the test dataset and DataLoader
    test_dataset = FeaturesDataset("./saved_features/saved_feat_I3D_10_dense_D1_test.pkl", 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4)
    logger.info(f"Test Dataset Size: {len(test_dataset)}")

    # Load the best model checkpoint
    model.load_state_dict(torch.load(f'./saved_models/{args.model}/final_{args.model}_epoch_50.pth'))  # or the best epoch
    model = model.to(DEVICE)

    # Evaluate the model
    test_loss, test_accuracy = evaluate(model, test_loader, DEVICE)
    logger.info(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
