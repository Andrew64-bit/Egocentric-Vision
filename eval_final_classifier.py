from models.FinalClassifier import  MLPWithDropout, LSTMClassifier, TRNClassifier
from utils.loaders import FeaturesDataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from torchmetrics import Accuracy
from tqdm import tqdm
from utils.logger import logger
from utils.args import args

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

            outputs = model(x)
            #logger.info(f"Outputs: {outputs.size()}")
            #logger.info(f"Y: {y.size()}")

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
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        DEVICE = 'mps'

    
    #### ARCHITECTURE SETUP
    # Create the Network Architecture object
    if args.model == 'MLPWithDropout':
        model = MLPWithDropout(1024,8)
    elif args.model == 'LSTMClassifier':
        model = LSTMClassifier(1024,8)
    elif args.model == 'TRNClassifier':
        model = TRNClassifier()
    else:
        raise ValueError(f"Invalid model: {args.model}")
    
    logger.info(f"Model: {model}")



    # Load the test dataset and DataLoader
    test_dataset = FeaturesDataset(args.features_file, 'test')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4)
    logger.info(f"Test Dataset Size: {len(test_dataset)}")

    # Load the best model checkpoint
    model.load_state_dict(torch.load(f'./saved_models/{args.model}/final_{args.model}_epoch_{args.epoch}.pth'))  # or the best epoch
    model = model.to(DEVICE)

    # Evaluate the model
    test_loss, test_accuracy = evaluate(model, test_loader, DEVICE)
    logger.info(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
