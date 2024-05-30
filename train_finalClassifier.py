from utils.loaders import FeaturesDataset
from models.FinalClassifier import Classifier
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import AlexNet_Weights
import torch.nn.functional as F
from torchmetrics import Accuracy
from tqdm import tqdm
from utils.logger import logger

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

    #### DATA SETUP
    # Define the transforms to use on images
    dataset_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define the Dataset object for training & testing
    train_dataset = FeaturesDataset("./saved_features/saved_feat_I3D_10_dense_D1_test.pkl",'train')
    #test_dataset = PACSDataset(domain='sketch', transform=dataset_transform)

    # Define the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    #test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)
    logger.info(f"Train Dataset Size: {len(train_dataset)}")

    #### ARCHITECTURE SETUP
    # Create the Network Architecture object
    model = Classifier(1024,8)
    logger.info(f"Model: {model}")

    #### TRAINING SETUP
    # Move model to device before passing it to the optimizer
    model = model.to(DEVICE)

    # Create Optimizer & Scheduler objects
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)


    #### TRAINING LOOP
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = [0.0, 0]
        for i_val,(x, y) in tqdm(enumerate(train_loader)):
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Category Loss
            cls_o = model(x)
            loss = F.cross_entropy(cls_o, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss[0] += loss.item()
            epoch_loss[1] += x.size(0)
            if (i_val + 1) % (len(train_loader) // 5) == 0:
                logger.info("[{}/{}]".format(i_val + 1, len(train_loader)))
            
        scheduler.step()
        logger.info(f'[EPOCH {epoch+1}] Avg. Loss: {epoch_loss[0] / epoch_loss[1]}')
        #save checkpoint in a file
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'./saved_models/final_classifier_epoch_{epoch+1}.pth')

    test_dataset = FeaturesDataset("./saved_features/saved_feat_I3D_10_dense_D1_test.pkl",'test')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4)
    logger.info(f"Test Dataset Size: {len(test_dataset)}")


    # Model Evaluation

    model.eval()
    accuracy = Accuracy()
    for i_val,(x, y) in tqdm(enumerate(test_loader)):
        x, y = x.to(DEVICE), y.to(DEVICE)
        cls_o = model(x)
        acc = accuracy(cls_o, y)
        


