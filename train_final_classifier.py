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

if __name__ == '__main__':
    BATCH_SIZE = 32
    LR = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    STEP_SIZE = 10
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
    #logger.info(f"Train Dataset Size: {len(train_dataset)}")

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
        
    #logger.info(f"Model: {model}")

    #### TRAINING SETUP
    # Move model to device before passing it to the optimizer
    model = model.to(DEVICE)

    # Create Optimizer & Scheduler objects
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)


    #### TRAINING LOOP
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = [0.0, 0]
        for i_val,(x, y) in tqdm(enumerate(train_loader)):
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Category Loss
            #logger.info(f"X: {x.size()}")

            if args.model == 'Transformer':
                # Reshape il tensore in [batch_size, num_channels, height, width]
                # Ogni vettore di 1024 elementi viene trasformato in una matrice 32x32
                x = x.view(BATCH_SIZE, 1, 32, 32)  # batch_size=32, num_channels=1, height=32, width=32
            
            #logger.info(f"X: {x.size()}")

            outputs = model(x)
            # Log details about the outputs
            #logger.info(f"Output type: {cls_o.logits.shape}")

            if args.model == 'Transformer':
                outputs = outputs.logits

            loss = F.cross_entropy(outputs, y.long())

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
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f'./saved_models/{args.model}/final_{args.model}_epoch_{epoch+1}.pth')
        if (epoch+1) % STEP_SIZE == 0:
            logger.info(f'Current LR: {scheduler.get_last_lr()}')

        


