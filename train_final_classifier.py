from models.FinalClassifier import MLP, MLPWithDropout, LSTMClassifier, TransformerClassifier, LSTM_Emb_Classifier, TRNClassifier
from utils.loaders import FeaturesDataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch import nn
from torchmetrics import Accuracy
from tqdm import tqdm
from utils.logger import logger
from utils.args import args

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = correct / total
    return accuracy




if __name__ == '__main__':
    BATCH_SIZE = 32
    LR = float(args.lr) 
    #MOMENTUM = 0.9
    #WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = int(args.epochs) 
    STEP_SIZE = args.step_size
    STEP_ACC = args.step_acc
    GAMMA = 0.1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.backends.mps.is_available():
        DEVICE = 'mps'
        logger.info("------ USING APPLE SILICON GPU ------")

    #### DATA SETUP
    # Define the transforms to use on images
    dataset_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FeaturesDataset(args.features_file,'train', args.emg)
    val_dataset = FeaturesDataset(args.features_file,'test', args.emg) 

    # Define the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
    #logger.info(f"Train Dataset Size: {len(train_dataset)}")

    #### ARCHITECTURE SETUP
    # Create the Network Architecture object
    if args.model == 'MLP':
        model = MLP(1024,8)
    elif args.model == 'MLPWithDropout':
        if args.emg:
            num_bottleneck = 32
            num_bottleneck1 = 16
            d_model = 64
        else:
            d_model = 1024
            num_bottleneck = 512
            num_bottleneck1 = 256
        model = MLPWithDropout(d_model,8, num_bottleneck, num_bottleneck1)
    elif args.model == 'TransformerClassifier':
        if args.emg:
        # Iperparametri
            d_model = args.input_size
            num_heads = 4
            num_layers = 4
            d_ff = args.input_size*2
            max_seq_length = 5
            num_classes = 8
            num_bottleneck = 32
            dropout = 0.3
        else:
            d_model = 1024
            num_heads = 8
            num_layers = 4
            d_ff = 2048
            max_seq_length = 5
            num_classes = 8
            num_bottleneck = 512
            dropout = 0.3
        model = TransformerClassifier(d_model, num_heads, num_layers, d_ff, max_seq_length, num_classes, num_bottleneck, dropout)
    elif args.model == 'LSTMClassifier':
        if args.emg:
            model = LSTMClassifier(args.input_size,8)
        else:
            model = LSTMClassifier(1024,8)
    elif args.model == 'TRNClassifier':
        if args.emg:
            model = TRNClassifier(num_bottleneck=16, clip_feature_dim=64, num_clips=5, num_class=8, dropout=0.5)
        else:
            model = TRNClassifier()
    elif args.model == 'LSTM_Emb_Classifier':
        model = LSTM_Emb_Classifier(input_dim=args.input_size, num_class=8, hidden_dim=128, num_layers=4, dropout=0.5)
    else:
        raise ValueError(f"Invalid model: {args.model}")
        
    logger.info(f"Model: {model}")
    logger.info(f"len train_dataset: {len(train_dataset)}")
    logger.info(f"len train_loader: {len(train_loader)}")


    #### TRAINING SETUP
    # Move model to device before passing it to the optimizer
    model = model.to(DEVICE)
    #model.apply(init_weights)

    # Create Optimizer & Scheduler objects
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)


    #### TRAINING LOOP
    model.train()
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = [0.0, 0]
        for i_val,(x, y) in tqdm(enumerate(train_loader)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            print(f'x: {x.shape}')
            # Controlla se ci sono nan nei dati di input
            if torch.isnan(x).sum() > 0 or torch.isinf(x).sum() > 0:
                print("Input contains nan or inf values")
                x = torch.nan_to_num(x, nan=0.0, posinf=1e8, neginf=-1e8)
            #logger.info(f"X: {x[0][0]}")
            # Category Loss
            #logger.info(f"X: {x.size()}")

            
            #logger.info(f"X: {x.size()}")
            #logger.info(f'x : {x}')

            outputs = model(x)
            # Log details about the outputs
            #logger.info(f"Output type: {cls_o.logits.shape}")

                
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, y.long())
            #logger.info(f'Outputs : {outputs}')
            #logger.info(f'y : {y.long()}')
            #logger.info(f'Loss = {loss}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss[0] += loss.item()
            epoch_loss[1] += x.size(0)
            #logger.info(f'epoch_loss[0]={epoch_loss[0]}+loss.item()={loss.item()}')
            #logger.info(f'epoch_loss[1]={epoch_loss[1]}+x.size(0)={x.size(0)}')
            if (i_val + 1) % (len(train_loader) // 5) == 0:
                logger.info("[{}/{}]".format(i_val + 1, len(train_loader)))
        
        scheduler.step()
        logger.info(f'[EPOCH {epoch+1}] Avg. Loss: {epoch_loss[0] / epoch_loss[1]}')


        #save checkpoint in a file
        if (epoch+1) % STEP_ACC == 0:
            train_accuracy = evaluate(model, train_loader, DEVICE)
            val_accuracy = evaluate(model, val_loader, DEVICE)
            logger.info(f'[EPOCH {epoch+1}] Train Accuracy: {train_accuracy}')
            logger.info(f'[EPOCH {epoch+1}] Val Accuracy: {val_accuracy}')
            torch.save(model.state_dict(), f'./saved_models/{args.model}/final_{args.model}{"_emg" if args.emg else ""}_epoch_{epoch+1}.pth')
            logger.info(f'Current LR: {scheduler.get_last_lr()}')
        


