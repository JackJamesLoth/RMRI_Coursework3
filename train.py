import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from settings import *
from utils import *
from dataset import *
from models.singlelayer import SingleLayer

def train(model, optimizer, scaler, loss_fn, train, device):
    print('Training...')

    trainLossData = []

    # Training loop
    for i in range(MAX_EPOCH_NUM):
        print('Epoch {}/{}'.format(i+1, MAX_EPOCH_NUM))

        # Needed for average calculation
        avg = 0
        total_batches = len(train)
        j = 1

        for train_data in train:

            # Needed for autocast
            with torch.cuda.amp.autocast():

                # Get data
                input = train_data[0].float().to(device)
                labels = train_data[1].long().to(device)
                input.requires_grad = True

                # Zero the gradient
                optimizer.zero_grad()
                
                # Run model on data and get loss
                output = model(input)
                loss = loss_fn(output, labels)

                del input
                del output
                del labels

                # Update gradient
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Calculate average
            old_avg = avg
            avg = old_avg + ((loss - old_avg) / j)
            j += 1

        # save loss
        trainLossData.append(avg)
        print('Average loss: {}'.format(avg))

    # Save loss plot
    plt.plot(trainLossData)
    plt.ylabel('Training loss (ESR)')
    plt.xlabel('Epoch')
    plt.show()

def main():

    dev = getDevice()

    # Parse arguments
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-m',
                         action='store',
                         dest='model',
                         help='-m model to train')
    args = aparser.parse_args()

    # Load data
    print("Loading metadata")
    dataset = pd.read_csv(IRMAS_TRAINING_META_PATH, names=["filename", "class_id"])

    # Create test/val split
    print("Creating dataset split")
    X_train, X_val, y_train, y_val = train_test_split(list(dataset.filename),
                                                    to_categorical(np.array(dataset.class_id, dtype=int), IRMAS_N_CLASSES),
                                                    test_size=VALIDATION_SPLIT)

    # Create datasets
    print("Creating datasets")
    dataset_train = InstrTagDataset(X_train, y_train, args.model)
    dataset_val = InstrTagDataset(X_val, y_val, args.model)

    # Create dataloaders
    print("Creating dataloaders")
    dataloader_train = DataLoader(dataset_train, batch_size = BATCH_SIZE, num_workers=2, persistent_workers=True, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size = BATCH_SIZE, num_workers=2, persistent_workers=True, shuffle=True)

    # Create model
    model = SingleLayer()
    model.to(dev)
    
    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Create scaler
    scaler = torch.cuda.amp.GradScaler(growth_interval=1000)

    # Create loss function
    #loss = CategoricalCrossEntropy()
    loss = nn.CrossEntropyLoss()

    # Load model
    if LOAD_MODEL:
        print('loading model')
        checkpoint = torch.load(LOAD_MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Begin training
    train(model, optimizer, scaler, loss, dataloader_train, dev)

    # Save the model
    if SAVE_MODEL:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, SAVE_MODEL_PATH)   

if __name__ == "__main__":
    main()