import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from settings import *
from utils import *
from dataset import *
from models.singlelayer import SingleLayer

def train(model, optimizer, loss_fn, train, device):
    print('training...')

    # Training loop
    for i in range(MAX_EPOCH_NUM):
        print('Epoch {}/{}'.format(i+1, MAX_EPOCH_NUM))

        for train_data in train:

            input = train_data[0].float().to(device)
            labels = train_data[1].float().to(device)

            train_input = torch.transpose(input, 0,1)
            #train_labels = torch.transpose(labels, 1,2)

            optimizer.zero_grad()
            output = model(train_input)
            loss = loss_fn(output, train_labels)

            

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
    dataset = pd.read_csv(IRMAS_TRAINING_META_PATH, names=["filename", "class_id"])

    # Create test/val split
    X_train, X_val, y_train, y_val = train_test_split(list(dataset.filename),
                                                    to_categorical(np.array(dataset.class_id, dtype=int), IRMAS_N_CLASSES),
                                                    test_size=VALIDATION_SPLIT)

    # Create datasets
    dataset_train = InstrTagDataset(X_train, y_train, args.model)
    dataset_val = InstrTagDataset(X_val, y_val, args.model)

    # Create dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size = BATCH_SIZE, num_workers=2, persistent_workers=True)
    dataloader_val = DataLoader(dataset_val, batch_size = BATCH_SIZE, num_workers=2, persistent_workers=True)

    # Create model
    model = SingleLayer()
    model.to(dev)
    
    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Create loss function
    # NOTE: Should be categorical cross entropy loss, which I think is slightly different from CrossEntropyLoss
    loss = nn.CrossEntropyLoss()

    # Begin training
    train(model, optimizer, loss, dataloader_train, dev)

if __name__ == "__main__":
    main()