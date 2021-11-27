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

def train(model, optimizer, loss):
    print('training...')

    # Training loop
    for i in range(MAX_EPOCH_NUM):
        print('Epoch {}/{}'.format(i+1, MAX_EPOCH_NUM))

def main():

    dev = getDevice()

    # Parse arguments
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-m',
                         action='store',
                         dest='model',
                         help='-m model to train')
    args = aparser.parse_args()

    # Create dataset
    dataset = pd.read_csv(IRMAS_TRAINING_META_PATH, names=["filename", "class_id"])
    X_train, X_val, y_train, y_val = train_test_split(list(dataset.filename),
                                                      to_categorical(np.array(dataset.class_id, dtype=int), 9),
                                                      test_size=VALIDATION_SPLIT)

    print(X_train)
    print(X_val)
    
    #dataset = InstrTagDataset()

    # Create dataloaders
    #dataloader = Dataloader(dataset, batch_size = BATCH_SIZE, num_workers=2, persistent_workers=True)

    # Create model
    model = SingleLayer()
    model.to(dev)
    
    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Create loss function
    # NOTE: Should be categorical cross entropy loss, which I think is slightly different from CrossEntropyLoss
    loss = nn.CrossEntropyLoss()

    # Begin training
    #train(model, optimizer, loss)

if __name__ == "__main__":
    main()