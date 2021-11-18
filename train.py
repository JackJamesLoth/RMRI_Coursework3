import argparse
import torch
import torch.optim as optim
import torch.nn as nn

from settings import *
from models.singlelayer import SingleLayer

def train(model, optimizer, loss):
    print('training...')

    # Training loop
    for i in range(MAX_EPOCH_NUM):
        print('Epoch {}/{}'.format(i+1, MAX_EPOCH_NUM))

def main():

    # Parse arguments
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-m',
                         action='store',
                         dest='model',
                         help='-m model to train')
    args = aparser.parse_args()

    # Create dataset
    

    # Create model
    model = SingleLayer()

    # Create optimizer
    optimizer = optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Create loss function
    # NOTE: Should be categorical cross entropy loss, which I think is slightly different from CrossEntropyLoss
    loss = nn.CrossEntropyLoss()

    # Begin training
    train(model, optimizer, loss)

if __name__ == "__main__":
    main()