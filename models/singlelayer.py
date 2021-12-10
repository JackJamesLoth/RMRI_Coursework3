import torch.nn as nn
import torch

from settings import *

class SingleLayer(nn.Module):
  def __init__(self):
    super(SingleLayer, self).__init__()

    m_sizes = [50, 80]
    n_sizes = [1, 3, 5]
    n_filters = [128, 64, 32]
    maxpool_const = 4

    self.layers = nn.ModuleList()
    self.normLayers = nn.ModuleList()

    # Create layers
    for m in m_sizes:
      for i, n in enumerate(n_sizes):
        
        # NOTE: W_regularizer?

        # Create Conv2d layers
        self.layers.append(nn.Conv2d(1, n_filters[i], (m, n), padding='same'))

        # Create batchnorm layers
        self.normLayers.append(nn.BatchNorm2d(n_filters[i]))
    
    # Create other layers
    self.maxpool = nn.MaxPool2d((N_MEL_BANDS, 32))
    self.ELU = nn.ELU()
    self.drop = nn.Dropout(0.5)
    self.lin = nn.Linear(1792, 11)
    self.softmax = nn.Softmax(dim=1)


  def forward(self, x):

    seq = list()

    # run input through each layer
    for i in range(len(self.layers)):
      temp = self.layers[i](x)
      temp = self.normLayers[i](temp)
      temp = self.ELU(temp)
      temp = self.maxpool(temp)
      temp = torch.flatten(temp, start_dim=1)
      seq.append(temp)

    # Basically just reshapes the output to something that can be fed into a linear layer
    x = torch.cat(seq, dim=1)
    x = self.drop(x)

    # Final linear layer with softmax
    x = self.lin(x)
    #x = self.softmax(x)

    return x
    
