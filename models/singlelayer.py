import torch.nn as nn

class SingleLayer(nn.Module):
  def __init__(self):
    super(SingleLayer, self).__init__()

    # temporary, just to get things running
    self.m = nn.Conv2d(16, 33, 3)

  def forward(self, x):

    print('test')
    
