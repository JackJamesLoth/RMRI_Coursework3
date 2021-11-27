import torch.nn as nn

class SingleLayer(nn.Module):
  def __init__(self):
    super(SingleLayer, self).__init__()

    m_sizes = [50, 80]
    n_sizes = [1, 3, 5]
    n_filters = [128, 64, 32]
    maxpool_const = 4

    self.layers = nn.ModuleList()

    # Create layers
    for m in m_sizes:
      for i, n in enumerate(n_sizes):
        
        # NOTE: W_regularizer
        self.layers.append(nn.Conv2d(1, n_filters[i], (m, n)))



  def forward(self, x):

    # run inut thruogh each layer
    for i in range(len(self.layers)):
      x = self.layers[0](x)
    
