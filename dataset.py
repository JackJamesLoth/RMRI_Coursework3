from torch.utils.data import Dataset, DataLoader

from utils import *

class InstrTagDataset(Dataset):
    def __init__(self, input, target, model_name):
        dataset_mean = np.load(os.path.join(MODEL_MEANS_BASEPATH, "{}_mean.npy".format(model_name)))
        extended_x, extended_y = get_extended_data(input, target)
        self.y = extended_y
        self.X = load_features(extended_x, model_name, dataset_mean)

    def __getitem__(self, index):
        return self.X[0]
        
    def __len__(self):
        return self.X.shape[0]