from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer

from utils import *

class InstrTagDatasetTest(Dataset):
    def __init__(self, dataset, model_name):

        # Definitions
        self.feature_filenames = os.listdir(os.path.join(IRMAS_TEST_FEATURE_BASEPATH, model_name))
        self.model_name = model_name
        self.dataset_mean = np.load(os.path.join(MODEL_MEANS_BASEPATH, "{}_mean.npy".format(model_name)))

        # Create list of filename
        self.X = list(dataset.filename)

        # Create array of targets
        targets = [[int(category) for category in target.split()] for target in dataset.class_id]
        self.ml_binarizer = MultiLabelBinarizer().fit(targets)
        self.y_true = self.ml_binarizer.transform(targets)

    def __getitem__(self, index):
        return self.X[index], self.load_features(self.X[index]), self.y_true[index]
        
    def __len__(self):
        return len(self.X)

    def load_features(self, audio_filename):
        features = list()
        for feature_filename in self.feature_filenames:
            if audio_filename in feature_filename:
                filename_full_path = os.path.join(IRMAS_TEST_FEATURE_BASEPATH,
                                                  self.model_name,
                                                  feature_filename)
                feature = np.load(filename_full_path)
                feature -= self.dataset_mean
                features.append(feature)

        features = np.array(features).reshape(-1, 1, N_MEL_BANDS, SEGMENT_DUR)
        return features

    def get_y_true(self):
        return self.y_true