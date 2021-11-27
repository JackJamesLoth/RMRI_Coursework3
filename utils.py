import torch
import numpy as np

def getDevice():
      # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU
    if is_cuda:
        dev = torch.device("cuda")
        print('Using GPU')
    else:
        dev = torch.device("cpu")
        print('GPU not available, using CPU')

    return dev

def _load_features(self, filenames):
    features = list()
    for filename in filenames:
        feature_filename = os.path.join(IRMAS_TRAIN_FEATURE_BASEPATH, self.model_module.BASE_NAME,
                                        "{}.npy".format(filename))
        feature = np.load(feature_filename)
        feature -= self.dataset_mean
        features.append(feature)

    if K.image_dim_ordering() == 'th':
        features = np.array(features).reshape(-1, 1, self.model_module.N_MEL_BANDS, self.model_module.SEGMENT_DUR)
    else:
        features = np.array(features).reshape(-1, self.model_module.N_MEL_BANDS, self.model_module.SEGMENT_DUR, 1)
    return features

def _get_extended_data(self, inputs, targets):
    extended_inputs = list()
    for i in range(0, self.model_module.N_SEGMENTS_PER_TRAINING_FILE):
        extended_inputs.extend(['_'.join(list(x)) for x in zip(inputs, [str(i)]*len(inputs))])
    extended_inputs = np.array(extended_inputs)
    extended_targets = np.tile(np.array(targets).reshape(-1),
                                self.model_module.N_SEGMENTS_PER_TRAINING_FILE).reshape(-1, IRMAS_N_CLASSES)
    return extended_inputs, extended_targets

def _batch_generator(self, inputs, targets):
    assert len(inputs) == len(targets)
    while True:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - BATCH_SIZE + 1, BATCH_SIZE):
            excerpt = indices[start_idx:start_idx + BATCH_SIZE]
            if self.in_memory_data:
                yield inputs[excerpt], targets[excerpt]
            else:
                yield self._load_features(inputs[excerpt]), targets[excerpt]

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]