IRMAS_TRAIN_DATA_PATH = 'D:/School/PhD/Modules/RMRI/GroupProject/dataset/train'
IRMAS_TEST_DATA_PATH = 'data/test'
IRMAS_TRAIN_FEATURE_BASEPATH = 'features/train'
IRMAS_TEST_FEATURE_BASEPATH = 'features/test'
IRMAS_TRAINING_META_PATH = './metadata/irmas_train_meta.csv'
IRMAS_TESTING_META_PATH = './metadata/irmas_test_meta.csv'
MODEL_WEIGHT_BASEPATH = './weights/'
MODEL_HISTORY_BASEPATH = './history/'
MODEL_MEANS_BASEPATH = './means/'
IRMAS_N_CLASSES = 11
TRAIN_SPLIT = 0.85
VALIDATION_SPLIT = 0.15
N_TRAINING_SET = 6705
MAX_EPOCH_NUM = 50
EARLY_STOPPING_EPOCH = 20
SGD_LR_REDUCE = 5
BATCH_SIZE = 16
ALLOWED_MODELS = ['han16', 'singlelayer', 'multilayer']
N_SEGMENTS_PER_TRAINING_FILE = 1
N_MEL_BANDS = 96
SEGMENT_DUR = 128
SAVE_MODEL = True
LOAD_MODEL = False
SAVE_MODEL_PATH = 'saved_models/singlelayer/singlelayer'
LOAD_MODEL_PATH = 'saved_models/singlelayer/singlelayer'