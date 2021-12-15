import argparse
import importlib
import os
import numpy as np
import librosa

from settings import *


def compute_spectrograms(filename):
    out_rate = 12000
    N_FFT = 512
    HOP_LEN = 256

    frames, rate = librosa.load(filename, sr=out_rate, mono=True)
    if len(frames) < out_rate*3:
        # if less then 3 second - can't process
        raise Exception("Audio duration is too short")

    logam = librosa.power_to_db
    melgram = librosa.feature.melspectrogram
    x = logam(melgram(y=frames, sr=out_rate, hop_length=HOP_LEN,
                      n_fft=N_FFT, n_mels=N_MEL_BANDS) ** 2,
              ref=1.0)

    # now going through spectrogram with the stride of the segment duration
    for start_idx in range(0, x.shape[1] - SEGMENT_DUR + 1, SEGMENT_DUR):
        yield x[:, start_idx:start_idx + SEGMENT_DUR]

# Go through all train/test files, compute spectrogram and save in features dir
def preprocess(model_module):
    for data_path, feature_path in [(IRMAS_TEST_DATA_PATH, IRMAS_TEST_FEATURE_BASEPATH)]:
        for root, dirs, files in os.walk(data_path):
            files = [filename for filename in files if filename.endswith('.wav')]
            for filename in files:
                for i, spec_segment in enumerate(compute_spectrograms(os.path.join(root, filename))):
                    feature_filename = os.path.join(feature_path, model_module,
                                                    "{filename}_{segment_idx}".format(filename=filename,
                                                                                      segment_idx=i))
                    np.save(feature_filename, spec_segment)

def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-m',
                         action='store',
                         dest='model',
                         help='-m model for preprocessing')
    args = aparser.parse_args()

    if not args.model:
        aparser.error('Please, specify the model!')
    try:
        if args.model in ALLOWED_MODELS:
            #model_module = importlib.import_module(".{}".format(args.model), "experiments.models")

            # just set the model_module to the name of the model for now
            # TODO change later?
            model_module = args.model
            print("{} imported as 'model'".format(args.model))
        else:
            print("The specified model is not allowed")
    except ImportError as e:
        print(e)
    preprocess(model_module)


if __name__ == "__main__":
    main()