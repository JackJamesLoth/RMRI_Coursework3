# Need to create 1D array of predicted values and actual values (easy to get from dataloader and model output)
# Use sklearn metrics
# Compute metrics for different threshold values, getting rid of unconfident values
# Sum together predictions for same file, average and make final prediction based on that
# s1 and s2 are different ways of summing predictions together

# Loads target values into binarized array to indicate correct class for each batch
# For each batch, it predicts the class using the model output (batch is the features from one specific file)
# This is stored in a single prediction array
# Each of these arrays will be size [N,11] (since I think there are 11 classes total)


import argparse
import importlib
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from settings import *
from dataset_test import *
from models.singlelayer import SingleLayer

class Evaluator(object):
    def __init__(self, y_true, dev, eval_strat='s2'):

        # Create empty arrays
        self.y_true = y_true
        self.y_pred = np.zeros(shape=y_true.shape)
        self.y_pred_raw = np.zeros(shape=y_true.shape)
        self.y_pred_raw_average = np.zeros(shape=y_true.shape)
        self.eval = eval_strat
        self.device = dev

        # Define thresholds
        self.thresholds_s1 = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24]
        self.thresholds_s2 = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    def report_metrics(self, threshold):
        for average_strategy in ["micro", "macro"]:
            print("{} average strategy, threshold {}".format(average_strategy, threshold))
            print("precision:\t{}".format(precision_score(self.y_true, self.y_pred, average=average_strategy)))
            print("recall:\t{}".format(recall_score(self.y_true, self.y_pred, average=average_strategy)))
            print("f1:\t{}".format(f1_score(self.y_true, self.y_pred, average=average_strategy)))

    def _compute_prediction_sum(self, predictions):
        prediction_sum = np.zeros(IRMAS_N_CLASSES)
        for prediction in predictions:
            prediction_sum += prediction
        if self.eval == "s1":    # simple averaging strategy
            prediction_sum /= predictions.shape[0]
        return prediction_sum

    def compute_prediction_scores_raw(self, model, dataloader):

        # Progress bar
        printProgressBar(0, len(dataloader), prefix = 'Progress:', suffix = 'Complete', length = 50)
        names = set()

        # Go through all of the data
        for i, (name, x, y) in enumerate(dataloader):

            # Update progress bar
            printProgressBar(i + 1, len(dataloader), prefix = 'Progress:', suffix = 'Complete', length = 50)

            # Feed data into model and get output
            x = x.squeeze(0).float().to(self.device)
            one_excerpt_prediction  = model(x).cpu().detach().numpy()

            # Compute sum depending on evaluation strategy
            if self.eval == "s2":
                self.y_pred_raw[i, :] = self._compute_prediction_sum(one_excerpt_prediction)
            else:
                self.y_pred_raw_average[i, :] = self._compute_prediction_sum(one_excerpt_prediction)

    def evaluate(self, model, dataloader):

        # Compute predictions for all data
        self.compute_prediction_scores_raw(model, dataloader)

        # Evaluate predictions depending on evaluation strategy
        if self.eval == "s2":
            for threshold in self.thresholds_s2:
                self.y_pred = np.copy(self.y_pred_raw)
                for i in range(self.y_pred.shape[0]):
                    #if (self.y_pred[i, :].max() == 0):
                        #print(self.y_pred[i, :])
                    self.y_pred[i, :] /= self.y_pred[i, :].max()
                self.y_pred[self.y_pred >= threshold] = 1
                self.y_pred[self.y_pred < threshold] = 0
                self.report_metrics(threshold)
        else:
            for threshold in self.thresholds_s1:
                self.y_pred = np.copy(self.y_pred_raw_average)
                self.y_pred[self.y_pred < threshold] = 0
                self.y_pred[self.y_pred > threshold] = 1
                self.report_metrics(threshold)
    



def main():

    dev = getDevice()

    aparser = argparse.ArgumentParser()
    aparser.add_argument("-m",
                         action="store",
                         dest="model",
                         help="-m model to evaluate")
    aparser.add_argument("-s",
                         action="store",
                         dest="evaluation_strategy",
                         default="s2",
                         help="-s evaluation strategy: `s1` (simple averaging and thresholding) or `s2` ("
                              "summarization, normalization by max probability and thresholding)")

    args = aparser.parse_args()

    # Load data
    print("Loading metadata")
    dataset = pd.read_csv(IRMAS_TESTING_META_PATH, names=["filename", "class_id"])

    # Create datasets
    print("Creating dataset")
    dataset_test = InstrTagDatasetTest(dataset, args.model)

    # Create dataloaders
    print("Creating dataloader")
    dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=2, persistent_workers=True, shuffle=True)

    # Create model
    model = SingleLayer()
    model.to(dev)

    print('Loading model')
    checkpoint = torch.load(LOAD_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    evaluator = Evaluator(dataset_test.get_y_true(), dev, args.evaluation_strategy)
    evaluator.evaluate(model, dataloader_test)

if __name__ == "__main__":
    main()