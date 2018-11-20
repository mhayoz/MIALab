"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit

import pickle
import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
from sklearn import svm
from sklearn import model_selection
from sklearn import preprocessing
import numpy as np
import pymia.data.conversion as conversion
import pymia.data.loading as load
import util

import matplotlib.pyplot as plt
import scipy

sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))  # append the MIALab root directory to Python path
# fixes the ModuleNotFoundError when executing main.py in the console after code changes (e.g. git pull)
# somehow pip install does not keep track of packages

import mialab.data.structure as structure
import mialab.utilities.file_access_utilities as futil
import mialab.utilities.pipeline_utilities as putil

IMAGE_KEYS = [structure.BrainImageTypes.T1,
              structure.BrainImageTypes.T2,
              structure.BrainImageTypes.GroundTruth]  # the list of images we will load


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    print('-' * 5, 'Training...')

    # load feature matrix and label vector
    # precomputed by preprocessAndStore.py1057278
    file_id = open('data_train_reduced.pckl', 'rb')
    data_train = pickle.load(file_id)
    file_id.close()

    file_id = open('labels_train_reduced.pckl', 'rb')
    labels_train = pickle.load(file_id)
    file_id.close()


    ##########################################


    # perform a grid search over the parameter grid and choose the optimal parameters
    param_grid = {'C': [ 2, 3, 4, 5, 10, 20, 100]}  # grid to search for best parameter C = 0.02
    #svm_classifier = model_selection.GridSearchCV(svm.LinearSVC(C=1, class_weight='balanced', dual=False), param_grid, verbose=1)

    data_train_scaled, scaler = util.scale_features(data_train)

    util.print_class_count(labels_train)

    x = np.arange(1,10,1)
    plt.plot(x, x)

    # use balanced class weights to include classes with small sample size
    # solve the primal problem since n_features < n_samples

    svm_classifier = svm.LinearSVC(C=1, class_weight='balanced', dual=False)  # probability=False, kernel= 'rbf') #kernel='linear')
    start_time = timeit.default_timer()

    svm_classifier.fit(data_train_scaled, labels_train)

    #util.print_feature_importance(svm_classifier.coef_)
    util.plot_feature_importance(svm_classifier.coef_)

    #print(svm_classifier.best_params_)
    #print(svm_classifier.best_estimator_)

    # store trained SVM
    file_id = open('svm_linear.pckl', 'wb')
    pickle.dump(svm_classifier, file_id)
    file_id.close()
    file_id = open('scaler.pckl', 'wb')
    pickle.dump(scaler, file_id)
    file_id.close()

    print(' Time elapsed:', timeit.default_timer() - start_time, 's')



if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, '../data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir)
