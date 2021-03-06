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
import numpy as np
import pymia.data.conversion as conversion
import pymia.data.loading as load
import util

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
    # precomputed by preprocessAndStore.py
    file_id = open('data_train.pckl', 'rb')
    data_train = pickle.load(file_id)
    file_id.close()

    file_id = open('labels_train.pckl', 'rb')
    labels_train = pickle.load(file_id)
    file_id.close()


    ##########################################
    # use if GridSearchCV is used
    # perform a grid search over the parameter grid and choose the optimal parameters
    #Cs = [10, 12, 15, 20]#a list best = 15
    #gammas = [1 ,2 ,3, 5,10]#a list best = 10
    #param_grid = {'C': Cs, 'gamma': gammas}#a dictionary
    #svm_rbf_classifier = model_selection.GridSearchCV(svm.SVC(kernel='rbf'), param_grid, verbose=1)

    data_train_scaled, scaler = util.scale_features(data_train)



    # printing out how much labels of each group were taken by the mask
    util.print_class_count(labels_train)


    # use if GridSearchCV is not used
    svm_rbf_classifier = svm.SVC(kernel= 'rbf', C=15, gamma= 10 ,class_weight='balanced', decision_function_shape='ovo')


    start_time = timeit.default_timer()

    print("start training")
    # for position features only: svm_rbf_classifier.fit(data_train_scaled[:, 0:3], labels_train)
    svm_rbf_classifier.fit(data_train_scaled, labels_train)

    #####svm_rbf_classifier.coef_ can not be used with rbf kernel
    #util.print_feature_importance(svm_rbf_classifier.best_estimator_.coef_)


    #use if GridSearchCV is used
    #print("importance of features: ", svm_rbf_classifier.best_estimator_.coef_)#####svm_rbf_classifier.coef_ can not be used with rbf kernel
    #print("best estimator: ", svm_rbf_classifier.best_estimator_)
    #print("best parameter: ", svm_rbf_classifier.best_params_)

    #use if GridSearchCV is not used
    print("best estimator: ", svm_rbf_classifier)
    print("estimator dual_coef_: ", svm_rbf_classifier.dual_coef_)



    file_id = open('svm_rbf_fullset_C15_G5.pckl', 'wb')
    pickle.dump(svm_rbf_classifier, file_id)
    file_id.close()
    file_id = open('scaler_rbf_fullset_C15_G5.pckl', 'wb')
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
