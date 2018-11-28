"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit
import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import numpy as np
import pymia.data.conversion as conversion
import pymia.data.loading as load
from sklearn import svm, linear_model

sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))  # append the MIALab root directory to Python path
# fixes the ModuleNotFoundError when executing main.py in the console after code changes (e.g. git pull)
# somehow pip install does not keep track of packages

import mialab.data.structure as structure
import mialab.utilities.file_access_utilities as futil
import mialab.utilities.pipeline_utilities as putil
import util

IMAGE_KEYS = [structure.BrainImageTypes.T1,
              structure.BrainImageTypes.T2,
              structure.BrainImageTypes.GroundTruth]  # the list of images we will load


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str, ml_method: str, verbose: bool):
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

    print('-' * 5, 'Training '+ ml_method + '...')

    # crawl the training image directories
    crawler = load.FileSystemDataCrawler(data_train_dir,
                                         IMAGE_KEYS,
                                         futil.BrainImageFilePathGenerator(),
                                         futil.DataDirectoryFilter())
    pre_process_params = {'zscore_pre': True,
                          'registration_pre': False,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True,
                          'second_oder_coordinate_feature': False,
                          'label_percentages': [0.0005, 0.005, 0.005, 0.05, 0.09, 0.022]}

    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

    if verbose:
        util.print_class_count(labels_train)

    start_time = timeit.default_timer()
    if ml_method == 'random_forest':
        classifier = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1],
                                                    n_estimators=20,
                                                    max_depth=25)
        data_train_scaled = data_train # do not scale features to keep original RF
    elif ml_method == 'svm_linear':
        classifier = svm.LinearSVC(C=1, class_weight='balanced', dual=False)
        data_train_scaled, scaler = util.scale_features(data_train)
    elif ml_method == 'svm_rbf':
        classifier = svm.SVC(kernel='rbf', C=15, gamma=5, class_weight='balanced',
                                     decision_function_shape='ovo')
        data_train_scaled, scaler = util.scale_features(data_train)

    elif ml_method == 'logistic_regression':
        classifier = linear_model.LogisticRegression(class_weight='balanced')
        data_train_scaled, scaler = util.scale_features(data_train)
    else:
        assert False, "No valid segmentation algorithm selected in argument ml_method"

    classifier.fit(data_train_scaled, labels_train)
    print(' Time elapsed:', timeit.default_timer() - start_time, 's')

    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    # print and plot feature importance for each structure
    if verbose:
        if ml_method == 'svm_linear':
            util.print_feature_importance(classifier.coef_)
            util.plot_feature_importance(classifier.coef_, result_dir)
        if ml_method == 'random_forest':
            util.print_feature_importance(classifier.feature_importances_)
            util.plot_feature_importance(classifier.feature_importances_, result_dir)

    print('-' * 5, 'Testing...')

    # initialize evaluator
    evaluator = putil.init_evaluator(result_dir)

    # crawl the training image directories
    crawler = load.FileSystemDataCrawler(data_test_dir,
                                         IMAGE_KEYS,
                                         futil.BrainImageFilePathGenerator(),
                                         futil.DataDirectoryFilter())

    # load images for testing and pre-process
    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=True)

    images_prediction = []
    images_probabilities = []

    for img in images_test:
        print('-' * 10, 'Testing', img.id_)

        start_time = timeit.default_timer()
        if ml_method == 'random_forest':
            scaled_features = img.feature_matrix[0]
        else:
            scaled_features, s = util.scale_features(img.feature_matrix[0], scaler)


        predictions = classifier.predict(scaled_features)
        probabilities = classifier.predict_proba(scaled_features)
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # convert prediction and probabilities back to SimpleITK images
        image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                        img.image_properties)
        image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

        # evaluate segmentation without post-processing
        evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

        images_prediction.append(image_prediction)
        images_probabilities.append(image_probabilities)

    # post-process segmentation and evaluate with post-processing
    post_process_params = {'crf_post': False}
    images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
                                                     post_process_params, multi_process=True)

    for i, img in enumerate(images_test):
        evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
                           img.id_ + '-PP')

        # save results
        sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)
        sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha'), True)


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

    parser.add_argument(
        '--ml_method',
        type=str,
        default='random_forest',
        help='ML learning algorithm for segmentation'
    )

    parser.add_argument(
        '--verbose',
        type=bool,
        default=True,
        help='Enable prints and plots.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir, args.ml_method, args.verbose)
