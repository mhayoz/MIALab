"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import os
import sys

import pickle
import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import numpy as np
import pymia.data.loading as load
import matplotlib as plt

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


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction

    """

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    print('-' * 5, 'Training...')


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
                          'label_percentages': [0.0005, 0.005, 0.005, 0.05, 0.09, 0.022]}#[0.0003, 0.004, 0.003, 0.04, 0.04, 0.02]}


    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

    # compute sum over all ground truth images
    Label = putil.LabelImageTypes.AMYGDALA
    img_summed = util.compute_label_dist(images, Label)
    # save results
    sitk.WriteImage(img_summed, os.path.join(result_dir, 'groundtruth_sum_' + Label.name + '.mha'), True)


    # store preprocessed images to file
    file_id = open('data_train_reduced2.pckl', 'wb')
    pickle.dump(data_train, file_id)
    file_id.close()
    file_id = open('labels_train_reduced.pckl', 'wb')
    pickle.dump(labels_train, file_id)
    file_id.close()
    print('-' * 5, 'Preprocessed images stored')

    # n, m = img_summed.shape
    # x = np.arange(0, n-1, 1)
    # y = np.arange(0, m-1, 1)
    # meshgrid = np.meshgrid(x,y, sparse=False)
    # plt.pyplot.contour(meshgrid, img_summed)
    #printing out how much labels of each group were taken by the mask
    util.print_class_count(labels_train)

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
