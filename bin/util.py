import numpy as np
from sklearn import preprocessing
import mialab.utilities.pipeline_utilities as putil
import mialab.data.structure as structure
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

feature_key = ['x', 'y', 'z', 'T1 intensity', 'T2 intensity', 'T1 grad', 'T2 grad','x^2', 'xy', 'y^2', 'yz', 'z^2', 'xz']
class_key = ['background', 'white matter', 'grey matter', 'hippocampus', 'amygdala', 'thalamus']

def print_feature_importance(coefficients):
    # for each classifier print the corresponding feature importance
    if coefficients.ndim == 1:
        ranking = np.argsort(abs(coefficients))
    else:
        ranking = np.argsort(abs(coefficients), axis=1)
    ranking = np.flip(ranking)
    print('Importance of features (important -> unimportant)')
    if coefficients.ndim == 1:
        print([feature_key[j] for j in ranking])
    else:
        for i, cls in enumerate(class_key):
            print(cls, ":")
            print([feature_key[j] for j in ranking[i, :]])

def plot_feature_importance(coefficients, result_dir: str):
    if coefficients.ndim == 1:
        idx = np.argsort(abs(coefficients))
        coefficients = np.sort(abs(coefficients))
        xx = np.arange(0, len(coefficients), 1)
    else:
        idx = np.argsort(abs(coefficients), axis=1)
        coefficients = np.sort(abs(coefficients), axis=1)
        xx = np.arange(0, len(coefficients[0, :]), 1)

    idx = np.flip(idx)
    coefficients = np.flip(coefficients)

    plt.interactive(False)

    if coefficients.ndim == 1:
        f1 = plt.figure()
        plt.bar(xx, coefficients)
        labels = []
        for j in range(len(xx)):
            labels.append(feature_key[idx[j]])
        plt.xticks(xx, tuple(labels), rotation='vertical')
        plt.ylabel('absolute value of coefficient')
        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(os.path.join(result_dir, 'FeatureImportance'))

    else:
        for i, cls in enumerate(class_key):
            f1 = plt.figure()
            plt.bar(xx, coefficients[i, :])
            plt.title(cls)
            labels = []
            for j in range(len(xx)):
                labels.append(feature_key[idx[i, j]])
            plt.xticks(xx, tuple(labels), rotation='vertical')
            plt.ylabel('absolute value of coefficient')
            # Tweak spacing to prevent clipping of tick-labels
            plt.subplots_adjust(bottom=0.2)
            plt.savefig(os.path.join(result_dir, 'FeatureImportance' + cls))



def scale_features(feature_matrix, scaler=None):
    # scale each feature to zero mean and unit variance
    # scale features before training
    if scaler is None:
        scaler = preprocessing.StandardScaler().fit(feature_matrix)
    scaled_feature_matrix = scaler.transform(feature_matrix)
    return scaled_feature_matrix, scaler


def print_class_count(labels):
    # count the number of samples in each class
    classes, count = np.unique(labels, return_counts=True)
    print('Number of Samples in Class')
    for i, cls in enumerate(class_key):
        print(cls, ": ", count[i])


def compute_label_dist(images: sitk.Image, label: putil.LabelImageTypes) -> sitk.Image:
    # compute the sum over all images of a single label to asses the distribution in coordination features
    x, y, z = images[0].images[structure.BrainImageTypes.GroundTruth].GetSize()
    ground_truth_sum = np.zeros((x, y, z))
    for img in images:
        # get ground truth from image
        ground_truth = sitk.GetArrayFromImage(img.images[structure.BrainImageTypes.GroundTruth])
        # set all labels other than Amygdala to 0
        ground_truth[ground_truth != label] = 0
        # sum up over all images
        ground_truth_sum = ground_truth_sum + ground_truth

    ground_truth_sum/ label.value
    img_out = sitk.GetImageFromArray(ground_truth_sum)
    img_out.CopyInformation(images[0].images[structure.BrainImageTypes.GroundTruth])
    return img_out