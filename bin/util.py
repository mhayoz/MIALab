import numpy as np
from sklearn import preprocessing

feature_key = ['x', 'y', 'z', 'T1 intensity', 'T2 intensity', 'T1 grad', 'T2 grad']
class_key = ['background', 'white matter', 'grey matter', 'hippocampus', 'amygdala', 'thalamus']

def print_feature_importance(coefficients):
    # for each classifier print the corresponding feature importance

    ranking = np.argsort(abs(coefficients), axis=1)
    ranking = np.flip(ranking)
    print('Importance of features (important -> unimportant)')
    for i, cls in enumerate(class_key):
        print(cls, ":")
        print([feature_key[j] for j in ranking[i, :]])


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
