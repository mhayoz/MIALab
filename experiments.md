# Experiments

You will form groups of 2-3 people and select one topic to investigate and experiment.

## Pre-processing

Investigate the influence of pre-processing on the segmentation performance.

- Image normalization
- Histogram matching
- Skull stripping (separate the brain from the skull and other surrounding structures)

###### Materials

- `mialab.filtering.preprocessing`, e.g. `SkullStrip`
- L. G. Nyúl, J. K. Udupa, and X. Zhang, "New variants of a method of MRI scale standardization.", IEEE Trans. Med. Imaging, vol. 19, no. 2, pp. 143–50, Feb. 2000.
- J.-P. Bergeest and F. Jäger, "A Comparison of Five Methods for Signal Intensity Standardization in MRI", in Bildverarbeitung für die Medizin 2008, Berlin Heidelberg: Springer, 2008, pp. 36–40.
- http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing

## Registration

Investigate the registration.

- Metric type
- Optimizer type

###### Materials

- `mialab.filtering.registration`, e.g. use the `RegistrationPlotter`
- P. Viola and W. M. I. Wells, "Alignment by maximization of mutual information", Proc. IEEE Int. Conf. Comput. Vis., vol. 24, no. 2, pp. 16–23, 1995.
- P. Cattin and V. Roth, Biomedical Image Analysis, 2016. [Online]. Available: https://miac.unibas.ch/BIA/ [Accessed: 25-Aug-2017].
- [SimpleITK Notebooks](http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/): See chapters 60-67
- [ITK Software Guide, Book 2](https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch3.html): In C++ but with a thorough description

## Post-processing

Can we leverage the segmentation performance by post-processing?

- Morphological operators
- Dense conditional random field (CRF)
- Manual user interaction (e.g., brushing)

###### Materials

- `mialab.filtering.postprocessing`, e.g. use `DenseCRF`
- https://arxiv.org/abs/1210.5644
- P. Cattin and V. Roth, Biomedical Image Analysis - Mathematical Morphology, 2013. [Online]. Available: http://informatik.unibas.ch/fileadmin/Lectures/FS2013/CS252/morphology13.pdf [Accessed: 25-Aug-2017]

## Evaluation

Which metrics are suitable for our task? Can we change the validation procedure?

- Metric types
- Influence of e.g. small structures

###### Materials

- A. A. Taha and A. Hanbury, "Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool", BMC Med. Imaging, vol. 15, no. 1, pp. 1–28, 2015.

## Decision Forest Parameter Tuning

Can we improve the segmentation performance by parameter tuning? What is the influence of each parameter?

- Overfitting
- Tree depth
- Forest size

###### Materials

- A. Criminisi and J. Shotton, Decision Forests for Computer Vision and Medical Image Analysis, 1st ed. London: Springer, 2013.

## Machine Learning Algorithms

Do other machine learning algorithms perform better on our task?

- Support Vector Machine (SVM)
- Variants of decision forests (e.g., gradient boosted trees)

###### Materials

- [scikit-learn](http://scikit-learn.org/stable/supervised_learning.html#supervised-learning): Supervised machine learning algorithms
- ​https://arxiv.org/abs/1708.05070

## Feature Engineering

What features could be used to improve our model?

- Investigate other features (e.g., histogram of oriented gradients (HOGs))
- Hemisphere feature

## Feature Selection

Can we reduce the number of features to decrease the model complexity and the computational burden.

- Decision forest feature importance
- Principal component analysis (PCA)
- Mutual information based feature selection

###### Materials

- `mialab.feature_selection.mutual_information`
- [scikit-learn](http://scikit-learn.org/stable/modules/decomposition.html#decompositions): Dimensionality reduction
- http://ieeexplore.ieee.org/document/1453511/