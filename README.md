# copycat-detector

![love_island_new_roster](https://github.com/cptanalatriste/copycat-detector/blob/master/notebook_ims/plag_example.png?raw=true)

A Naive-Bayes classifier for detecting plagiarism, trained over a dataset of short answers [developed by Clough and Stevenson](https://link.springer.com/article/10.1007/s10579-009-9112-1).

## Getting started
To train the classifier, be sure to do the following first:

1. Clone this repository.
2. Download a [modified version of the dataset](https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c4147f9_data/data.zip). 
3. Place the dataset files in your cloned copy of the repository.
4. Make sure you have installed all the Python packages defined in `requirements.txt`.

## Instructions
The feature engineering steps are defined in the `2_Plagiarism_Feature_Engineering.ipynb` jupyter notebook.
Most of the code is contained in the `copycat_detector` module.

For training, notebook `3_Training_a_Model.ipynb` was run on an Amazon SageMaker instance.
