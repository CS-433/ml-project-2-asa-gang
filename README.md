# Decoding Brain States Triggered via Video Game with fMRI Based MVPA Approach
## Summary

This repository contains the code and the report for the second project of CS-433 Machine Learning 2022 course at EPFL. This project was conducted under the supervision of members of the [MIP:lab](https://miplab.epfl.ch/).\
The goal is to classify fMRI brain images of subjects playing a video game that triggers predefined emotional states specified as the classes. Using different classification methods and visualization of the weights of the classifiers, this project allows to see which parts of the brain are responsible for a given emotional state.

## File structure
```
├── configs
    ├── baseline_per_run.ini
    ├── baseline_per_subs.ini
    ├── decoder.ini
    ├── spacenet_graphnet.ini
    ├── spacenet_tvl1.ini
├── analysis_baseline.py
├── analysis_spacenet.py
├── dataset.py
├── generate_decoder.py
├── helpers.py
├── README.md
├── run.py
└── visualization.ipynb
```
## Implementation
`analysis_baseline.py` contains a function that sequentially trains baseline models and logs the results.

`analysis_spacenet.py` contains a function that sequentially trains [SpaceNet Classifiers](https://nilearn.github.io/dev/modules/generated/nilearn.decoding.SpaceNetClassifier.html#nilearn.decoding.SpaceNetClassifier) with different 'alphas', logs the results and saves the models in a specified directory.

`dataset.py` contains class to wrap the dataset that data-specific helpers as methods.

`generate_decoder.py` contains a function that trains two [Decoder](https://nilearn.github.io/dev/modules/generated/nilearn.decoding.Decoder.html) (Linear SVC and Logistic Classifier), logs the results and saves them in a specific directory. Note that the implementation allows the training with only ONE set of parameters per decoder.

`helpers.py` contains helper functions for the cross validation of the models.

`run.py` containes parser functions and initiates a simulation. More specifically, this file takes as input argument a config file that contains all the parameters of the simulation to run. See the directory `configs/`.

`visualization.ipynb` contains plotter functions to visualize the results.

## Libraries
[numpy](https://numpy.org/)\
[sklearn](https://scikit-learn.org/stable/)\
[nilearn](https://nilearn.github.io/stable/index.html)\
[nibabel](https://nipy.org/nibabel/)
## How to use
Example config files are available in the `configs/` directory.\
Simply run the following (as an example):
```
python run.py --config configs/baseline_per_run.ini
```
## Team Members

This project belongs to the team `ASA_GANG` with members:

- Alicia Milloz: [@Alicia-max](https://github.com/Alicia-max)
- Sevda Öğüt: [@ogutsevda](https://github.com/ogutsevda)
- Alexandre de Skowronski: [@alexdesko](https://github.com/alexdesko)

## Acknowledgments
Many thanks to [Mi Xue Tan](https://www.unige.ch/cisa/education/swiss-doctoral-school/members/students/mi-xue-tan/) for providing the data, feedback, and in general a great supervision.
