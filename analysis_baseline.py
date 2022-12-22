import sys
import logging

from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid, cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dataset import Dataset

from helpers import make_chunks_per_run, make_chunks_per_subjects

def _select_model(modelstr, params):
    '''
    Internal function to instanciate the model with its parameters for the baseline tests.
    Will throw an error if the parameters do not match the model.

    Returns the model instance with right parameters

    Inputs:
        modelstr: string
        params: dict
    '''
    if modelstr == 'logistic':
        return LogisticRegression(**params)
    elif modelstr == 'ridge':
        return RidgeClassifier(**params)
    elif modelstr == 'linearsvc':
        return LinearSVC(**params)
    else:
        logging.info('Model {} not implemented. Aborting.'.format(modelstr))
        sys.exit(-1)

def _analysis_baseline_per_model(X, y, model, cv, chunks):
    '''
    Internal function to perform the cross-validation specified by cv and chunks.
    Applied the cross validation to the model with X, y as samples, labels

    Returns the score of the sklearn function cross_vals_score

    Inputs:
        X: np.array of size (n_samples, n_features)
        y: np.array of size (n_samples,)
        model: sklearn classifier supported by cross_val_score
        cv: cross-validation generator supported by cross_val_score
        chunks: np.array of size (n_samples,) that contains groups for cross-validation
    '''    
    scores = cross_val_score(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        groups=chunks,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    return scores

def analysis_baseline(datadir, cv_strategy, models, params_models, debug=False):
    '''
    Function to perform the baseline analysis and log the results in the logging file specified in run.py
    The dataset is first loaded in debug specified mode.
    Then the CV strategy is specified.
    To finally loop throught the models and the parameters,
    perform CV on the data using internal functions and log the scores.

    Inputs:
        datadir: string being the relative path of where the data is stored
        cv_strategy: string to specify the cv_strategy
        models: list of strings of the models to test
        params_models: list of dict of the associated parameters for the models
        debug: bool to specify debug mode
    '''
    ## Loading data
    dataset = Dataset(datadir, debug)
    X, y = dataset.get_samples(), dataset.get_labels()
    
    ## Setting up CV strategy
    if cv_strategy == 'per_run':
        cv = LeaveOneGroupOut()
        chunks = make_chunks_per_run(dataset.nb_subs_, dataset.nb_runs_per_sub_)
    elif cv_strategy == 'per_subs':
        cv = LeaveOneGroupOut()
        chunks = make_chunks_per_subjects(dataset.nb_subs_)
    elif cv_strategy == 'random':
        cv = 5
        chunks = None
    else:
        logging.info('ERROR, {} cv not implemented for this method'.format(cv_strategy))
        sys.exit(-1)
    

    for modelstr, params in zip(models, params_models):
        param_grid = ParameterGrid(param_grid=params)
        for p in param_grid:
            logging.info('\n------------------------')
            logging.info('Scores for the model {}'.format(str(modelstr)))
            try:
                model = _select_model(modelstr, p)
                pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
                accuracies = _analysis_baseline_per_model(X, y, pipe, cv, chunks)
                logging.info('Accuracy: {} +/- {}'.format(accuracies.mean(), accuracies.std()))
            except:
                logging.info('ERROR, could not perform CV')
            logging.info('With parameters: {}'.format(p))
            logging.info('------------------------')
