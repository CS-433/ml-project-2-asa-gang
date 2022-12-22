import pickle
import os
import sys
import logging

from nilearn.decoding import SpaceNetClassifier
import numpy as np

from dataset import Dataset

from helpers import make_chunks_per_run, make_chunks_per_subjects, make_folds_from_chunks

def _spacenet_custom_CV(decoder, dataset, folds):
    '''
    Internal function to perform cross validation on the spacenent classifier.
    The internal CV system of Spacenet differs from decoder, and is not adequate.

    returns a dict with a list of the trained decoders and a list with the associated accuracies

    Inputs:
        decoder: Spacenet classifier to fit
        dataset: dataset to fit the decoder to
        folds: list of tuple of two np.array containing idx for train val for each folds
    '''
    accuracies = []
    for train_idx, validation_idx in folds:
        X_train, X_val, y_train, y_val = dataset.split_train_val(train_idx, validation_idx)
        decoder.fit(X_train, y_train)
        y_pred = decoder.predict(X_val)
        accuracies.append(np.mean(y_pred == y_val))

    print(accuracies)
    
    return {
        'trained_decoder': decoder,
        'accuracies': accuracies
    }


def analysis_spacenet(datadir, cv_strategy, penalty, params_spacenet, saving_dir, debug=False):
    '''
    !!! This function supports only varying the alpha parameter !!!!

    Function to train spacenet classifiers with a given penalty and varying alpha
    and log the results in the logging file specified in run.py
    The dataset is first loaded in debug specified mode.
    Then the CV strategy is specified !!! random not supported !!!
    To finally train the decoders using param_spacenent as a set of parameters,
    using the custom CV (Because the internal one is not adequate)

    Inputs:
        datadir: string being the relative path of where the data is stored
        cv_strategy: string to specify the cv_strategy
        penalty: string to specify the penalty to use
        params_spacenet: list of dict of the associated parameters for the decoder
        saving_dir: string being the relative path where the trained decoder is saved
        debug: bool to specify debug mode
    '''

    logging.info('Running spacenent with penalty {}'.format(penalty))

    ## Loading data and scaling it
    dataset = Dataset(datadir, debug)

    ## Setting up CV strategy
    if cv_strategy == 'per_run':
        chunks = make_chunks_per_run(dataset.nb_subs_, dataset.nb_runs_per_sub_)
    elif cv_strategy == 'per_subs':
        chunks = make_chunks_per_subjects(dataset.nb_subs_)
    else:
        logging.info('ERROR, {} cv not implemented for this method'.format(cv_strategy))
        sys.exit(-1)

    folds = make_folds_from_chunks(chunks)

    for alpha in params_spacenet['alphas']:
        decoder = SpaceNetClassifier(
            penalty=penalty,
            mask=dataset.mask,
            max_iter = params_spacenet['max_iter'][0],
            alphas=alpha,
            cv=1,
            n_jobs=1,
            standardize=True,
        )
        results = _spacenet_custom_CV(decoder, dataset, folds)
        logging.info('\n------------------------')
        logging.info('Scores for alpha= {}'.format(str(alpha)))
        logging.info(results['accuracies'])
        logging.info('------------------------')

        pickle.dump(results, open(os.path.join(saving_dir, 'trained_spacenet_{}.sav'.format(alpha)), 'wb'))