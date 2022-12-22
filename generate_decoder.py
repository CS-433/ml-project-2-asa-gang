import pickle
import os
import sys
import logging

from nilearn.decoding import Decoder
from sklearn.model_selection import LeaveOneGroupOut

from dataset import Dataset

from helpers import make_chunks_per_run, make_chunks_per_subjects

def generate_decoder(datadir, cv_strategy, decoders, params_decoders, saving_dir, debug=False):
    '''
    !!! This function is different than the baseline
    It traines only ONE set of parameters per decoder!!!

    Function to train a decoder using a set of parameters and log the results in the logging file specified in run.py
    The dataset is first loaded in debug specified mode.
    Then the CV strategy is specified !!! random not supported !!!
    To finally train the decoder using param_decoder as a set of parameters,
    using the CV with internal functions

    Inputs:
        datadir: string being the relative path of where the data is stored
        cv_strategy: string to specify the cv_strategy
        decoder: string to specify the decoder to use
        param_decoder: dict of the associated parameters for the decoder
        saving_dir: string being the relative path where the trained decoder is saved
        debug: bool to specify debug mode
    '''

    ## Loading data
    dataset = Dataset(datadir, debug)

    ## Setting up CV strategy
    if cv_strategy == 'per_run':
        cv = LeaveOneGroupOut()
        chunks = make_chunks_per_run(dataset.nb_subs_, dataset.nb_runs_per_sub_)
    elif cv_strategy == 'per_subs':
        cv = LeaveOneGroupOut()
        chunks = make_chunks_per_subjects(dataset.nb_subs_)
    else:
        logging.info('ERROR, {} cv not implemented for this method'.format(cv_strategy))
        sys.exit(-1)
    
    for decoderstr, param_decoder in zip(decoders, params_decoders):
        logging.info('Starting to fit {} decoder...'.format(decoderstr))
        decoder = Decoder(
            estimator=decoderstr,
            mask=dataset.mask,
            cv=cv,
            param_grid=param_decoder,
            n_jobs=-1,
            verbose=1,
            standardize=True
        )

        decoder.fit(dataset.get_beta_maps(), dataset.get_labels(), groups = chunks)
        
        ## Here save the decoder in saving_dir
        logging.info('\n------------------------')
        logging.info('Decoder fitted with scores')
        logging.info(decoder.cv_scores_)
        logging.info('------------------------')
        
        name = 'train_decoder_{}.sav'.format(decoderstr)
        pickle.dump(decoder, open(os.path.join(saving_dir, name), 'wb'))
        logging.info('Fitted decoder saved')