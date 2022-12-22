import logging
import os
import sys

from configparser import ConfigParser
from argparse import ArgumentParser

from analysis_baseline import analysis_baseline
from generate_decoder import generate_decoder
from analysis_spacenet import analysis_spacenet

import random

def parse_baseline(config):
    ''''
    Function to parse parameters for the baseline simulation
    '''
    ## Parse parameters that are common to all models
    cv_strategy = config.get('general', 'cv_strategy')
    datadir = config.get('general', 'datadir')
    debug = config.getboolean('setup', 'debug')
    if debug:
        logging.info('\nDEBUG SESSION')

    models, params_models = [], []

    if 'logistic' in config.sections():
        logging.info('Parsing parameters for Logistic Classification')
        models.append('logistic')

        params_logistic = {}
        for str_ in ['penalty', 'max_iter', 'C', 'multi_class', 'solver']:
            params_logistic[str_] = eval(config.get('logistic', str_))
        params_models.append(params_logistic)
        logging.info('Done')
    
    if 'ridge' in config.sections():
        logging.info('Parsing parameters for Ridge Classification')
        models.append('ridge')

        params_ridge = {} 
        for str_ in ['max_iter', 'alpha', 'solver']:
            params_ridge[str_] = eval(config.get('ridge', str_))
        params_models.append(params_ridge)
        logging.info('Done')

    if 'linearsvc' in config.sections():
        logging.info('Parsing parameters for Linear SVC')
        models.append('linearsvc')

        params_linearsvc = {}
        for str_ in ['penalty', 'max_iter', 'loss', 'C', 'multi_class']:
            params_linearsvc[str_] = eval(config.get('linearsvc', str_))
        params_models.append(params_linearsvc)
        logging.info('Done')

    return {
        'cv_strategy': cv_strategy,
        'datadir': datadir,
        'debug': debug,
        'models': models,
        'params_models': params_models
    }

def parse_decoder(config):
    ''''
    Function to parse parameters for the decoder simulation
    '''
    ## Parse parameters that are common to all models
    cv_strategy = config.get('general', 'cv_strategy')
    datadir = config.get('general', 'datadir')
    debug = config.getboolean('setup', 'debug')

    decoders, params_decoder = [], []
    if debug:
        logging.info('\nDEBUG SESSION')

    if 'logistic' in config.sections():
        logging.info('Parsing parameters for the Logistic decoder')
        decoders.append('logistic')

        params_logistic = {}
        for str_ in ['penalty', 'max_iter', 'C', 'multi_class', 'solver']:
            params_logistic[str_] = eval(config.get('logistic', str_))
        
        params_decoder.append(params_logistic)

    
    if 'linearsvc' in config.sections():
        logging.info('Parsing parameters for the L-SVC decoder')
        decoders.append('svc') # Notice the difference with the above func

        params_linearsvc = {}
        for str_ in ['penalty', 'max_iter', 'loss', 'C', 'multi_class']:
            params_linearsvc[str_] = eval(config.get('linearsvc', str_))
        params_decoder.append(params_linearsvc)


    return {
        'cv_strategy': cv_strategy,
        'datadir': datadir,
        'debug': debug,
        'decoders': decoders,
        'params_decoders': params_decoder
    }

def parse_spacenet(config):
    ''''
    Function to parse parameters for the spacenent simulation
    '''
    ## Parse parameters that are common to all models
    cv_strategy = config.get('general', 'cv_strategy')
    datadir = config.get('general', 'datadir')
    debug = config.getboolean('setup', 'debug')
    if debug:
        logging.info('\nDEBUG SESSION')

    if 'model' in config.sections():
        logging.info('Parsing parameters for Spacenet')
        penalty = config.get('model', 'penalty')
        param_spacenet = {}
        for str_ in ['alphas', 'max_iter']:
            param_spacenet[str_] = eval(config.get('model', str_))

    return {
        'cv_strategy': cv_strategy,
        'datadir': datadir,
        'debug': debug,
        'penalty': penalty,
        'param_spacenet': param_spacenet
    }
    


def configure_logging(logdir, config):
    '''
    Function to configure the logging of the simulation
    Opens a directory where the logs and simulation-specific models are saved
    '''
    os.makedirs(logdir, exist_ok=True)

    f = os.path.join(logdir, 'config.ini')
    with open(f, 'w') as file:
        config.write(file)

    ### configure logging
    log_filename = os.path.join(logdir, 'logs.log')
    logging.basicConfig(format='%(message)s',
						filename=log_filename,
						filemode='w', # Allows to overwrite and not append to an old log
						level=logging.INFO)
    logging.info('Setting up the simulation')

if __name__ == "__main__":
    # Reproducability
    random.seed(42)
    # Configuration of parsing

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline_per_run.ini')
    args = parser.parse_args()
    config = ConfigParser()
    config.read(args.config)
    
    # suffix for the log directory
    suffix = '_CV_' + config.get('general', 'cv_strategy')
    if config.get('setup', 'type') == 'baseline':
        logdir = os.path.join(config.get('setup', 'logdir'), 'baseline' + suffix)
        configure_logging(logdir, config)
        logging.info('Baseline simulation type')
        simulation_parameters = parse_baseline(config)        
        ## Run baseline
        analysis_baseline(**simulation_parameters)

    elif config.get('setup', 'type') == 'decoder':
        logdir = os.path.join(config.get('setup', 'logdir'), 'decoder' + suffix)
        configure_logging(logdir, config)
        logging.info('Decoder simulation type')
        simulation_parameters = parse_decoder(config)
        ## Run decoder
        simulation_parameters['saving_dir'] = logdir
        generate_decoder(**simulation_parameters)

    elif config.get('setup', 'type') == 'spacenet':
        logdir = os.path.join(config.get('setup', 'logdir'), 'spacenet' + suffix + '_' + config.get('model', 'penalty'))
        configure_logging(logdir, config)
        logging.info('Spacenet simulation type')
        simulation_parameters = parse_spacenet(config)
        ## Run decoder
        simulation_parameters['saving_dir'] = logdir
        analysis_spacenet(**simulation_parameters)
    else:
        logging.info('Simulation type not available')
        sys.exit(-1)

    logging.info('\nDONE')

