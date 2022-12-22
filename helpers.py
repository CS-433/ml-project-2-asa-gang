import numpy as np

###############################
### CV helpers
###############################

def make_chunks_per_subjects(n_subs, n_maps_per_subs = 30, n = 7):
    '''
    Helper function to make chunks for cross validation.
    Puts n subject data in one chunk
    Left to the reader to see if the chunks will have and equal number of subs in them
    Ideally one wants n_subs %n == 0

    returns a np.array containing the chunks
    ex: [0, 1, 3, 4, 2, 0, 5, ..., 0, 1]

    Inputs:
        n_subs: int specifying the number of subjects in the dataset
        n_maps_per_subs: int specifying how many maps are there per subs
        n: how many subs in one chunk
    '''
    chunks = np.ravel([[i]*n_maps_per_subs for i in range(n_subs)])
    chunks = chunks % n
    return chunks

# --------------------------

def make_chunks_per_run(n_subs, n_runs_per_sub) :
    '''
    Helper function to make chunks for cross validation.
    Puts one run across all subject in one chunk

    returns a np.array containing the chunks
    ex: [0, 1, 3, 4, 2, 0, 5, ..., 0, 1]
    8
    Inputs:
        n_subs: int specifying the number of subjects in the dataset
        n_maps_per_subs: int specifying how many maps are there per subs
    '''
    chunks = []
    for _ in range(n_subs): 
        for i in range(n_runs_per_sub): # Because 5 runs
            chunks.append(i*np.ones((6,), dtype=int)) # Because 6 conditions
    chunks = np.concatenate(chunks)
    return chunks

# --------------------------

def make_folds_from_chunks(chunks):
    '''
    Helper function to transforms chunks into actual fold idxs
    Helpful for the spacenet part, as the CV was written by hand

    return a list of (train_idx, val_idx) per fold
    Inputs:
        chunks: np.array specifying the chunk a sample belongs to
    '''
    idx = np.arange(len(chunks))

    folds = []
    for i in np.unique(chunks):
        train_idx = idx[chunks != i]
        validation_idx = idx[chunks == i]
        folds.append((train_idx, validation_idx))
    return folds