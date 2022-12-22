import nibabel as nib
import numpy as np
from nibabel.funcs import concat_images
from nilearn.masking import compute_brain_mask, apply_mask

import os

from nilearn.image import clean_img

class Dataset():
    '''
    Class dataset to store the data during a simulation

    Variables
    ---------
    beta_maps (nib.Nift1Image): Images used as inputs of the models
    mask (nib.Nifti1Image): Mask associated to the images
    labels (list): associated labels
    nb_subs_ (int): number of subjects in the dataset
    nb_runs_per_subs (int): number of runs per subject


    Methods
    -------
    split_train_val(self, train_idx, validation_idx):
        returns train and validation data specified by the input indices.
        Useful for spacenet
    get_beta_maps(self):
        returns the beta_maps
    def get_samples(self):
        returns the samples, i.e. the flattened and masked beta_maps
    def get_labels(self):
        returns the labels
    
    '''
    def __init__(self, directory, debug=False):
        '''
        Initialization of an instance of the class
        If debug==True only 5 subjects are kept

        Inputs:
            directory: string specifying where the data is stored
            debug: bool to specify debug mode
        '''
        beta_maps_dir = os.path.join(directory, 'beta_maps/')
        mask_dir = os.path.join(directory, 'anatomy/')
        self.mask = compute_brain_mask(nib.load(os.path.join(mask_dir, 'mask.nii')), threshold=.3)

        self.beta_maps = []
        for file in sorted(os.listdir(beta_maps_dir)):
            if file.endswith('.nii.gz'):
                map = nib.load(os.path.join(beta_maps_dir, file))
                self.beta_maps.append(clean_img(map, standardize=False, ensure_finite=True))
        
        self.nb_subs_ = len(self.beta_maps)
        classes = ['caught', 'chase', 'checkpoint', 'close_enemy', 'protected_by_wall', 'vc_hit']
        self.nb_runs_per_sub_ = self.beta_maps[0].shape[-1] // len(classes)

        if debug:
            self.nb_subs_ = 5
            self.beta_maps = self.beta_maps[:self.nb_subs_]

        self.beta_maps = concat_images(self.beta_maps, axis=-1)
        self.labels = np.tile(classes, self.nb_runs_per_sub_*self.nb_subs_)

    
    def split_train_val(self, train_idx, validation_idx):
        '''
        Returns train and validation data specified by the input indices.
        Useful for spacenet
        
        Inputs:
            train_idx: list of idx to be put in the train fold
            validation_idx: list of idx to be put in the validation fold
        '''
        raw_data, affine = self.beta_maps.get_fdata(), self.beta_maps.affine
        train_raw_data, val_raw_data = raw_data[..., train_idx], raw_data[..., validation_idx]
        train_data, val_data = nib.Nifti1Image(train_raw_data, affine), nib.Nifti1Image(val_raw_data, affine)
        return train_data, val_data, self.labels[train_idx], self.labels[validation_idx]

    def get_beta_maps(self):
        '''
        Returns the beta maps of the dataset.
        '''
        return self.beta_maps
        
    def get_samples(self):
        '''
        Returns the samples of the dataset.
        Samples are the masked and flattened beta maps.
        '''
        samples = apply_mask(self.beta_maps, self.mask)
        return samples
    
    def get_labels(self):
        '''
        Returns the labels of the dataset.
        '''
        return self.labels

