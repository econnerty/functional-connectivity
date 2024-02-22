#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mne
import concurrent.futures
from mne.coreg import Coregistration
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
import numpy as np
from pathlib import Path
import pandas as pd
import os
from pathlib import Path
import xarray as xr
import time
import gc
import dynsys_orig as dynsys


# In[2]:


def filter_labels_with_vertices(labels_parc, src):
    # Get the vertices from both hemispheres in the source space
    src_vertices = [src[0]['vertno'], src[1]['vertno']]
    
    # Initialize an empty list to hold valid labels
    valid_labels = []
    
    for label in labels_parc:
        # Determine the hemisphere index: 0 for 'lh' and 1 for 'rh'
        hemi_idx = 0 if label.hemi == 'lh' else 1
        
        # Check if any of the label's vertices are in the source space for that hemisphere
        if any(v in src_vertices[hemi_idx] for v in label.vertices):
            valid_labels.append(label)
            
    return valid_labels


# In[6]:


subjects = [subject for subject in os.listdir('/work/erikc/inspected') if subject.startswith('sub')]
subjects_dir = "/work/erikc/inspected"
#subjects = ['sub-032304']
#subjects_dir = '~/Code/functional_connectivity/'
conditions = ['EC','EO']
#eeg_dir = '~/Code/functional_connectivity/eeg'
eeg_dir = '/work/erikc/eeg'


# In[36]:


start_time = time.time()
def process_subject(subject, subjects_dir, eeg_dir, conditions):
    for condition in conditions:
        try:
            src = mne.setup_source_space(subject, add_dist="patch", subjects_dir=subjects_dir)
            raw = mne.io.read_raw_eeglab(f"{eeg_dir}/{subject}/{subject}_{condition}.set")

            #Filter to alpha band
            raw.load_data()
            raw.filter(8,12)
            
            info = raw.info
            fiducials = "estimated"
            coreg = Coregistration(info, subject, subjects_dir, fiducials=fiducials)
            
            conductivity = (0.3, 0.006, 0.3)
            model = mne.make_bem_model(subject=subject, conductivity=conductivity, subjects_dir=subjects_dir)
            bem = mne.make_bem_solution(model)
            
            epochs = mne.make_fixed_length_epochs(raw, duration=96.0, preload=False)
            epochs.set_eeg_reference(projection=True)
            epochs.apply_baseline((None,None))
            fwd = mne.make_forward_solution(
                epochs.info, trans=coreg.trans, src=src, bem=bem, verbose=True
            )
            
            cov = mne.compute_covariance(epochs)
            
            inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, verbose=True)
            
            method = "sLORETA"
            snr = 3.0
            lambda2 = 1.0 / snr**2
            stc = apply_inverse_epochs(
                epochs,
                inv,
                lambda2,
                method=method,
                pick_ori=None,
                verbose=True,
                return_generator=False
            )
            
            labels_parc = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
            
            filtered_labels = filter_labels_with_vertices(labels_parc, src)
            label_ts = mne.extract_label_time_course(stc, filtered_labels, src, mode='auto', return_generator=False, allow_empty=False)
            
            n=len(epochs)
            region = [label.name for label in filtered_labels]
        
            mats=[]
            condition_numbers = []
            for i in range(100):
                inds = np.random.choice(range(n),int(n/2),replace=False)
                epoch_data = np.array(label_ts)
                epoch_idx = np.arange(len(inds))
                dynsys_mat,condition_number = dynsys.dynSys(epoch_data[inds], epoch_idx, region, sampling_time=0.0025)
                mats.append(dynsys_mat)
                condition_numbers.append(condition_number)
        
            region = [label.name for label in filtered_labels]
            bootstrap_samples = list(range(100))
        
            # Create xarray DataArray
            xarray = xr.DataArray(
                np.array(mats), 
                dims=["bootstrap_samples", "region1", "region2"],
                coords={
                    "bootstrap_samples": bootstrap_samples,
                    "region1": region, 
                    "region2": region,
                    "condition_number": ("bootstrap_samples", condition_numbers)  # Adding as a coordinate
                }
            )
            xarray.to_netcdf(f'./dynsys/{subject}_array_dynsys_{condition}_alpha.nc')
        except Exception as e:
            print(f'failed on {subject}')
            print(e)


if __name__ == "__main__":

    start_time = time.time()

    # Use ProcessPoolExecutor to parallelize the processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        # Map the process_subject function to all subjects
        futures = [executor.submit(process_subject, subject, subjects_dir, eeg_dir, conditions) for subject in subjects]

    print(time.time() - start_time)




