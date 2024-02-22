#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mne
from mne.coreg import Coregistration
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
import numpy as np
from mne_connectivity import spectral_connectivity_epochs
from matplotlib import pyplot as plt
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout
import matplotlib.image as mpimg
from pathlib import Path
import seaborn as sns
import pandas as pd
import gc
import os
from pathlib import Path
from spectral_connectivity import Multitaper, Connectivity
from pymatreader import read_mat
import xarray as xr
import dynsys
import time
import logging
import argparse


# In[2]:


#subjects = [subject for subjects in os.listdir('/scratch/MPI-LEMON/freesurfer/subjects/inspected') if subject.startswith('sub')]
subjects_dir = '/scratch/MPI-LEMON/freesurfer/subjects'

#subjects = ['sub-032304','sub-032310']
#subjects_dir = "/Applications/freesurfer/7.4.1/subjects/inspected"


# In[3]:


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


# In[ ]:


def setup_logger(subject):
    logger = logging.getLogger(subject)
    logger.setLevel(logging.INFO)
    
    log_path = Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)
    
    fh = logging.FileHandler(f'logs/{subject}.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger


# In[5]:


def process_subject(subject):
    subjects_dir = "/scratch/MPI-LEMON/freesurfer/subjects"
    conditions = ['EC','EO']
    condition_mats = []
    logger = setup_logger(subject)
    try:
        logger.info(f"Processing subject: {subject}")

        condition_mats = []
        for condition in conditions:
            src = mne.setup_source_space(subject, add_dist="patch", subjects_dir=subjects_dir)
            raw = mne.io.read_raw_eeglab(f"/scratch/MPI-LEMON/freesurfer/subjects/eeg/{subject}/{subject}_{condition}.set")
            
            info = raw.info
            fiducials = "estimated"
            coreg = Coregistration(info, subject, subjects_dir, fiducials=fiducials)
            
            conductivity = (0.3, 0.006, 0.3)
            model = mne.make_bem_model(subject=subject, conductivity=conductivity, subjects_dir=subjects_dir)
            bem = mne.make_bem_solution(model)
            
            epochs = mne.make_fixed_length_epochs(raw, duration=1.0, preload=False)
            epochs.set_eeg_reference(projection=True)
            
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
        
            methods_mats = []
            
            mne_methods = ['coh']
            n=len(epochs)
            mne_mats=[]
            for i in range(100):
                inds = np.random.choice(range(n),int(n/2),replace=False)
                mne_con = spectral_connectivity_epochs(np.array(label_ts)[inds], 
                                                    method=mne_methods, sfreq=250, mode='multitaper', 
                                                    fmin=8, fmax=13, fskip=0, faverage=False, 
                                                    tmin=None, tmax=None, mt_bandwidth=None, mt_adaptive=False, 
                                                    mt_low_bias=True, block_size=1000, n_jobs=1, verbose=None)
                mat = mne_con.get_data("dense").mean(2) + mne_con.get_data("dense").mean(2).T
                mne_mats.append(mat)
        
            methods_mats.append(mne_mats)
            
            dynsys_mats=[]
            region = [label.name for label in filtered_labels]
            times = stc[0].times
            epoch_data = np.array(label_ts)
            for i in range(100):
                inds = np.random.choice(range(n),int(n/2),replace=False)
                epoch_idx = np.arange(len(inds))
                dynsys_mat = dynsys.dynSys(times, epoch_data[inds], epoch_idx, region, sampling_time=0.0025)
                dynsys_mats.append(dynsys_mat)
        
            methods_mats.append(dynsys_mats)
            condition_mats.append(methods_mats)
        
        region = [label.name for label in filtered_labels]
        methods = ['coh','dynsys']
        bootstrap_samples = list(range(0,100))
        
        xarray = xr.DataArray(np.array(condition_mats), dims=["conditions","methods","bootstrap_samples","region1","region2"],
                            coords={"conditions":conditions,
                                    "methods":methods,"bootstrap_samples":bootstrap_samples,
                                    "region1":region, "region2":region})
    
        output_dir = Path("MPI_OUTPUT_sept23")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the xarray for this subject to a NetCDF file
        output_file = output_dir / f"{subject}_array.nc"
        xarray.to_netcdf(output_file)
        logger.info(f"Saved xarray for subject {subject} to {output_file}")
            
    except Exception as e:
        logger.error(f"An error occurred while processing subject {subject}: {e}")


# In[ ]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a subject.')
    parser.add_argument('subject', type=str, help='The name of the subject to process.')
    
    args = parser.parse_args()
    process_subject(args.subject)
    print(time.time())

