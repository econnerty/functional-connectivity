#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mne
from mne.coreg import Coregistration
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
import numpy as np
from pathlib import Path
import pandas as pd
import os
from pathlib import Path
from spectral_connectivity import Multitaper, Connectivity
import xarray as xr
import time
import gc


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
conditions = ['EC','EO']


# In[36]:


start_time = time.time()

for subject in subjects:
    try:
        for condition in conditions:
            src = mne.setup_source_space(subject, add_dist="patch", subjects_dir=subjects_dir)
            raw = mne.io.read_raw_eeglab(f"/work/erikc/eeg/{subject}/{subject}_{condition}.set")
            info = raw.info
            fiducials = "estimated"
            coreg = Coregistration(info, subject, subjects_dir, fiducials=fiducials)
            
            conductivity = (0.3, 0.006, 0.3)
            model = mne.make_bem_model(subject=subject, conductivity=conductivity, subjects_dir=subjects_dir)
            bem = mne.make_bem_solution(model)
            
            epochs = mne.make_fixed_length_epochs(raw, duration=8.0, preload=False)
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
        
            mats=[]
            for i in range(100):
                inds = np.random.choice(range(n),int(n/2),replace=False)
                m = Multitaper(time_series=np.array(label_ts)[inds].transpose(2, 0, 1),
                    sampling_frequency=250)
                
                c = Connectivity(fourier_coefficients=m.fft(),
                        frequencies=m.frequencies,
                        time=m.time)    
                
                con = getattr(c, 'generalized_partial_directed_coherence')()
                mat = np.nan_to_num(con.squeeze())
                mats.append(mat[64:112,:,:])
                frequencies = c.frequencies
                del m
                del c
                del con
                gc.collect()
        
            region = [label.name for label in filtered_labels]
            frequencies = list(frequencies[64:112])
            bootstrap_samples = list(range(100))
        
            xarray = xr.DataArray(np.array(mats), dims=["bootstrap_samples","frequencies","region1","region2"],
            coords={"bootstrap_samples":bootstrap_samples,"frequencies":frequencies,"region1":region, "region2":region})
            xarray.to_netcdf(f'./gpdc/{subject}_array_gpdc_{condition}.nc')
    except Exception as e:
        print(f'failed on {subject}')
        print(e)
        continue
print(time.time()-start_time)


# In[ ]:




