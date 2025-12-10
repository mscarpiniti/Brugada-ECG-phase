# -*- coding: utf-8 -*-
"""
Main script for creating all data used for implementing all models proposed in
[1] for the identification of the Brugada syndrome.

Data are available at: https://datadryad.org/stash/dataset/doi:10.5061/dryad.s1rn8pkd9


[1] M. Scarpiniti and A. Uncini, "Exploiting phase information for the
identification of Brugada syndrome: A preliminary study", in Italian Workshop
on Neural Networks (WIRN 2024), Vietri sul Mare (SA), Italy, June 05-07, 2024.


Created on Fri Jun 28 16:00:54 2024

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""



import pandas as pd
import numpy as np
import utils as ut



# Set variables
leads = ['I', 'II', 'III', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

N_lead = 9
N_leng = 150
N_fft = 256

start_lead = 4
end_lead = 6



# %% Load the training data

path = './Data/'
file_X = 'X.xlsx'
file_y = 'y.xlsx'

filename_X = path + file_X
filename_y = path + file_y

data = pd.read_excel(filename_X)
label = pd.read_excel(filename_y)


X = data.values
y = label.values

save_file_X = path + 'X.npy'
save_file_y = path + 'y.npy'

np.save(save_file_X, X)
np.save(save_file_y, y)




# %% Segmentation of all the dataset for each lead

N_ecg  = X.shape[0]

X_seg = np.reshape(X, (N_ecg,N_lead,N_leng))

X_all = np.moveaxis(X_seg, [1,2], [2,1])
fname = path + 'X_all.npy'
np.save(fname, X_all)




# %% Extract and save each lead

for k in range(start_lead-1, end_lead):
    Xl = X_seg[:,k,:]
    fname = path + 'X_' + leads[k] + '.npy'
    np.save(fname, Xl)




# %% Extract and save the FFT magnitude and phase of each lead

for k in range(start_lead-1, end_lead):
    Xl = X_seg[:,k,:]
    Mo, Ph = ut.extract_fft(Xl, N_fft)

    file1 = path + 'Module_' + leads[k] + '.npy'
    file2 = path + 'Phase_' + leads[k] + '.npy'

    np.save(file1, Mo)
    np.save(file2, Ph)




# %% Extract and save the FFT magnitude and phase of all 9-leads

Mo, Ph = ut.extract_fft_9leads(X_all, N_fft)


file1 = path + 'Module_all.npy'
file2 = path + 'Phase_all.npy'

np.save(file1, Mo)
np.save(file2, Ph)




# %% Extract and save the scalograms and phasograms of each lead

N = 224
fs = 200


for k in range(start_lead-1, end_lead):
    Xl = X_seg[:,k,:]
    Wm, Wp = ut.extract_cwt(Xl, N=N, fs=fs)

    save_file_m = path + 'X_' + leads[k] + '_scalograms.npy'
    save_file_p = path + 'X_' + leads[k] + '_phasograms.npy'

    np.save(save_file_m, Wm)
    np.save(save_file_p, Wp)
