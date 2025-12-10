# -*- coding: utf-8 -*-
"""
This file contains useful function used in the main code implementing the 
identification of Brugada syndrome in ECGs, proposed in [1].


[1] M. Scarpiniti and A. Uncini, "Exploiting phase information for the 
identification of Brugada syndrome: A preliminary study", in Italian Workshop 
on Neural Networks (WIRN 2024), Vietri sul Mare (SA), Italy, June 05-07, 2024.

Created on Fri Jun 28 10:41:44 2024

@author: Michele Scarpiniti -- DIET Dpt. (Sapienza University of Rome)
"""

import numpy as np

import cv2
from ssqueezepy import cwt
from scipy.fft import fft



# Function for loading the input data to be used in the proposed model --------
def load_data(data_folder, L=1):
    """
    Function for loading input data for the proposed model.
    

    Parameters
    ----------
    data_folder : folder containing data.
    L : integer equal to the number of used leads. The default is 1.

    Returns
    -------
    X : time-domain data.
    Xm : FFT-magnitude data.
    Xp : FFT-phase data.
    y : labels.

    """
    
    training_set1 = data_folder + 'X_V1.npy'
    training_set2 = data_folder + 'X_V2.npy'
    training_set3 = data_folder + 'X_V3.npy'

    training_setm1 = data_folder + 'Module_V1.npy'
    training_setp1 = data_folder + 'Phase_V1.npy'
    training_setm2 = data_folder + 'Module_V2.npy'
    training_setp2 = data_folder + 'Phase_V2.npy'
    training_setm3 = data_folder + 'Module_V3.npy'
    training_setp3 = data_folder + 'Phase_V3.npy'
    
    
    if L==1:  
        X = np.load(training_set1)
        X = X[:,:,np.newaxis]
        Xm = np.load(training_setm1)
        Xp = np.load(training_setp1)
    elif L==2:
        X1  = np.load(training_set1)
        X2  = np.load(training_set2)
        Xm1 = np.load(training_setm1)
        Xp1 = np.load(training_setp1)
        Xm2 = np.load(training_setm2)
        Xp2 = np.load(training_setp2)

        X1 = X1[:,:,np.newaxis]
        X2 = X2[:,:,np.newaxis]
        X  = np.concatenate((X1,X2), axis=2)
        Xm = np.concatenate((Xm1,Xm2), axis=1)
        Xp = np.concatenate((Xp1,Xp2), axis=1)
    elif L==3:
        X1 = np.load(training_set1)
        X2 = np.load(training_set2)
        X3 = np.load(training_set3)
        Xm1 = np.load(training_setm1)
        Xp1 = np.load(training_setp1)
        Xm2 = np.load(training_setm2)
        Xp2 = np.load(training_setp2)
        Xm3 = np.load(training_setm3)
        Xp3 = np.load(training_setp3)

        X1 = X1[:,:,np.newaxis]
        X2 = X2[:,:,np.newaxis]
        X3 = X3[:,:,np.newaxis]
        X  = np.concatenate((X1,X2,X3), axis=2)
        Xm = np.concatenate((Xm1,Xm2,Xm3), axis=1)
        Xp = np.concatenate((Xp1,Xp2,Xp3), axis=1)
    elif L==9:
        training_set_all  = data_folder + 'X_all.npy'
        training_setm_all = data_folder + 'Module_all.npy'
        training_setp_all = data_folder + 'Phase_all.npy'
        
        X  = np.load(training_set_all)
        Xm = np.load(training_setm_all)
        Xp = np.load(training_setp_all)
        
        Xm = np.moveaxis(Xm, [1,2], [2,1])
        Xm = np.reshape(Xm, (Xm.shape[0], 9*76))
        Xp = np.moveaxis(Xp, [1,2], [2,1])
        Xp = np.reshape(Xp, (Xp.shape[0], 9*76))
    else:
        print('Number of leads unsupported!')
    
      
    # Load labels
    training_lab  = data_folder + 'y.npy'
    y = np.load(training_lab)
    
    
    return X, Xm, Xp, y
#------------------------------------------------------------------------------





# Function for loading the input data to be used in the DNN model -------------
def load_data_DNN(data_folder, L=1):
    """
    Function for loading input data for the DNN model. This function can be 
    also used for loading data used in traditional ML models.
    

    Parameters
    ----------
    data_folder : folder containing data.
    L : integer equal to the number of used leads. The default is 1.

    Returns
    -------
    X : time-domain data.
    y : labels.

    """

    training_set1 = data_folder + 'X_V1.npy'
    training_set2 = data_folder + 'X_V2.npy'
    training_set3 = data_folder + 'X_V3.npy'
    training_set_all = data_folder + 'X.npy'
    
    if L==1:
        X = np.load(training_set1)
    elif L==2:
        X1 = np.load(training_set1)
        X2 = np.load(training_set2)
        X = np.concatenate((X1,X2), axis=1)
    elif L==3:
        X1 = np.load(training_set1)
        X2 = np.load(training_set2)
        X3 = np.load(training_set3)
        X = np.concatenate((X1,X2,X3), axis=1)
    elif L==9:
        X = np.load(training_set_all)
    else:
        print('Number of leads unsupported!')

    
    # Load labels
    training_lab  = data_folder + 'y.npy'
    y = np.load(training_lab)
    
    
    return X, y
#------------------------------------------------------------------------------





# Function for loading the input data to be used in the CNN model -------------
def load_data_CNN(data_folder, L=1):
    """
    Function for loading input data for the CNN model.
    

    Parameters
    ----------
    data_folder : folder containing data.
    L : integer equal to the number of used leads. The default is 1.

    Returns
    -------
    X : time-domain data.
    y : labels.

    """
      
    training_set1 = data_folder + 'X_V1.npy'
    training_set2 = data_folder + 'X_V2.npy'
    training_set3 = data_folder + 'X_V3.npy'
    training_set_all = data_folder + 'X.npy'
    
    if L==1:
        X = np.load(training_set1)
        X = X[:,:,np.newaxis]
    elif L==2:
        X1 = np.load(training_set1)
        X2 = np.load(training_set2)
        X1 = X1[:,:,np.newaxis]
        X2 = X2[:,:,np.newaxis]
        X  = np.concatenate((X1,X2), axis=2)
    elif L==3:
        X1 = np.load(training_set1)
        X2 = np.load(training_set2)
        X3 = np.load(training_set3)
        X1 = X1[:,:,np.newaxis]
        X2 = X2[:,:,np.newaxis]
        X3 = X3[:,:,np.newaxis]
        X  = np.concatenate((X1,X2,X3), axis=2)
    elif L==9:
        X = np.load(training_set_all)
        X = np.reshape(X, (X.shape[0],9,150))
        X = np.moveaxis(X, [1,2], [2,1])
    else:
        print('Number of leads unsupported!')
    
    
    # Load labels
    training_lab  = data_folder + 'y.npy'
    y = np.load(training_lab)
    
    
    return X, y
#------------------------------------------------------------------------------





# Function for loading the input data to be used in the S3 model --------------
def load_data_S3(data_folder, L=1):
    """
    Function for loading input data for the S3 strategy model.
    

    Parameters
    ----------
    data_folder : folder containing data.
    L : integer equal to the number of used leads. The default is 1.

    Returns
    -------
    X : scalogram+phasogram channel-like data.
    y : labels.

    """
    
    training_sets1 = data_folder + 'X_V1_scalograms.npy'    
    training_sets2 = data_folder + 'X_V2_scalograms.npy'    
    training_sets3 = data_folder + 'X_V3_scalograms.npy'
    training_setp1 = data_folder + 'X_V1_phasograms.npy'
    training_setp2 = data_folder + 'X_V2_phasograms.npy'
    training_setp3 = data_folder + 'X_V3_phasograms.npy'
    
    if L==1:
        Xs1 = np.load(training_sets1)
        Xp1 = np.load(training_setp1)
        
        Xs1 = Xs1[:,:,:,np.newaxis]
        Xp1 = Xp1[:,:,:,np.newaxis]
        
        X = np.concatenate((Xs1, Xp1), axis=3)
    elif L==2:
        Xs1 = np.load(training_sets1)
        Xs2 = np.load(training_sets2)
        Xp1 = np.load(training_setp1)
        Xp2 = np.load(training_setp2)
        
        Xs1 = Xs1[:,:,:,np.newaxis]
        Xs2 = Xs2[:,:,:,np.newaxis]
        Xp1 = Xp1[:,:,:,np.newaxis]
        Xp2 = Xp2[:,:,:,np.newaxis]
        
        X = np.concatenate((Xs1, Xp1, Xs2, Xp2), axis=3)
    elif L==3:
        Xs1 = np.load(training_sets1)
        Xs2 = np.load(training_sets2)
        Xs3 = np.load(training_sets3)
        Xp1 = np.load(training_setp1)
        Xp2 = np.load(training_setp2)
        Xp3 = np.load(training_setp3)
        
        Xs1 = Xs1[:,:,:,np.newaxis]
        Xs2 = Xs2[:,:,:,np.newaxis]
        Xs3 = Xs3[:,:,:,np.newaxis]
        Xp1 = Xp1[:,:,:,np.newaxis]
        Xp2 = Xp2[:,:,:,np.newaxis]
        Xp3 = Xp3[:,:,:,np.newaxis]
        
        X = np.concatenate((Xs1, Xp1, Xs2, Xp2, Xs3, Xp3), axis=3)
    else:
        print('Number of leads unsupported!')
        
    
    # Load labels
    training_lab  = data_folder + 'y.npy'
    y = np.load(training_lab)
    
    
    return X, y
#------------------------------------------------------------------------------





# Function for loading the input data to be used in the S3 model --------------
def load_data_S5(data_folder, L=1):
    """
    Function for loading input data for the S5 strategy model.
    

    Parameters
    ----------
    data_folder : folder containing data.
    L : integer equal to the number of used leads. The default is 1.

    Returns
    -------
    Xm : scalogram channel-like data.
    Xp : phasogram channel-like data.
    y  : labels.

    """
    
    training_sets1 = data_folder + 'X_V1_scalograms.npy'
    training_sets2 = data_folder + 'X_V2_scalograms.npy'
    training_sets3 = data_folder + 'X_V3_scalograms.npy'
    training_setp1 = data_folder + 'X_V1_phasograms.npy'
    training_setp2 = data_folder + 'X_V2_phasograms.npy'
    training_setp3 = data_folder + 'X_V3_phasograms.npy'
    
    if L==1:
        Xm = np.load(training_sets1)
        Xp = np.load(training_setp1)
        
        Xm = Xm[:,:,:,np.newaxis]
        Xp = Xp[:,:,:,np.newaxis]
    elif L==2:
        Xs1 = np.load(training_sets1)
        Xs2 = np.load(training_sets2)
        Xp1 = np.load(training_setp1)
        Xp2 = np.load(training_setp2)

        Xs1 = Xs1[:,:,:,np.newaxis]
        Xs2 = Xs2[:,:,:,np.newaxis]
        Xp1 = Xp1[:,:,:,np.newaxis]
        Xp2 = Xp2[:,:,:,np.newaxis]
        
        Xm = np.concatenate((Xs1, Xs2), axis=3)
        Xp = np.concatenate((Xp1, Xp2), axis=3)
    elif L==3:
        Xs1 = np.load(training_sets1)
        Xs2 = np.load(training_sets2)
        Xs3 = np.load(training_sets3)
        Xp1 = np.load(training_setp1)
        Xp2 = np.load(training_setp2)
        Xp3 = np.load(training_setp3)

        Xs1 = Xs1[:,:,:,np.newaxis]
        Xs2 = Xs2[:,:,:,np.newaxis]
        Xs3 = Xs3[:,:,:,np.newaxis]
        Xp1 = Xp1[:,:,:,np.newaxis]
        Xp2 = Xp2[:,:,:,np.newaxis]
        Xp3 = Xp3[:,:,:,np.newaxis]
        
        Xm = np.concatenate((Xs1, Xs2, Xs3), axis=3)
        Xp = np.concatenate((Xp1, Xp2, Xp3), axis=3)
    
    
    # Load labels
    training_lab  = data_folder + 'y.npy'
    y = np.load(training_lab)
    
    
    return Xm, Xp, y
#------------------------------------------------------------------------------




# Scale matrix in range [0, 1] ------------------------------------------------
def scale(matrix):
    """
    Perform min-max scaling of a matrix.

    Parameters
    ----------
    matrix : input matrix.

    Returns
    -------
    scaled_matrix : matrix scaled in range [0, 1].

    """

    min_val = np.min(matrix)
    max_val = np.max(matrix)
    
    scaled_matrix = (matrix - min_val) / (max_val - min_val)
    
    return scaled_matrix
#------------------------------------------------------------------------------





# Extract scalograms and phasograms -------------------------------------------
def extract_cwt(X, N=224, fs=200):
    """
    Exctract the scalograms and phasograms of input ECGs.

    Parameters
    ----------
    X : input ECG lead.
    N : pixel size of output scalogram. The default is 224.
    fs : Sampling frequency. The default is 200.

    Returns
    -------
    feat_Wm : list of extracted scalograms.
    feat_Wp : list of extracted phasograms.

    """

    L = X.shape[0]
    
    feat_Wm = []
    feat_Wp = []
    
    
    # Main loop
    for i in range(L):
        # Extract the CWT
        W, scales = cwt(X[i,:], wavelet='morlet', fs=fs)
        
        # Compute the scalogram and its phasogram
        Wm = np.abs(W)
        Wp = np.angle(W)
        
        # Resize to a suitable image size (e.g., 224x224 or 227x227)
        Wm = cv2.resize(Wm, dsize=(N, N), interpolation=cv2.INTER_LINEAR)
        Wp = cv2.resize(Wp, dsize=(N, N), interpolation=cv2.INTER_LINEAR)
        
        # Normalize features
        Wm = scale(Wm)
        Wp = scale(Wp)
        
        # Transform as an integer image
        Wm = np.array(255*Wm, dtype = 'uint8')
        Wp = np.array(255*Wp, dtype = 'uint8')
        
        # Append features
        feat_Wm.append(Wm)
        feat_Wp.append(Wp)
        
        # Print advancement
        if (i % 100):
            print("\rAdvancement: {}%".format(round(100*i/L, 1)), end='')
        
    
    print("\rAdvancement: {}%".format(100.0), end='\n')
    
    return feat_Wm, feat_Wp
#------------------------------------------------------------------------------





# Extract FFT magnitude and phase ---------------------------------------------
def extract_fft(X, N_fft):
    """
    Extract the FFT magnitude and phase of the input ECGs.

    Parameters
    ----------
    X : input ECG lead.
    N_fft : number of FFT bins.

    Returns
    -------
    Mo : FFT magnitude matrix.
    Ph : FFT phase matrix.

    """

    N = X.shape[0]
    M_fft = N_fft//2
    
    Mo = np.zeros((N,M_fft))
    Ph = np.zeros((N,M_fft))
    
    
    for i in range(N):
        x = X[i,:]
        x_fft = fft(x, n=N_fft)
        m = np.abs(x_fft)
        p = np.unwrap(np.angle(x_fft))
        
        Mo[i,:] = m[:M_fft]
        Ph[i,:] = p[:M_fft]
        
    return Mo, Ph
#------------------------------------------------------------------------------





# Extract FFT magnitude and phase when 9 leads are used------------------------
def extract_fft_9leads(X, N_fft):
    """
    Extract the FFT magnitude and phase of the input 9-leads ECG.

    Parameters
    ----------
    X : input 9-leads ECG.
    N_fft : number of FFT bins.

    Returns
    -------
    Mo : FFT magnitude matrix.
    Ph : FFT phase matrix.

    """
       
    N, _, L = X.shape
    M_fft = N_fft//2

    Mo = np.zeros((N,M_fft,L))
    Ph = np.zeros((N,M_fft,L))


    for i in range(N):
        for j in range(L):
            x = X[i,:,j]
            x_fft = fft(x, n=N_fft)
            m = np.abs(x_fft)
            p = np.unwrap(np.angle(x_fft))
            
            Mo[i,:,j] = m[:M_fft]
            Ph[i,:,j] = p[:M_fft]    
    
    return Mo, Ph
#------------------------------------------------------------------------------




