#%% Import Necessary Packages

import sys
import os
import pickle
import scipy 
import mne
import numpy as np
import pandas as pd
from sklearn import mixture
from scipy import signal as sg
from scipy.integrate import simps
from scipy import stats
from functools import partial
from datetime import datetime
import matplotlib.pyplot as plt


#%% Import Functions

#Function to take raw edf file, a list of channels, and a range of time
#samples, and return a downsampled set of signals.  Antialiasing is implemented
#via scipy's decimate function
def edf_decimate(raw_edf,channels,dec=10,range=(0,None)):
    signal_dict = dict()
    for signal in channels:
        print('decimating signal: {x}'.format(x=signal))
        y = raw_edf.get_data(
            picks=[signal],
            start=range[0],
            stop=range[1]).flatten()
        signal_dict[signal] = sg.decimate(x = y, q = dec)
        print('new length: {x}'.format(x=len(signal_dict[signal])))
    signal_frame = pd.DataFrame.from_dict(signal_dict)
    return signal_frame

#Function to create bin index array.  
def create_binid(max,binsize,scale_binsize=False):
    array = np.arange(0,max)
    top = np.ceil(max/binsize)*binsize
    bin_breaks = np.arange(0,top,binsize)
    bins = np.digitize(array, bin_breaks)-1
    bins = bins*binsize if scale_binsize else bins
    return bins

#Given array x, the sampling frequency, a dictionary with power band names as
#keys and tuple of lower/upper bounds as values, and a window for welch 
#periodogram, returns array of power bands, either in relative or absolute
#power, depending upon parameter ret_rel
def get_pbands(x, sf, power_bands=None, win_sec=4, ret_rel=True):
    win =  win_sec * sf
    freq_res = 1/win_sec
    freqs, psd = sg.welch(
        x.to_numpy().flatten(), 
        sf, 
        nperseg=win)
    p_abs = np.zeros(len(power_bands))
    for id, pband in enumerate(power_bands):
        low, high = power_bands[pband]
        idx = np.logical_and(freqs >= low, freqs <= high)
        p_abs[id] = simps(psd[idx], dx=freq_res)
    p_tot = simps(psd, dx=freq_res)
    p_rel = p_abs / p_tot
    p_rel = tuple(p_rel.tolist())
    return [(p_rel)] if ret_rel else [(p_abs)]

#Given array x, return RMS
def get_rms(x):
    rms = (((x**2).sum())/len(x))**(1/2)
    return rms

#Given array x with 2 underlying components, fit gaussian mixture model and
#return assignments.  Lower mean sample has assignment of 1.
def fit_gmm(x, cov_type='full'):
    #convert to 2d numpy array and z score
    x = stats.zscore(np.array(x).reshape(-1,1))
    #create gmm object
    gmm = mixture.GaussianMixture(
        n_components = 2,
        covariance_type = cov_type
    ).fit(x)
    #create gmm assignments
    assignments = gmm.predict(x)
    #make assignment of lower mean = 0
    if x[assignments==0].mean() > x[assignments==1].mean():
        assignments = -1 * (assignments-1)
    return assignments


#Given datafraame containing columns 'EEG_rms', 'EEG_tdratio',
#and 'wake', classify 3 sleep states (0=sws, 1=rem, 2=wake)
#Argument usewake specifies whether to include wakefullness
#in gmm separating sws/rem.
def get_state(df, usewake=False):

    #fit gmm to sleep data to separate deep/non-deepsleep
    if usewake:
        x = df[['EEG_tdratio']]
    else:
        x = df.loc[df.wake=='sleep',['EEG_tdratio']]
    x.EEG_tdratio = stats.zscore(x.EEG_tdratio)
    gmm = mixture.GaussianMixture(
        n_components = 2,
        covariance_type = 'full').fit(x)

    #create gmm assignments
    assignments = gmm.predict(x)
    mean0 = x.loc[assignments==0,['EEG_tdratio']].mean()[0]
    mean1 = x.loc[assignments==1,['EEG_tdratio']].mean()[0]
    if mean0 > mean1:
        assignments = (assignments-1)*-1
    
    #create state variable, including waking state
    df['state'] = 2
    if usewake:
        df.loc[df.wake=='sleep',['state']] = assignments[df.wake=='sleep'] 
    else:
        df.loc[df.wake=='sleep',['state']] = assignments 
    return df['state']


#Given array states (rem, sws, and wake), reclassify any rem state preceded 
#by wakefullness as wakefullness
def rem_req(state): 
    try: 
        state = state.to_numpy()
    except: 
        pass
    pr_state = np.insert(state,0,state[0])[:-1]
    wake2rem = np.logical_and(state=='rem',pr_state=='wake')
    while wake2rem.sum()>0:
        state[wake2rem]='wake'
        pr_state = np.insert(state,0,state[0])[:-1]
        wake2rem = np.logical_and(state=='rem',pr_state=='wake')
    return state