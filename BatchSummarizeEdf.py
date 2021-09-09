#%% Import Necessary Packages

import sys
import os
import fnmatch
import pickle
import scipy 
import mne
import numpy as np
import pandas as pd
import plotnine as p9
from scipy import signal as sg
from scipy.integrate import simps
from scipy import stats
from sklearn import mixture
from functools import partial
from datetime import datetime
from matplotlib import pyplot as plt
from importlib import reload as reload


import SleepScore_functions
reload(SleepScore_functions)
from SleepScore_functions import edf_decimate, create_binid, get_pbands, get_rms, fit_gmm, get_state





#%% Specify Directory Containing All EDF Files to Process

directory = '/Volumes/csstorage/Zach/ToBeFiled/MSCPpilot/edf/ms01'
files = fnmatch.filter(
    sorted(os.listdir(directory)), 
    '*.edf'
)
print('files to process: ', files)





# %% Specify parameters common to all files


#revert from daylight savings to standard time
ds_revert = False

#Channel information
ext_chs = ['Activity','EEG','EMG','SignalStr','Temp'] #names of channels to extract
smpl_rg = (0,None) #range of original samples to extract.  Def = (0,None) 
dsmpl_order = 5 #order by which to downsample (1=no downsample, 10, 1 in 10)
sf = 100 #sampling frequency, after downsampling

#Binning parameters
bin_col = 'tbin' #string specifying column with binning  index
bin_length = 6 #bin length, in seconds

#highpass filter for EMG
highpass = {
    'use' : True,
    'ch' : 'EMG',
    'cut' : 10,
    'order' : 8
}

#PSD parameters:
EEG_chs = ['EEG'] #list of EEG column names
win_sec = 4 #window length in seconds
power_bands = {
    'delta' : (0.5, 4),
    'theta' : (5, 9),
    'sigma' : (10, 14),
    'beta' : (15, 20)
}

#Channels to computer summary info for
rms_chs = ['EEG','EMG'] #list of signals to return rms for
mean_chs = ['Activity','SignalStr','Temp'] #list of signals ot return mean  for




# %% 


if highpass['use'] == True:
    highpass['sos'] = sg.butter(
        N=highpass['order'],
        Wn=highpass['cut'], 
        btype='hp',
        fs=sf, 
        output='sos')

for file in files:

    fpath = os.path.join(directory, file) 
    data_edf = mne.io.read_raw_edf(fpath)
    print('processing: {x}'.format(x=file))
    print('start time: {x}'.format(x=data_edf.info['meas_date']))

    #get downsampled signals
    sig_df = edf_decimate(
        raw_edf=data_edf,
        channels=ext_chs,
        dec=dsmpl_order,
        range=(smpl_rg[0],smpl_rg[1]))

    #Lowpass filter EMG
    if highpass['use'] == True:
        print('highpass filtering EMG at: {x} hz'.format(x=highpass['cut']))
        sig_df[highpass['ch']] = sg.sosfilt(
            highpass['sos'], 
            sig_df[highpass['ch']])

    #Create binning array
    sig_df[bin_col] = create_binid(
        max = len(sig_df),
        binsize = bin_length*sf)

    #Create pdSeries with time stamps
    start_timestamp = data_edf.info['meas_date'].timestamp()
    if ds_revert:
        start_timestamp = start_timestamp-(60*60) 
        print('adjusting start time to: {x}'.format(x=datetime.utcfromtimestamp(start_timestamp)))
    times = pd.Series(
        [datetime.utcfromtimestamp(
        start_timestamp+x*bin_length
        ) for x in np.arange(sig_df.tbin.max()+1)], name=bin_col)

    #Perform spectral analysis
    print('performing binned spectral analysis...')
    pbands_binned = sig_df[EEG_chs].groupby(sig_df[bin_col]).agg(
        get_pbands,
        sf=sf, power_bands=power_bands, win_sec=win_sec, ret_rel=True)
    for ch in EEG_chs:
        pbands_binned[
            ['_'.join([ch,letter]) for letter in power_bands.keys()]
            ] = pbands_binned[ch].apply(pd.Series)
    pbands_binned.drop(columns=EEG_chs, inplace=True)
    pbands_binned['EEG_tdratio'] = pbands_binned.EEG_theta/pbands_binned.EEG_delta

    #Get RMS
    print('computing binned rms values...')
    rms_binned = sig_df[rms_chs].groupby(sig_df[bin_col]).agg(get_rms)
    rms_binned.columns = ['_'.join([x,'rms']) for x in rms_binned.columns]
    for col in rms_binned.columns:
        rms_binned[col] = stats.zscore(rms_binned[col])
    rms_binned['EMG_rms_medfilt'] = sg.medfilt(
        volume = rms_binned.EMG_rms,
        kernel_size = 7)

    #Get means
    print('computing binned means...')
    mean_binned = sig_df[mean_chs].groupby(sig_df[bin_col]).agg('mean')
    mean_binned.columns = ['_'.join([x,'mean']) for x in mean_binned.columns]

    #Create summary data frame and save
    print('saving summaries to disk...\n ')
    summary = pd.concat([times,rms_binned,pbands_binned,mean_binned],axis=1)
    summary['file'] = file
    summary = summary[['file'] + summary.columns.to_list()[:-1]]
    summary.to_pickle('.'.join([os.path.splitext(fpath)[0],'pickle']))

    #plot EEG and 
    plt.figure(1)
    plt.plot(summary.EEG_rms)
    plt.title("{x} EEG RMS".format(x=file))
    plt.figure(2)
    plt.plot(summary.EMG_rms)
    plt.title("{x} EMG RMS".format(x=file))
    plt.figure(3)
    plt.plot(summary.SignalStr_mean)
    plt.title("{x} Signal Strength".format(x=file))
    plt.show()

    fwith = sig_df.copy()

    #del data
    del times,rms_binned, pbands_binned, mean_binned, sig_df, data_edf, start_timestamp, summary
    




# %%
