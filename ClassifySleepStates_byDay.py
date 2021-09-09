
#%% #*Import Necessary Packages

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
from SleepScore_functions import edf_decimate, create_binid, get_pbands, get_rms, fit_gmm, get_state, rem_req





#%% #*Specify Directory Containing All EDF Files to Process


ms = 'ms01'

directory = '/Volumes/csstorage/Zach/ToBeFiled/MSCPpilot/edf/{x}'.format(x=ms)
plot_pfx = ms
sig_cutoff = 25





#%% #*Load file list and create output directory

files = fnmatch.filter(
    sorted(os.listdir(directory)), 
    '*.pickle'
)
print('files to process: ', files)

directory_out = os.path.join(directory,'summary_files')
if not os.path.isdir(directory_out):
    os.mkdir(directory_out)





#%% #*Loop through each day of data and score

try:
    del summary
except:
    pass

for file in files:

    #load file
    fpath = os.path.join(directory, file) 
    df = pd.read_pickle(fpath)

    #Drop rows where signal strength drops below cutoff
    df = df.loc[df.SignalStr_mean>sig_cutoff,]

    #discard data with missing values, saving them to separate file
    if df.EEG_theta.isna().sum() > 0:
        d_out = os.path.join(
            directory_out,
            '_'.join([file,'_dropped.csv']))
        print('\nWarning: dropping data from {x}'.format(x=fpath))
        print('\nSaving bad data to {x}'.format(x=d_out))
        df[df.EEG_theta.isna()].to_csv(d_out)
        df.dropna(inplace=True)
        df.EEG_rms = stats.zscore(df.EEG_rms)
        df.EMG_rms = stats.zscore(df.EMG_rms)
    
    #classify wakefulness using gmm
    df['wake'] = pd.Categorical(
        fit_gmm(df.EMG_rms)
        ).rename_categories(['sleep','wake'])

    #classify state using gmm
    df['state'] = pd.Categorical(
        get_state(df, usewake=True)
        ).rename_categories(['sws','rem','wake'])

    #combine data from different days
    try:
        summary = pd.concat([summary,df])
    except:
        summary = df
    del df

#redfine any REM periods preceded by wake state as wake
summary.sort_values(by=['tbin'],inplace=True)
summary['state'] = pd.Categorical(
    rem_req(summary.state), categories=['sws','rem','wake'])
summary.wake[summary.state=='wake']='wake'

#Save to pickle
summary[['file','tbin','wake','state','Activity_mean','Temp_mean']].to_pickle(
    os.path.join(directory_out,'_'.join([plot_pfx,'summary.pickle'])))





# %% #*Create Powerband Summary According to State

power_summary=summary[['state','EEG_delta','EEG_theta','EEG_sigma','EEG_beta']]
power_summary.columns = [ind_names.replace('EEG_','') for ind_names in power_summary]
power_summary = pd.melt(
    frame = power_summary, id_vars = 'state',
    value_vars =  power_summary.columns[1:].to_list(),
    var_name = 'pband', value_name = 'power')
power_summary.pband = pd.Categorical(
    power_summary.pband, categories = ['delta','theta','sigma','beta'])
power_summary = power_summary.groupby(
    ['state','pband'],as_index=False).agg(
    {'power' : ['mean','sem','std']})
power_summary.columns = [''.join(ind_names) if len(ind_names[-1])==0 else '_'.join(ind_names) for ind_names in power_summary]

#save to csv
power_summary.to_csv(
    os.path.join(directory_out,'_'.join([plot_pfx,'powersummary.csv'])))





# %% #*Mean summary by state

summary_means = summary.groupby('state').agg({
    'EEG_rms' : ['mean','sem'],
    'EMG_rms' : ['mean','sem'],
    'Activity_mean' : ['mean','sem'],
    'Temp_mean' : ['mean','sem']
}).reset_index(drop=False)
summary_means.columns = [''.join(ind_names) if len(ind_names[-1])==0 else '_'.join(ind_names) for ind_names in summary_means]

#save to csv
summary_means.to_csv(
    os.path.join(directory_out,'_'.join([plot_pfx,'avgsummary.csv'])))





# %% #*Transition Summary

transitions = pd.DataFrame({
    'file' : summary.file,
    'state' : summary.state,
    'prior_state' : np.insert(
        summary.state.to_numpy(),0,summary.state.iloc[0])[:-1]
    }).set_index(summary.tbin)
transitions = transitions[transitions.state!=transitions.prior_state]

trans_crosstab = pd.crosstab(
    transitions.state,transitions.prior_state#,normalize='index'
    ).reset_index(drop=False).melt(
        id_vars='state',
        value_vars=['rem','sws','wake'],
        value_name='number').sort_values('state')
trans_crosstab.prior_state = pd.Categorical(trans_crosstab.prior_state, categories=['sws','rem','wake'])

#save to csv/pickle
transitions.to_pickle(
    os.path.join(directory_out,'_'.join([plot_pfx,'transitions.pickle'])))





# %%


################################################################################
################################################################################
#############################* CREATE PLOTS BELOW ##############################
################################################################################
################################################################################


# %% #*Specify global plot parametersSpecify global plot parameters

plt_save = True
plt_param = {
    'txt_fam' : p9.element_text(family = 'Arial'),
    'plt_title' : p9.element_text(weight="heavy",margin={'b': 20},size=18),
    'axs_title' : p9.element_text(
        weight="heavy",margin={'r':10,'t':10},size=12,color='black'),
    'strip_text' : p9.element_text(weight="heavy",margin={'b':2,'t':2},size=14,color='firebrick'),
    'fig_size' : (6,5),
    'dpi' : 300,
    'fill' : ['r','mediumblue','darkgray']
}

if not os.path.isdir(os.path.join(directory,'images')):
    os.mkdir(os.path.join(directory,'images'))





# %% #*Plot Signal Strength Histogram

sig_min = 'Cutoff: {y}'.format(y=sig_cutoff)

pdata = summary
p9plot = p9.ggplot(
    data=pdata,
    mapping = p9.aes(
        x = 'SignalStr_mean'
    )) 

p9plot = p9plot \
    + p9.geom_histogram(
        binwidth = 1,
        colour = None,
        fill = 'gray') \
    + p9.geom_vline(xintercept=sig_cutoff,color='red',size=1) \
    + p9.scale_x_continuous(limits=[0,40]) \
    + p9.scale_y_continuous(expand=[0,0]) \
    + p9.labels.ggtitle(': '.join([plot_pfx,'Signal Strength'])) \
    + p9.labels.xlab('Signal Strength') + p9.labels.ylab('Count')\
    + p9.annotate("text",x=0,y=2000,label=sig_min,ha='left') \
    + p9.theme_classic() \
    + p9.theme(
        figure_size = plt_param['fig_size'],
        text = plt_param['txt_fam'],
        plot_title = plt_param['plt_title'],
        axis_title = plt_param['axs_title']
    ) 

if plt_save == True:
    p9plot.save(
        filename= '_'.join([plot_pfx, 'SigStrHist.png']),
        path= os.path.join(directory,'images'),
        dpi = plt_param['dpi'])

p9plot





# %% #*Plot EMG RMS Histogram

pdata = summary

p9plot = p9.ggplot(
    data=pdata,
    mapping = p9.aes(
        x = 'EMG_rms',
        fill = 'state'
    )) 

p9plot = p9plot \
    + p9.geom_histogram(
        binwidth=.1,
        show_legend = False) \
    + p9.scale_y_continuous(expand=[0,0]) \
    + p9.scale_x_continuous(limits = [-3,3]) \
    + p9.labels.ggtitle(': '.join([plot_pfx,'EMG RMS'])) \
    + p9.labels.xlab('EMG RMS(z)') + p9.labels.ylab('Count')\
    + p9.scale_fill_manual(plt_param['fill']) \
    + p9.theme_classic() \
    + p9.theme(
        figure_size = plt_param['fig_size'],
        text = plt_param['txt_fam'],
        plot_title = plt_param['plt_title'],
        axis_title = plt_param['axs_title'],
        strip_text = plt_param['strip_text']
    ) \
    #+ p9.facet_wrap(facets='state',ncol=1) \

if plt_save == True:
    p9plot.save(
        filename= '_'.join([plot_pfx, 'emgHist.png']),
        path= os.path.join(directory,'images'),
        dpi = plt_param['dpi'])

p9plot





# %% #*Plot EEG RMS Histogram

pdata = summary

p9plot = p9.ggplot(
    data=pdata,
    mapping = p9.aes(
        x = 'EEG_rms',
        fill = 'state'
    )) 

p9plot = p9plot \
    + p9.geom_histogram(
        binwidth=.1,
        show_legend = False) \
    + p9.scale_y_continuous(expand=[0,0]) \
    + p9.scale_x_continuous(limits = [-3,3]) \
    + p9.labels.ggtitle(': '.join([plot_pfx,'EEG RMS'])) \
    + p9.labels.xlab('EEG RMS(z)') + p9.labels.ylab('Count')\
    + p9.scale_fill_manual(plt_param['fill']) \
    + p9.theme_classic() \
    + p9.theme(
        figure_size = plt_param['fig_size'],
        text = plt_param['txt_fam'],
        plot_title = plt_param['plt_title'],
        axis_title = plt_param['axs_title'],
        strip_text = plt_param['strip_text']
    ) \
    + p9.facet_wrap(facets='state',ncol=1) \

if plt_save == True:
    p9plot.save(
        filename= '_'.join([plot_pfx, 'eegHist.png']),
        path= os.path.join(directory,'images'),
        dpi = plt_param['dpi'])

p9plot





# %% #*Plot EEG Delta Histogram
 
pdata = summary

p9plot = p9.ggplot(
    data=pdata,
    mapping = p9.aes(
        x = 'EEG_delta',
        fill = 'state'
    )) 

p9plot = p9plot \
    + p9.geom_histogram(
        binwidth=.005,
        show_legend = False) \
    + p9.scale_y_continuous(expand=[0,0]) \
    + p9.scale_x_continuous(limits = [0,1]) \
    + p9.labels.ggtitle(': '.join([plot_pfx,'EEG Delta'])) \
    + p9.labels.xlab('EEG Delta') + p9.labels.ylab('Count')\
    + p9.scale_fill_manual(plt_param['fill']) \
    + p9.theme_classic() \
    + p9.theme(
        figure_size = plt_param['fig_size'],
        text = plt_param['txt_fam'],
        plot_title = plt_param['plt_title'],
        axis_title = plt_param['axs_title'],
        strip_text = plt_param['strip_text']
    ) \
    + p9.facet_wrap(facets='state',ncol=1) \

if plt_save == True:
    p9plot.save(
        filename= '_'.join([plot_pfx, 'deltaHist.png']),
        path= os.path.join(directory,'images'),
        dpi = plt_param['dpi'])

p9plot





# %% #*Plot EEG Theta Histogram

pdata = summary

p9plot = p9.ggplot(
    data=pdata,
    mapping = p9.aes(
        x = 'EEG_theta',
        fill = 'state'
    )) 

p9plot = p9plot \
    + p9.geom_histogram(
        binwidth=.005,
        show_legend = False) \
    + p9.scale_y_continuous(expand=[0,0]) \
    + p9.scale_x_continuous(limits = [0,1]) \
    + p9.labels.ggtitle(': '.join([plot_pfx,'EEG Theta'])) \
    + p9.labels.xlab('EEG Theta') + p9.labels.ylab('Count')\
    + p9.scale_fill_manual(plt_param['fill']) \
    + p9.theme_classic() \
    + p9.theme(
        figure_size = plt_param['fig_size'],
        text = plt_param['txt_fam'],
        plot_title = plt_param['plt_title'],
        axis_title = plt_param['axs_title'],
        strip_text = plt_param['strip_text']
    ) \
    + p9.facet_wrap(facets='state',ncol=1) \

if plt_save == True:
    p9plot.save(
        filename= '_'.join([plot_pfx, 'thetaHist.png']),
        path= os.path.join(directory,'images'),
        dpi = plt_param['dpi'])

p9plot





# %% #*Plot EEG Theta/Delta Histogram
 
pdata = summary

p9plot = p9.ggplot(
    data=pdata,
    mapping = p9.aes(
        x = 'EEG_tdratio',
        fill = 'state'
    )) 

p9plot = p9plot \
    + p9.geom_histogram(
        binwidth=.01,
        show_legend = False) \
    + p9.scale_y_continuous(expand=[0,0]) \
    + p9.scale_x_continuous(limits=[0,np.percentile(pdata.EEG_tdratio,99)]) \
    + p9.labels.ggtitle(': '.join(
        [plot_pfx,'EEG Theta/Delta Ratio'])) \
    + p9.labels.xlab('EEG Theta/Delta') + p9.labels.ylab('Count')\
    + p9.scale_fill_manual(plt_param['fill']) \
    + p9.theme_classic() \
    + p9.theme(
        figure_size = plt_param['fig_size'],
        text = plt_param['txt_fam'],
        plot_title = plt_param['plt_title'],
        axis_title = plt_param['axs_title'],
        strip_text = plt_param['strip_text']
    ) \
    + p9.facet_wrap(facets='state',ncol=1) \

if plt_save == True:
    p9plot.save(
        filename= '_'.join([plot_pfx, 'tdratioHist.png']),
        path= os.path.join(directory,'images'),
        dpi = plt_param['dpi'])

p9plot





# %% #*Plot Correlation Between EMG Signal and Delta During Sleep
 
pdata = summary.iloc[np.arange(0,len(summary),10)]

p9plot = p9.ggplot(
    data=pdata,
    mapping = p9.aes(
        x = 'EMG_rms_medfilt',
        y = 'EEG_tdratio',
        fill = 'state'
    ))

p9plot = p9plot \
    + p9.geom_point(alpha=.5) \
    + p9.scale_y_continuous(
        limits = [0,np.percentile(summary.EEG_tdratio, 99.99)],
        expand=[0,0]) \
    + p9.scale_x_continuous(limits=[-4,4]) \
    + p9.scale_fill_manual(plt_param['fill']) \
    + p9.labels.ggtitle(': '.join(
        [plot_pfx,'Sleep Classification'])) \
    + p9.theme_classic() \
    + p9.theme(
        figure_size = plt_param['fig_size'],
        text = plt_param['txt_fam'],
        plot_title = plt_param['plt_title'],
        axis_title = plt_param['axs_title']
    ) 

if plt_save == True:
    p9plot.save(
        filename= '_'.join([plot_pfx, 'SleepClass.png']),
        path= os.path.join(directory,'images'),
        dpi = plt_param['dpi'])

p9plot



# %% #*Plot Correlation Between EMG Signal and EEG Signal, by day
 
pdata = summary.iloc[np.arange(0,len(summary),100)]

p9plot = p9.ggplot(
    data=pdata,
    mapping = p9.aes(
        x = 'EMG_rms',
        y = 'EEG_rms',
        fill = 'state'
    ))

p9plot = p9plot \
    + p9.geom_point(alpha=.5) \
    + p9.scale_x_continuous(
        limits = [
            np.percentile(summary.EMG_rms, 0),
            np.percentile(summary.EMG_rms, 99.99)],
        expand=[0,0]) \
    + p9.scale_y_continuous(
        limits = [
            np.percentile(summary.EEG_rms, 0),
            np.percentile(summary.EEG_rms, 99.99)],
        expand=[0,0]) \
    + p9.facet_wrap(facets='file',ncol=7) \
    + p9.scale_x_continuous(limits=[-4,4]) \
    + p9.scale_fill_manual(plt_param['fill']) \
    + p9.labels.ggtitle(': '.join(
        [plot_pfx,'EMG-EEG Correlation'])) \
    + p9.theme_classic() \
    + p9.theme(
        figure_size = (6,3),
        text = plt_param['txt_fam'],
        plot_title = plt_param['plt_title'],
        axis_title = plt_param['axs_title']
    ) 

if plt_save == True:
    p9plot.save(
        filename= '_'.join([plot_pfx, 'EMG_EEG_Corr.png']),
        path= os.path.join(directory,'images'),
        dpi = plt_param['dpi'])

p9plot






# %% #*Plot Power Bands
 
pdata = power_summary

p9plot = p9.ggplot(
    data=pdata,
    mapping = p9.aes(
        x = 'pband',
        y = 'power_mean',
        fill = 'state'
    )) 

p9plot = p9plot \
    + p9.geom_bar(
        stat='identity',
        position=p9.position_dodge(width=.9),
        width=.9,
        show_legend=True) \
    + p9.geom_errorbar(
        mapping=p9.aes(
            ymin = 'power_mean - power_sem',
            ymax = 'power_mean + power_sem'),
            position=p9.position_dodge(width=.9),
            width = .5) \
    + p9.labels.ggtitle(': '.join([plot_pfx,'Power Bands'])) \
    + p9.labels.xlab('Power Band') + p9.labels.ylab('Mean Power')\
    + p9.scale_y_continuous(limits=(0, 1),expand=[0,0]) \
    + p9.scale_fill_manual(plt_param['fill']) \
    + p9.theme_classic() \
    + p9.theme(
        figure_size = plt_param['fig_size'],
        text = plt_param['txt_fam'],
        plot_title = plt_param['plt_title'],
        axis_text_x = p9.element_text(size=12,color='black'),
        axis_title = plt_param['axs_title'],
        legend_title = p9.element_blank(),
        legend_text = p9.element_text(size=12,color='black'),
        legend_position = (.8,.8)
    ) 

if plt_save == True:
    p9plot.save(
        filename= '_'.join([plot_pfx, 'pbands.png']),
        path= os.path.join(directory,'images'),
        dpi = plt_param['dpi'])

p9plot





# %% #*Plot Activity by State
 
pdata = summary_means

p9plot = p9.ggplot(
    data=pdata,
    mapping = p9.aes(
        x = 'state',
        y = 'Activity_mean_mean',
        fill = 'state'
    )) 

p9plot = p9plot \
    + p9.geom_errorbar(
        mapping=p9.aes(
            ymin = 'Activity_mean_mean - Activity_mean_sem',
            ymax = 'Activity_mean_mean + Activity_mean_sem'),
            width = .5,
            size = 1) \
    + p9.geom_point(
        stat='identity',
        size=5,
        show_legend=False) \
    + p9.scale_y_continuous(limits=[0,.3],expand=[0,0]) \
        + p9.labels.ggtitle(': '.join([plot_pfx,'Activity'])) \
    + p9.labels.xlab('State') + p9.labels.ylab('Activity (au)')\
    + p9.scale_fill_manual(plt_param['fill']) \
    + p9.theme_classic() \
    + p9.theme(
        figure_size = (2,4),
        text = plt_param['txt_fam'],
        plot_title = plt_param['plt_title'],
        axis_text_x = p9.element_text(size=12,color='black'),
        axis_title = plt_param['axs_title'],
        legend_title = p9.element_blank(),
        legend_text = p9.element_text(size=12,color='black')
    ) 

if plt_save == True:
    p9plot.save(
        filename= '_'.join([plot_pfx, 'activity.png']),
        path= os.path.join(directory,'images'),
        dpi = plt_param['dpi'])

p9plot





# %% #*Plot Temperature by State
 
pdata = summary_means

p9plot = p9.ggplot(
    data=pdata,
    mapping = p9.aes(
        x = 'state',
        y = 'Temp_mean_mean',
        fill = 'state'
    )) 

p9plot = p9plot \
    + p9.geom_errorbar(
        mapping=p9.aes(
            ymin = 'Temp_mean_mean - Temp_mean_sem',
            ymax = 'Temp_mean_mean + Temp_mean_sem'),
            width = .5,
            size = 1) \
    + p9.geom_point(
        stat='identity',
        size=5,
        show_legend=False) \
    + p9.scale_y_continuous(limits=[33,36],expand=[0,0]) \
        + p9.labels.ggtitle(': '.join([plot_pfx,'Temperature'])) \
    + p9.labels.xlab('State') + p9.labels.ylab('Temp (c)')\
    + p9.scale_fill_manual(plt_param['fill']) \
    + p9.theme_classic() \
    + p9.theme(
        figure_size = (2,4),
        text = plt_param['txt_fam'],
        plot_title = plt_param['plt_title'],
        axis_text_x = p9.element_text(size=12,color='black'),
        axis_title = plt_param['axs_title'],
        legend_title = p9.element_blank(),
        legend_text = p9.element_text(size=12,color='black')
    ) 

if plt_save == True:
    p9plot.save(
        filename= '_'.join([plot_pfx, 'temp.png']),
        path= os.path.join(directory,'images'),
        dpi = plt_param['dpi'])

p9plot





# %% #*Plot State Transitions
 
pdata = trans_crosstab

p9plot = p9.ggplot(
    data=pdata,
    mapping = p9.aes(
        x = 'state',
        y = 'number',
        fill = 'prior_state'
    )) 

p9plot = p9plot \
    + p9.geom_bar(stat='identity') \
    + p9.scale_y_continuous(expand=[0,0]) \
    + p9.labels.ggtitle(': '.join([plot_pfx,'State Transitions'])) \
    + p9.labels.xlab('State') + p9.labels.ylab('Prior State')\
    + p9.scale_fill_manual(plt_param['fill']) \
    + p9.theme_classic() \
    + p9.theme(
        figure_size = (2,4),
        text = plt_param['txt_fam'],
        plot_title = plt_param['plt_title'],
        axis_text_x = p9.element_text(size=12,color='black'),
        axis_title = plt_param['axs_title'],
        legend_title = p9.element_blank(),
        legend_text = p9.element_text(size=12,color='black')
    ) 

if plt_save == True:
    p9plot.save(
        filename= '_'.join([plot_pfx, 'state_transitions.png']),
        path= os.path.join(directory,'images'),
        dpi = plt_param['dpi'])

p9plot






