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
from pytz import common_timezones
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





#%% Specify Pickle File with Sleep States by Time Bin

animal = 'ms01'
directory = '/Volumes/csstorage/Zach/ToBeFiled/MSCPpilot/edf/{x}/summary_files'.format(x=animal)


file = '{x}_summary.pickle'.format(x=animal)
pfx = '{x}'.format(x=animal)
df = pd.read_pickle(os.path.join(directory,file))
df.sort_values(by=['tbin'],inplace=True)
df.index = df.tbin





#%% Specify design

#squad 1 (Mice 1-8)
design = pd.DataFrame({
    'day' : [
        1,2,3],
    'date' : [
        '7/13/2021','7/14/2021','7/15/2021'],
    'phase' : [
        'bl','bl','bl'],
    'zstart' : [
        '07:00:00','07:00:00', '07:00:00'] 
})







#%% Create time columns for day and zeithour to summarize data by

df['day'] = np.nan
df['zhour'] = np.nan
df['zsec'] = np.nan
df['ztime'] = np.nan
for id in design.index:
    dt_range = pd.date_range(
        start = " ".join([design.date[id],design.zstart[id]]), 
        periods = 2, freq = 'D')
    df.day[dt_range[0]:dt_range[1]] = design.day[id]
    df.zhour[dt_range[0]:dt_range[1]] = df[dt_range[0]:dt_range[1]].index.shift(
        periods = -1*pd.to_datetime(design.zstart[id]).hour, freq='H').hour
    df.zsec[dt_range[0]:dt_range[1]] = df[dt_range[0]:dt_range[1]].index.shift(
        periods = -1*pd.to_datetime(design.zstart[id]).hour, freq='H').second
    df.ztime = df.zhour + (df.zsec/60)
del dt_range, id

#df.reset_index(inplace=True,drop=True)





#%% Create state summary and write to csv

#create hourly summary by day
summary = pd.crosstab(
    [df.day,df.zhour],df.state.astype('object'), 
    normalize='index').reset_index()
summary = summary.melt(
    id_vars = ['day','zhour'],
    value_vars = ['rem','sws','wake'],
    value_name = 'prop')
summary.state = pd.Categorical(summary.state, categories=['sws','rem','wake'])
summary = pd.merge(design,summary,on='day')
summary['id'] = pfx
summary.phase = pd.Categorical(summary.phase,categories=['pre','post'])
summary = summary[['id'] + summary.columns.to_list()[:-1]]


#save data
summary.to_csv(os.path.join(directory,'_'.join(
    [pfx,'hourlysummary_byDay.csv'])))









# %%

#
#
#
#
#                       CREATE PLOTS BELOW
#
#
#
#

# %% Specify global plot parametersSpecify global plot parameters
#
# 
#
#

plt_save =True
plt_param = {
    'txt_fam' : p9.element_text(family = 'Arial'),
    'plt_title' : p9.element_text(weight="heavy",margin={'b': 20},size=18),
    'axs_title' : p9.element_text(
        weight="heavy",margin={'r':10,'t':10},size=12,color='black'),
    'strip_text' : p9.element_text(weight="heavy",margin={'b':2,'t':2},size=14,color='black'),
    'fig_size' : (6,5),
    'dpi' : 300
}

img_directory = os.path.join(
    os.path.split(directory)[0],
    'images'
)

if not os.path.isdir(img_directory):
    os.mkdir(img_directory)




# %% Sleep states by hour and day plot

pdata = summary

p9plot = p9.ggplot(
    data=pdata,
    mapping = p9.aes(
        x = 'zhour',
        y = 'prop',
        colour = 'state'
    )) 

p9plot = p9plot \
    + p9.geom_rect(
        p9.aes(xmin=12, xmax=23, ymin=0, ymax=1),
        color='black',fill='black') \
    + p9.geom_line(size=1) \
    + p9.facet_wrap('day',ncol=3) \
    + p9.scale_y_continuous(expand=[0,0]) \
    + p9.labels.ggtitle(':\n'.join([pfx,'Sleep State by Day'])) \
    + p9.labels.xlab('Zeitgeist Hour') + p9.labels.ylab('Proportion')\
    + p9.scale_colour_manual(['darkred','seagreen','steelblue']) \
    + p9.theme_classic() \
    + p9.theme(
        figure_size = plt_param['fig_size'],
        text = plt_param['txt_fam'],
        plot_title = plt_param['plt_title'],
        axis_title = plt_param['axs_title'],
        strip_text = plt_param['strip_text']
    ) \

if plt_save == True:
    p9plot.save(
        filename= '_'.join([pfx, 'HourlyState_byDay.png']),
        path= img_directory,
        dpi = plt_param['dpi'])

p9plot






# %% Create State Raster Plot

pdata = df.copy()
pdata = df[df.ztime<25]
pdata.day = pd.Categorical(pdata.day)
pdata['dayidx'] =  (pdata.day.cat.codes)
pdata.dayidx = pdata.dayidx.astype('uint8') + 1
pdata.dayidx = (pdata.dayidx.max()+1) - pdata.dayidx 


p9plot = p9.ggplot(
    data=pdata,
    mapping = p9.aes(
        x = 'ztime',
        y = 'dayidx',
        colour = 'state',
        group = 'day'
    )) 

p9plot = p9plot \
    + p9.geom_vline(xintercept=12) \
    + p9.geom_line(size=4) \
    + p9.coord_cartesian(
        xlim=[0,24],
        ylim=[0,max(pdata.dayidx)+1]) \
    + p9.scale_x_continuous(
        breaks = np.arange(0,25,4),
        limits=[0,24],
        expand=[0,0]) \
    + p9.scale_y_continuous(
        breaks = pdata.dayidx.unique(),
        labels = pdata.day.cat.categories.astype('int'),
        limits=[0,max(pdata.dayidx)+1],
        expand=[0,0]) \
    + p9.scale_colour_manual(['red','blue','lightgrey']) \
    + p9.labels.ggtitle(':\n'.join([pfx,'Sleep Across Time'])) \
    + p9.labels.ylab('Day') \
    + p9.labels.xlab('Zeitgeist Time') \
    + p9.theme_classic() \
    + p9.theme(
        figure_size = (6,2),
        plot_title = p9.element_text(margin={'b':10},size=14),
        axis_ticks_minor = p9.element_blank(),
        axis_ticks_major = p9.element_blank(),
        axis_line_y = p9.element_blank()
    ) \

if plt_save == True:
    p9plot.save(
        filename= '_'.join([pfx, 'Raster.png']),
        path= img_directory,
        dpi = plt_param['dpi'])

p9plot




