# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:39:53 2020

@author: Jens Ringsholm
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from datetime import timedelta
import datetime
import matplotlib.ticker
from matplotlib.ticker import FormatStrFormatter

plt.close('all')


import matplotlib.style as style
style.use('fivethirtyeight')

# Styles

#https://matplotlib.org/3.2.1/gallery/style_sheets/style_sheets_reference.html


####################### import file ########################################################

df_int = pd.read_csv('data/df_int.csv', index_col='dates', parse_dates=True)



#################### calculate extra parameters

########### ABV

df_int['ABV'] = (df_int['SG-tilt'].max()-df_int['SG-tilt'])*131

########### CO2 dissolved




############# timeindex function

def datetimeindex_to_relative_timeindex(df, interval=None):
    
    '''
    changes pandas dataframe index from datetime to time relative to experiement start
    '''
    
    df_conv = df
    
    df_conv.index = df_conv.index.astype(np.int64) // 10**9 - (df_conv.index.astype(np.int64).min() // 10**9)

    if interval == None or interval == 's':
        pass
    if interval == 'm':
        df_conv.index = df_conv.index/60 
    
    if interval == 'h':
        df_conv.index = df_conv.index/360 
    
    if interval == 'd':
        df_conv.index = (df_conv.index/360)/24        
    
    return df_conv



###################### Plot functions ########################


def plot_multi(data, time=None,  title=None, cols=None, color=None, color2=None, 
               labelsize=None, nticks=None, pad=None, dec=None, legend=None, grid=None,
               split=None, smooth = None, **kwargs):

    '''
    !!! data , dec and color must have the same length !!!
    
    Plots chosen cols from df against time in the same plot with different axes
    color 0 will be used on plot 0 and so forth.
    plot 0 is to the left axes and subsequent plots will go on the right axis. 
    '''


    # Check and get columns
    if cols is None: cols = data.columns
    save = True
    if cols is None: save = False
    if len(cols) == 0: return
    
    colors = color
    colors2 = color2
    
    # Create first axis
 
    ax = data.loc[:, cols[0]][:split].plot(label=cols[0], color=colors[0], **kwargs)


    # create prediction by smoothing
    if smooth == None: smooth = 30
    ax = data.loc[:, cols[0]][split:].rolling(smooth).mean().plot(color=colors2[0], **kwargs)
    
    ax.tick_params(axis='y', colors=colors[0], labelsize = labelsize )
    # Define number of y-tics
    if nticks == None:
        nticks = 6
    else:
        nticks = nticks
    ax.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
    
    # Format first axis
    form = '%.{}f'.format(dec[0])
    ax.yaxis.set_major_formatter(FormatStrFormatter(form))
    
    
    plt.title(title)
    lines, labels = ax.get_legend_handles_labels()
    
    # define background
    ax.set_facecolor(bg)
    ax.grid(grid)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


    for n in range(1, len(cols)):
        # Create multiple plots on right y-axes
        ax_new = ax.twinx()
        ax_new.spines['right']
        data.loc[:, cols[n]][:split].plot(ax=ax_new, label=cols[n], color=colors[n], **kwargs)
        
        # create prediction by smoothing
        data.loc[:, cols[n]][split:].rolling(smooth).mean().plot(ax=ax_new, color=colors2[n], **kwargs)
        
        ax_new.tick_params(axis='y', colors=colors[n])
        ax_new.grid(None)
        
        # get number of y-ticks
        ax_new.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
        
        # get number of decimals for each parameter
        if dec == None:
            form = 1
        else:
            form = '%.{}f'.format(dec[n])
        ax_new.yaxis.set_major_formatter(FormatStrFormatter(form))
        
        # avoid overlap between y-labels + size
        ax_new.tick_params(axis = 'both', pad = pad*(n-1), which = 'major', labelsize = labelsize)
        
        ax_new.set_facecolor(bg)
        
        ax_new.spines['right'].set_visible(False)
        ax_new.spines['left'].set_visible(False)
        ax_new.spines['top'].set_visible(False)
        ax_new.spines['bottom'].set_visible(False)
        
        # Define legend and position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    if legend == True:        
        ax.legend(lines, labels, loc='center')
        
    
    
    
    ax.set_xlabel('Days', fontsize=labelsize)
    
    plt.tight_layout()

    if save == True:
        plt.savefig('figs\{}.png'.format(title))

    return ax


########### All parameters
    
######## set colorschemes and backgroundcolor


Colorscheme1 = ['#BF9000', '#456990', '#548235', '#974141']
                
Colorscheme2 = ['#EAB200', '#4674C6', '#6CA644' ,'#BF6969' ]
                
bg = '#eef1f4'
#bg = 'black'


########## define timeindex
### choose between s , m , h and d

df_sec = datetimeindex_to_relative_timeindex(df_int, 'd')


######### Plot multi-figure

fig, axes = plt.subplots(figsize = (10, 8))

ax = plot_multi(df_int,
                cols=['co2', 'SG-tilt', 'temp_ext', 'ABV'], 
                title= 'All plot', 
                color = Colorscheme1,
                color2 = Colorscheme2,
                labelsize = 18,
                nticks = 8,
                pad = 60,
                dec = [0,3,1,1],
                legend = False,
                grid = True,
                split = .001,
                smooth = 100
                )




fig.set_facecolor(bg)                                                
#plt.gcf().autofmt_xdate()
plt.tight_layout()



