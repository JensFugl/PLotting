# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:14:15 2020

@author: Jens Ringsholm
"""
    

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functions import fitexp, fit2exp, fitLin, fit2dpol, fit3dpol, OpenVolumeData, plot_Brew

from scipy import stats
from datetime import timedelta


plt.close('all')


########## define aestetics ################################# 

import seaborn as sns
sns.set_style("darkgrid" )
sns.set(rc={'axes.facecolor':'#eef1f4',})
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.color_palette("husl", 10)


figS = [12,7]

inspect = False


plot_devices = True

plot_variables = False



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functions import fitexp, fit2exp, fitLin, fit2dpol, fit3dpol, OpenVolumeData, plot_Brew

from scipy import stats
from datetime import timedelta


plt.close('all')


########## define aestetics ################################# 

import seaborn as sns

sns.set_style("darkgrid" )
sns.set(rc={'axes.facecolor':'#eef1f4',})
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.color_palette("husl", 10)



figS = [12,7]

inspect = True


plot_devices = False

plot_variables = False


################# Import and clean FA ############################

#################################################################

pp = pd.read_csv('data/Raw_data4.csv', index_col = 'rt', parse_dates=True)
BIR = pd.read_csv('data/Raw_data7.csv', index_col = 'rt', parse_dates=True)


############ remove noise with z-score #####################

pp =pp[(np.abs(stats.zscore(pp)) < 10).all(axis=1)]
BIR =BIR[(np.abs(stats.zscore(BIR)) < 30).all(axis=1)]

pp['d2/d1'] = pp['d2']/pp['d1']



################Remove unwanted columns ###########

pp = pp.drop(columns=['temp_ext', 'temp_ds18', 'tem_scd30', 'epoch', 'd1', 'd2', 'co2'])
BIR = BIR.drop(columns=['temp_ms5803', 'temp_ds18', 'tem_scd30', 'epoch', 'd1', 'd2', 'pres'])




############# test if any duplicates

#df6 = df6[df6.index.duplicated()]


############## adjust time #################################

BIR = BIR.tz_localize(None)
pp = pp.tz_localize(None)

#df4.index = df4.index + timedelta(hours=0)

############# Plot the data for inspection ################### 
if inspect == True:
    fig2, axes = plt.subplots(nrows=1, ncols=2, figsize = (10, 8))
    BIR.plot(subplots=True, ax=axes)
    plt.tight_layout()

    fig1, axes = plt.subplots(nrows=1, ncols=3, figsize = (10, 8))
    pp.plot(subplots=True, ax=axes)
    plt.tight_layout()
    
    fig1.savefig('figs\BIR_view.png')
    fig2.savefig('figs\pp_view.png')


################## Import and clean SG_tilt ######################

##################################################################

tilt = pd.read_csv('data/tilt.csv', index_col = 'Timepoint', parse_dates=True)

# remove empty 
tilt = tilt.dropna()


# align and cut time
tilt.index = tilt.index + timedelta(hours=-2)

# remove noise in the beginning 
#tilt = tilt[:'2020-04-14 17:45:00']

######### plot for inspection ################################

if inspect == True:
    fig2, axes = plt.subplots(nrows=2, ncols=1, figsize = (10, 8))
    tilt.plot(subplots=True, ax=axes)
    plt.tight_layout()
    fig2.savefig('figs\Tilt_view.png')

################## Import and clean SG_Platoo ######################

pla = pd.read_csv('data/Plaato_SG.csv',index_col = 'epoch', parse_dates=True)

pla.index=(pd.to_datetime(pla.index,unit='ms'))

BPM = pd.read_csv('data/Plaato_BPM.csv',index_col = 'epoch', parse_dates=True)

BPM.index=(pd.to_datetime(BPM.index,unit='ms'))

Ptemp = pd.read_csv('data/Plaato_Temp.csv',index_col = 'epoch', parse_dates=True)

Ptemp.index=(pd.to_datetime(Ptemp.index,unit='ms'))


# refine and create one df

pla  = pla.drop(columns=['lol'])
pla['BPM'] = BPM['BPM']
pla['Temp'] = Ptemp['Temp']



# cut data

pla = pla['2020-04-21 05:00:00':]


'''
################# correct dicontinuity #########################

mask = pla.index[259:]

pla.update(pla.loc[mask, 'SG'] + 0.044)

#################### calculate and add ABV

tilt['ABV-tilt'] = (tilt['SG'].max()-tilt['SG'])*131.25 
pla['ABV-plaat'] = (pla['SG'].max()-pla['SG'])*131.25 


##### fancy formula 

#tilt['ABV-tilt2'] = ((76.08*(tilt['SG'].max()-tilt['SG']))/(1.775-tilt['SG'].max()))*(tilt['SG']/0.794) 


'''
############### plot for inspection 
if inspect == True:
    fig3, axes = plt.subplots(nrows=3, ncols=1, figsize = (10, 8))
    pla.plot(subplots=True, ax=axes)
    plt.tight_layout()
    #fig3.savefig('figs\Plaato_view.png')
'''



############# Plot Function #################################################

def plot_multi(data,  title=None, cols=None, **kwargs):

    from pandas import plotting

    # Get default color style from pandas
    if cols is None: cols = data.columns
    save = True
    if cols is None: save = False
    if len(cols) == 0: return
    colors = getattr(getattr(plotting, '_matplotlib').style, '_get_standard_colors')(num_colors=len(cols))
    
    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.tick_params(axis='y', colors=colors[0 % len(colors)])
    plt.title(title)
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right']
        data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
        ax_new.tick_params(axis='y', colors=colors[n % len(colors)])
        ax_new.grid(None)
        
        # Legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc='best')
    plt.tight_layout()
    if save == True:
        plt.savefig('figs\{}.png'.format(title))
    return ax



def sameax_plot(data, title, save, size=None):
    
    if size == None: size = (10, 6)
    
    fig1, ax = plt.subplots(figsize=size)
    ax.plot(data)
    plt.title(title)
    plt.legend(data.columns)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    if save == True:
            plt.savefig('figs\{}.png'.format(title))

    return

########################## plot Devices ############################# 

df4_plot = df4.drop(columns= ['co2', 'rol'])

if plot_devices == True:
    fig1, ax = plt.subplots(figsize=(12,7)) 
    ax = plot_multi(tilt, title = 'Tilt Plot')
    
    
    fig2, ax2 = plt.subplots(figsize=(12,7)) 
    ax2 = plot_multi(pla, title='Plaato plot')
    
    

    
    fig3, ax3 = plt.subplots(figsize=(12,7)) 
    ax3 = plot_multi(df4_plot, title= 'BIR Plot')





######################## Resampling and Intepolation ################################

li = []

pla['SG-plaat'] = pla['SG']
pla['Temp-plaat'] = pla['Temp']

tilt ['SG-tilt'] = tilt['SG']
tilt ['Temp-tilt'] = tilt['Temp']

pla = pla.drop(columns=['SG', 'Temp'])
tilt = tilt.drop(columns=['SG', 'Temp'])


li.append(pla)
li.append(df4_plot)
li.append(tilt)

df_mv = pd.concat(li, axis=0, ignore_index=False, sort=True)

df_int = df_mv.resample('1min').mean().interpolate().dropna()


fig11, axes = plt.subplots(nrows=5, ncols=2, figsize = (14, 9))
df_int.plot(subplots=True, ax=axes)
plt.tight_layout()
plt.savefig('figs\InterpolatedOverview.png')

################## plot each variable together ##################

if plot_variables == True:

    #### temp
    
    df_temp = df_int.drop(columns=['pres', 'd2/d1', 'SG-tilt', 'SG-plaat', 'BPM'])
    
    sameax_plot(df_temp, 'Temp Plot', False, (12,7) )
    plt.savefig('figs\Temp.png')    
    ######### SG
    
    df_SG = df_int.drop(columns=['pres', 'temp_ext', 'Temp-tilt', 'Temp-plaat', 'BPM', 'd2/d1'])
    
    sameax_plot(df_SG, 'SG plot', False , (12,7))
    plt.savefig('figs\SG.png')



################# Correlation matrix ######################3

from pandas.plotting import scatter_matrix

axes = scatter_matrix(df_int, alpha=0.9, diagonal='hist')
corr = df_int.corr().as_matrix()

for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes[i, j].annotate("%.3f" %corr[i,j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')
plt.show()

###################### correlation matrix 2##################


f = plt.figure(figsize=(10, 8))
plt.matshow(df_int.corr(), fignum=f.number, cmap='rocket')
plt.xticks(range(df_int.shape[1]), df_int.columns, fontsize=14, rotation=45, )
plt.yticks(range(df_int.shape[1]), df_int.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.savefig('figs\correlationMatrix.png')




fit2dpol(df_int['d2/d1'][100:2400], df_int['SG-plaat'][100:2400], 1, False, 'd2/d1 vs SG plaato', (10,8)) 

fit2dpol(df_int['d2/d1'][100:2000], df_int['pres'][100:2000], 1, False, 'd2/d1 vs pressure', (10,8)) 



fitLin(df_int['SG-plaat'], df_int['SG-tilt'], 1, True, 'SG Plaato vs Tilt', (10,8)) 

fitLin(df_int['SG-plaat'],df_int['BPM'].expanding(1).sum().max()-df_int['BPM'].expanding(1).sum(),  1, True, ' Plaato SG vs BPM roling sum', (10,8)) 


from pandas_profiling import ProfileReport
prof = ProfileReport(df_mv)
prof.to_file(output_file='Pandasprofile_raw.html')
'''
