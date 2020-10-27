# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:27:17 2020

@author: 71020
"""
def MAE(A,B):
    MAE_temp = 0
    for kk in range(0, len(A)):
        MAE_temp += abs(A[kk]-B[kk])/len(A)
    return MAE_temp

def RMSE(A,B):
    MAE_temp = 0
    for kk in range(0, len(A)):
        MAE_temp += ((A[kk]-B[kk])**2)/len(A)
    MAE_temp = np.sqrt(MAE_temp)
    return MAE_temp

import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm
import seaborn as sns
temp = pd.read_csv('201022_result.csv')

time  = temp.values[:,0]
label = temp.values[:,4]
labela = temp.values[:,8]

Rv = temp.values[:,1]
PwC = temp.values[:,9]
PwoC = temp.values[:,10]



#%%
St = 805; print(time[St])  # Start time
len_win = 13 # window size
Real_len = np.zeros([len(Rv),])
Real_len[St:St+len_win]=1



Err_len = (Real_len==1)&(label>=2)
cum_len = np.cumsum(Err_len)
cum_Err = np.where(cum_len==1)[0] - St

if label[cum_Err]==3:
    LI_wc = np.zeros([int(np.max(cum_len)),])
    for ii in range(0, int(np.max(cum_len))):
        LI_wc[ii] = Rv[int(cum_Err)+St-1]+(Rv[int(cum_Err)+St+np.max(cum_len)]-Rv[int(cum_Err)+St-1])*(ii+1)/int(np.max(cum_len)+1)
else:
    LI_wc = np.zeros([int(np.max(cum_len)),])
    for ii in range(0, int(np.max(cum_len))):
        LI_wc[ii] = Rv[int(cum_Err)+St]+(Rv[int(cum_Err)+St+np.max(cum_len)]-Rv[int(cum_Err)+St])*(ii)/int(np.max(cum_len))
    
# Linear interpolration

plt.figure(figsize = (6,6), dpi=400)
plt.plot(range(0,len_win), Rv[Real_len==1], '-bx', linewidth=1, markersize=12)

plt.plot(range(int(cum_Err),int(cum_Err+np.max(cum_len))),PwoC[(Real_len==1)&(label>=2)], '-mv', linewidth=1, markersize=12)
plt.plot(range(int(cum_Err),int(cum_Err+np.max(cum_len))),LI_wc, '-cP', linewidth=1, markersize=12)
plt.plot(range(int(cum_Err),int(cum_Err+np.max(cum_len))),PwC[(Real_len==1)&(label>=2)], '-rd', linewidth=1, markersize=12)

plt.plot([int(cum_Err), int(cum_Err)],[0, 100], '--k', linewidth=.3)
plt.plot([int(cum_Err+np.max(cum_len)-1), int(cum_Err+np.max(cum_len)-1)],[0, 100], '--k', linewidth=.3)

plt.ylim([0,1.0])
plt.xticks([0,6,12,18,24])
plt.xlim([0,len_win-1])
plt.xlabel('Time [h]')
plt.ylabel('Power [kW]')
font = {'family' : 'normal', 'weight' : 'normal', 'size': 18}
plt.rc('font', **font)
plt.legend(['Actual data','AR w/o const.','LI w/ const.','AR w/ const.'])
plt.tight_layout()
plt.savefig('Fig_(a).pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

#%%
St = 330; print(time[St])  # Start time
len_win = 13 # window size
Real_len = np.zeros([len(Rv),])
Real_len[St:St+len_win]=1



Err_len = (Real_len==1)&(label>=2)
cum_len = np.cumsum(Err_len)
cum_Err = np.where(cum_len==1)[0] - St

LI_wc = np.zeros([int(np.max(cum_len)),])
for ii in range(0, int(np.max(cum_len))):
    LI_wc[ii] = Rv[int(cum_Err)+St-1]+(Rv[int(cum_Err)+St+np.max(cum_len)]-Rv[int(cum_Err)+St-1])*(ii+1)/int(np.max(cum_len)+1)

# Linear interpolration

plt.figure(figsize = (6,6), dpi=400)
plt.plot(range(0,len_win), Rv[Real_len==1], '-bx', linewidth=1, markersize=12)

plt.plot(range(int(cum_Err),int(cum_Err+np.max(cum_len))),PwoC[(Real_len==1)&(label>=2)], '-mv', linewidth=1, markersize=12)
plt.plot(range(int(cum_Err),int(cum_Err+np.max(cum_len))),LI_wc, '-cP', linewidth=1, markersize=12)
plt.plot(range(int(cum_Err),int(cum_Err+np.max(cum_len))),PwC[(Real_len==1)&(label>=2)], '-rd', linewidth=1, markersize=12)
plt.plot([int(cum_Err), int(cum_Err)],
         [PwoC[(Real_len==1)&(label>=2)][0], PwoC[(Real_len==1)&(label>=2)][0]], 'ks', linewidth=.7, markersize=12)

plt.plot([int(cum_Err), int(cum_Err)],[0, 100], '--k', linewidth=.3)
plt.plot([int(cum_Err+np.max(cum_len)-1), int(cum_Err+np.max(cum_len)-1)],[0, 100], '--k', linewidth=.3)

plt.ylim([0,1.5])
plt.xticks([0,6,12,18,24])
plt.xlim([0,len_win-1])
plt.xlabel('Time [h]')
plt.ylabel('Power [kW]')
font = {'family' : 'normal', 'weight' : 'normal', 'size': 18}
plt.rc('font', **font)
plt.legend(['Actual data','AR w/o const.','LI w/ const.','AR w/ const.', 'Outlier'])
plt.tight_layout()
plt.savefig('Fig_(b).pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)


#%%
TT = 4

A42  = (label[:len(label)-4]==4)&(label[1:len(label)-3]==2)&(label[2:len(label)-2]==2)&(label[3:len(label)-1]==2)
A32  = (label[:len(label)-4]==3)&(label[1:len(label)-3]==2)&(label[2:len(label)-2]==2)&(label[3:len(label)-1]==2)
A42_idx = np.where(A42==1)
A32_idx = np.where(A32==1)

MAE_42 = np.zeros([3, np.sum(A42)])
MAE_32 = np.zeros([3, np.sum(A32)])

RMSE_42 = np.zeros([3, np.sum(A42)])
RMSE_32 = np.zeros([3, np.sum(A32)])

for ii in range(0, np.sum(A42)):
    MAE_42[0,ii] = MAE(Rv[A42_idx[0][ii]:A42_idx[0][ii]+TT], PwoC[A42_idx[0][ii]:A42_idx[0][ii]+TT]) 
    
    
    LI_wc = np.zeros([TT,])
    for jj in range(0, TT):
        LI_wc[jj] = Rv[A42_idx[0][ii]-1]+(jj+1)*(Rv[A42_idx[0][ii]+TT] - Rv[A42_idx[0][ii]-1])/(TT+1)
    MAE_42[1,ii] = MAE(Rv[A42_idx[0][ii]:A42_idx[0][ii]+TT], LI_wc)
    
    MAE_42[2,ii] = MAE(Rv[A42_idx[0][ii]:A42_idx[0][ii]+TT], PwC[A42_idx[0][ii]:A42_idx[0][ii]+TT])
    
    if np.isnan(np.sum(LI_wc)):
        MAE_42[:,ii] = Rv[1]
        
    RMSE_42[0,ii] = RMSE(Rv[A42_idx[0][ii]:A42_idx[0][ii]+TT], PwoC[A42_idx[0][ii]:A42_idx[0][ii]+TT]) 
      
    LI_wc = np.zeros([TT,])
    for jj in range(0, TT):
        LI_wc[jj] = Rv[A42_idx[0][ii]-1]+(jj+1)*(Rv[A42_idx[0][ii]+TT] - Rv[A42_idx[0][ii]-1])/(TT+1)
    RMSE_42[1,ii] = RMSE(Rv[A42_idx[0][ii]:A42_idx[0][ii]+TT], LI_wc)
    
    RMSE_42[2,ii] = RMSE(Rv[A42_idx[0][ii]:A42_idx[0][ii]+TT], PwC[A42_idx[0][ii]:A42_idx[0][ii]+TT])
    
    if np.isnan(np.sum(LI_wc)):
        RMSE_42[:,ii] = Rv[1]
#%%
for ii in range(0, np.sum(A32)):
    MAE_32[0,ii] = MAE(Rv[A32_idx[0][ii]:A32_idx[0][ii]+TT], PwoC[A32_idx[0][ii]:A32_idx[0][ii]+TT])
    
    LI_wc = np.zeros([TT,])
    for jj in range(0, TT):
        LI_wc[jj] = Rv[A32_idx[0][ii]]+jj*(Rv[A32_idx[0][ii]+TT] - Rv[A32_idx[0][ii]])/(TT)
    MAE_32[1,ii] = MAE(Rv[A32_idx[0][ii]:A32_idx[0][ii]+TT], LI_wc)
    if np.isnan(np.sum(LI_wc)):
        MAE_32[:,ii] = Rv[1]
    
    MAE_32[2,ii] = MAE(Rv[A32_idx[0][ii]:A32_idx[0][ii]+TT], PwC[A32_idx[0][ii]:A32_idx[0][ii]+TT])
    if np.isnan(np.sum(LI_wc)):
        MAE_32[:,ii] = Rv[1]      
        
    RMSE_32[0,ii] = RMSE(Rv[A32_idx[0][ii]:A32_idx[0][ii]+TT], PwoC[A32_idx[0][ii]:A32_idx[0][ii]+TT])
    
    LI_wc = np.zeros([TT,])
    for jj in range(0, TT):
        LI_wc[jj] = Rv[A32_idx[0][ii]]+jj*(Rv[A32_idx[0][ii]+TT] - Rv[A32_idx[0][ii]])/(TT)
    RMSE_32[1,ii] = RMSE(Rv[A32_idx[0][ii]:A32_idx[0][ii]+TT], LI_wc)
    if np.isnan(np.sum(LI_wc)):
        MAE_32[:,ii] = Rv[1]
    
    RMSE_32[2,ii] = RMSE(Rv[A32_idx[0][ii]:A32_idx[0][ii]+TT], PwC[A32_idx[0][ii]:A32_idx[0][ii]+TT])
    if np.isnan(np.sum(LI_wc)):
        RMSE_32[:,ii] = Rv[1]          
        
hfont = {'fontname':'Helvetica'}
plt.figure(figsize = (4,4), dpi=400)
barlist = plt.bar(['AR w/o const.','LI w/ const.','AR w/ const.'], np.nanmean(MAE_42,axis=1), width=0.5)
barlist[0].set_color('r')
barlist[1].set_color('b')
barlist[2].set_color('g')
plt.ylabel('MAE [kW]', **hfont)
plt.rcParams["font.family"] = "Helvetica"
plt.savefig('Fig_MAE (b).pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)


hfont = {'fontname':'Helvetica'}
plt.figure(figsize = (4,4), dpi=400)
barlist = plt.bar(['AR w/o const.','LI w/ const.','AR w/ const.'], np.nanmean(MAE_32,axis=1), width=0.5)
barlist[0].set_color('r')
barlist[1].set_color('b')
barlist[2].set_color('g')
plt.ylabel('MAE [kW]', **hfont)
plt.rcParams["font.family"] = "Helvetica"
plt.savefig('Fig_MAE (b).pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

#%%

yy = pd.DataFrame(MAE_42.T, columns=['AR w/o const.','LI w/ const.','AR w/ const.'])

hfont = {'fontname':'Helvetica'}
plt.figure(figsize = (6,6), dpi=400)
sns.set(style="ticks", palette='Set2', font='Helvetica')
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 1.1})
g=sns.factorplot(data=yy, kind="box", size=7, aspect=0.6,
                 width=.8,fliersize=2.5, linewidth=1.1, notch=False, orient="v")
plt.ylim([0,3])
plt.ylabel('MAE [kW]', **hfont)
plt.xticks(rotation=45)
plt.rcParams["font.family"] = "Helvetica"
plt.tight_layout()
plt.savefig('Fig_MAE (a).pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

yy = pd.DataFrame(MAE_32.T, columns=['AR w/o const.','LI w/ const.','AR w/ const.'])


hfont = {'fontname':'Helvetica'}
plt.figure(figsize = (6,6), dpi=400)
sns.set(style="ticks", palette='Set2', font='Helvetica')
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 1.1})
g=sns.factorplot(data=yy, kind="box", size=7, aspect=0.6,
                 width=.8,fliersize=2.5, linewidth=1.1, notch=False, orient="v")
plt.ylabel('MAE [kW]', **hfont)
plt.rcParams["font.family"] = "Helvetica"
plt.ylim([0,1])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Fig_MAE (b).pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)


yy = pd.DataFrame(RMSE_42.T, columns=['AR w/o const.','LI w/ const.','AR w/ const.'])

hfont = {'fontname':'Helvetica'}
plt.figure(figsize = (6,6), dpi=400)
sns.set(style="ticks", palette='Set2', font='Helvetica')
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 1.1})
g=sns.factorplot(data=yy, kind="box", size=7, aspect=0.6,
                 width=.8,fliersize=2.5, linewidth=1.1, notch=False, orient="v")
plt.ylim([0,4])
plt.ylabel('RMSE [kW]', **hfont)
plt.xticks(rotation=45)
plt.rcParams["font.family"] = "Helvetica"
plt.tight_layout()
plt.savefig('Fig_RMSE (a).pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

yy = pd.DataFrame(RMSE_32.T, columns=['AR w/o const.','LI w/ const.','AR w/ const.'])


hfont = {'fontname':'Helvetica'}
plt.figure(figsize = (6,6), dpi=400)
sns.set(style="ticks", palette='Set2', font='Helvetica')
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 1.1})
g=sns.factorplot(data=yy, kind="box", size=7, aspect=0.6,
                 width=.8,fliersize=2.5, linewidth=1.1, notch=False, orient="v")
plt.ylabel('RMSE [kW]', **hfont)
plt.rcParams["font.family"] = "Helvetica"
plt.ylim([0,1])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Fig_RMSE (b).pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='pdf',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

#%%
print(np.nanmean(MAE_32,axis=1))
print(np.nanmean(MAE_42,axis=1))
print(np.nanmean(MAE_32,axis=1)/2+np.nanmean(MAE_42,axis=1)/2)

print(np.nanmean(RMSE_32,axis=1))
print(np.nanmean(RMSE_42,axis=1))
print(np.nanmean(RMSE_32,axis=1)/2+np.nanmean(RMSE_42,axis=1)/2)


print(np.sum((label<4)&(labela==4))/(np.sum((label==4)&(labela==4))+np.sum((label<4)&(labela==4))))
print(np.sum((label==4)&(labela<4))/(np.sum((label==4))))



