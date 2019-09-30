import sys
sys.path.append("../code")
import registration_util as util
import registration as reg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import registration_project as proj
import Point_based_registration as pbr
from os import walk

# extracting and sorting images from dir
for dirpath, dirnames, filenames in walk(r'..\data\image_data'):
    f=filenames
[t1d,t1,t2] = [list() for i in range(3)]
for i in f:
    if i[-5] == 'd':
        t1d.append(i)
    elif i[5] == '1':
        t1.append(i)
    else:
        t2.append(i)

# assigning space
[cor_pbr,cor_pbr2,cor_cc_rig,cor_cc_rig2,cor_cc_af,cor_cc_af2,cor_mi,cor_mi2]=[ np.zeros(len(t1)) for i in range(8)]
[MI_pbr,MI_pbr2,MI_cc_rig,MI_cc_rig2,MI_cc_af,MI_cc_af2,MI_mi,MI_mi2]= [np.zeros(len(t1)) for i in range(8)]

countimages=len(t1)

# executing registration and saving coralation and mutual information
for i in range(countimages):
    I,Im,It = pbr.pbr(r'..\data\image_data\\'+ t1[i] , r'..\data\image_data\\'+t1d[i])
    cor_pbr[i] = reg.correlation(I,It)
    MI_pbr[i] = reg.mutual_information(reg.joint_histogram(I,It))

for i in range(countimages):
    I,Im,It = pbr.pbr(r'..\data\image_data\\'+ t1[i] , r'..\data\image_data\\'+t2[i])
    cor_pbr2[i] = reg.correlation(I,It)
    MI_pbr2[i] = reg.mutual_information(reg.joint_histogram(I,It))
print('Point-based: done')



# t1 to t1
for i in range(countimages):
    sim1, I_cc_rig, Im_cc_rig = proj.intensity_based_registration(r'..\data\image_data\\'+ t1[i] , r'..\data\image_data\\'+t1d[i] , 0 , 0 , 0)
    cor_cc_rig[i] = sim1[-1]
    MI_cc_rig[i] = reg.mutual_information(reg.joint_histogram(I_cc_rig,Im_cc_rig))
    sim2, I_cc_af, Im_cc_af = proj.intensity_based_registration(r'..\data\image_data\\'+ t1[i] , r'..\data\image_data\\'+t1d[i] , 1 , 0 , 0)
    cor_cc_af[i] = sim2[-1]
    MI_cc_af[i] = reg.mutual_information(reg.joint_histogram(I_cc_af,Im_cc_af))
print('CC, T1 to T1: done')

# t1 to t2
for i in range(countimages):
    sim1, I_cc_rig, Im_cc_rig = proj.intensity_based_registration(r'..\data\image_data\\'+ t1[i] , r'..\data\image_data\\'+t2[i] , 0 , 0 , 0)
    cor_cc_rig2[i] = sim1[-1]
    MI_cc_rig2[i] = reg.mutual_information(reg.joint_histogram(I_cc_rig,Im_cc_rig))
    sim2, I_cc_af, Im_cc_af = proj.intensity_based_registration(r'..\data\image_data\\'+ t1[i] , r'..\data\image_data\\'+t2[i] , 1 , 0 , 0)
    cor_cc_af2[i] = sim2[-1]
    MI_cc_af2[i] = reg.mutual_information(reg.joint_histogram(I_cc_af,Im_cc_af))
print('CC, T1 to T2: done')

# t1 to t2
for i in range(countimages):
    sim1, I_mi, Im_mi = proj.intensity_based_registration(r'..\data\image_data\\'+ t1[i] , r'..\data\image_data\\'+t1d[i] , 1 , 1 , 0)
    cor_mi[i] = reg.correlation(I_mi,Im_mi)
    MI_mi[i] = sim1[-1]
    sim2, I_mi2, Im_mi2 = proj.intensity_based_registration(r'..\data\image_data\\'+ t1[i] , r'..\data\image_data\\'+t2[i] , 1 , 1 , 0)
    cor_mi2[i] = reg.correlation(I_mi2,Im_mi2)
    MI_mi2[i] = sim2[-1]
    print(i)
print('MI, T1 to T2: done')


#plotting
plt.close()
#plt.plot(np.arange(1,len(sim1)+1,1),sim1)
fig1,ax1 = plt.subplots()
toplot_cor=[cor_pbr,cor_pbr2,cor_cc_rig,cor_cc_rig2,cor_cc_af,cor_cc_af2,cor_mi,cor_mi2]

for i in range(len(toplot_cor)):
    ax1.plot((i+1)*np.ones(len(toplot_cor[i])),toplot_cor[i],'k.')
    ax1.plot((i+1),np.mean(toplot_cor[i]),'rx')

plt.xlim([0,len(toplot_cor)+1])
plt.ylim([0,1])
plt.grid(True)
plt.title('Cross-correlation')
plt.ylabel('Cross-correlation after registration')
x_ticks_labels = ['pbr_t1t1','pbr_t1t2','ib_cc_rig_t1t1','ib_cc_rig_t1t2','ib_cc_af_t1t1','ib_cc_af_t1t1','cor_mi_t1t1','cor_mi_t1t2']
plt.xticks(np.arange(1,len(toplot_cor)+1,1),x_ticks_labels)

fig2,ax2 = plt.subplots()
toplot_MI = [MI_pbr,MI_pbr2,MI_cc_rig,MI_cc_rig2,MI_cc_af,MI_cc_af2,MI_mi,MI_mi2]
for i in range(len(toplot_MI)):
    ax2.plot((i+1)*np.ones(len(toplot_MI[i])),toplot_MI[i],'k.')
    ax2.plot((i+1),np.mean(toplot_MI[i]),'rx')

plt.xlim([0,len(toplot_MI)+1])
plt.ylim([0,1])
plt.grid(True)
plt.title('Mutual Information')
plt.ylabel('Mutual information after registration')
x_ticks_labels2 = ['pbr_t1t1','pbr_t1t2','ib_cc_rig_t1t1','ib_cc_rig_t1t2','ib_cc_af_t1t1','ib_cc_af_t1t1','MI_mi_t1t1','MI_mi_t1t2']
plt.xticks(np.arange(1,len(toplot_MI)+1,1),x_ticks_labels2)

plt.show()