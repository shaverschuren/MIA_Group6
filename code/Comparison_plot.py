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
t1d = list()
t1 = list()
t2 = list()
for i in f:
    if i[-5] == 'd':
        t1d.append(i)
    elif i[5] == '1':
        t1.append(i)
    else:
        t2.append(i)

# assigning space
[cor_pbr,cor_pbr2,cor_cc_rig,cor_cc_rig2,cor_cc_af,cor_cc_af2]=[ np.zeros(len(t1)) for i in range(6)]
[MI_pbr,MI_pbr2,MI_cc_rig,MI_cc_rig2,MI_cc_af,MI_cc_af2]= [np.zeros(len(t1)) for i in range(6)]



# executing registration and saving coralation and mutual information
for i in range(1):
    I,Im,It = pbr.pbr(r'..\data\image_data\\'+ t1[i] , r'..\data\image_data\\'+t1d[i])
    cor_pbr[i] = reg.correlation(I,It)
    MI_pbr[i] = reg.mutual_information(reg.joint_histogram(I,It))

for i in range(1):
    I,Im,It = pbr.pbr(r'..\data\image_data\\'+ t1[i] , r'..\data\image_data\\'+t2[i])
    cor_pbr2[i] = reg.correlation(I,It)
    MI_pbr2[i] = reg.mutual_information(reg.joint_histogram(I,It))
print('Point-based: done')



# t1 to t1
for i in range(1):
    sim1, I_cc_rig, Im_cc_rig = proj.intensity_based_registration(r'..\data\image_data\\'+ t1[i] , r'..\data\image_data\\'+t1d[i] , 0 , 0 , 0)
    cor_cc_rig[i] = sim1[-1]
    MI_cc_rig[i] = reg.mutual_information(reg.joint_histogram(I_cc_rig,Im_cc_rig))
    sim2, I_cc_af, Im_cc_af = proj.intensity_based_registration(r'..\data\image_data\\'+ t1[i] , r'..\data\image_data\\'+t1d[i] , 1 , 0 , 0)
    cor_cc_af[i] = sim2[-1]
    MI_cc_af[i] = reg.mutual_information(reg.joint_histogram(I_cc_af,Im_cc_af))
print('T1 to T1: done')

# t1 to t2
for i in range(1):
    sim1, I_cc_rig, Im_cc_rig = proj.intensity_based_registration(r'..\data\image_data\\'+ t1[i] , r'..\data\image_data\\'+t2[i] , 0 , 0 , 0)
    cor_cc_rig2[i] = sim1[-1]
    MI_cc_rig2[i] = reg.mutual_information(reg.joint_histogram(I_cc_rig,Im_cc_rig))
    sim2, I_cc_af, Im_cc_af = proj.intensity_based_registration(r'..\data\image_data\\'+ t1[i] , r'..\data\image_data\\'+t2[i] , 1 , 0 , 0)
    cor_cc_af2[i] = sim2[-1]
    MI_cc_af2[i] = reg.mutual_information(reg.joint_histogram(I_cc_af,Im_cc_af))
print('T1 to T2: done')


#    TODO add mutual information
print(cor_cc_rig)
print(MI_cc_af)


#plotting
plt.close()
#plt.plot(np.arange(1,len(sim1)+1,1),sim1)
fig,ax = plt.subplots()
toplot=[cor_pbr,cor_pbr2,cor_cc_rig,cor_cc_rig2,cor_cc_af,cor_cc_af2]

for i in range(len(toplot)):
    ax.plot((i+1)*np.ones(len(toplot[i])),toplot[i],'k.')
    ax.plot((i+1),np.mean(toplot[i]),'rx')

plt.xlim([0,7])
plt.ylim([0,1])
plt.grid(True)
x_ticks_labels = ['pbr_t1t1','pbr_t1t2','ib_cc_rig_t1t1','ib_cc_rig_t1t2','ib_cc_af_t1t1','ib_cc_af_t1t1']
plt.xticks(np.arange(1,7,1),x_ticks_labels)


plt.show()