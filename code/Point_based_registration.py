

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
import registration_util as util
from IPython.display import display, clear_output

def pbr(I_path,Im_path):
    I=plt.imread(I_path)
    Im=plt.imread(Im_path)
    X, Xm = util.my_cpselect(I_path, Im_path)
    if len(X[1])==1:
        print('x must be larger than 1')
        return
    X=util.c2h(X)
    Xm=util.c2h(Xm)
    T=reg.ls_affine(X,Xm)
    It, Xt = reg.image_transform(Im,T)

    fig = plt.figure(figsize=(30, 30))

    ax1 = fig.add_subplot(121)
    ax1.set_title('Overlay of original images',fontsize=35)
    ax1.axis('off')
    im11 = ax1.imshow(I)
    im12 = ax1.imshow(Im, alpha=0.7)
    
    
    ax2 = fig.add_subplot(122)
    ax2.set_title('Overlay transformed image over the fixed image',fontsize=35)
    ax2.axis('off')   
    im21 = ax2.imshow(I)
    im22 = ax2.imshow(It, alpha=0.7)
    return I,Im,It

