"""
Code used for the plots used for comparison of pbr and ibr.
Main author:    R.J.P. van Bergen
                28/09/2019
"""

import numpy as np
import matplotlib.pyplot as plt
import registration as reg
from IPython.display import display, clear_output
import registration_project as pro

def plotcor():
    I_path = '../data/image_data/1_1_t1.tif'
    Im_path = '../data/image_data/1_1_t1_d.tif'
    Im_path_t2 = '../data/image_data/1_1_t2.tif'
    ex1=pro.intensity_based_registration(I_path, Im_path, 0, 0, 0)
    ex2=pro.intensity_based_registration(I_path, Im_path, 1, 0, 0)
    ex3=pro.intensity_based_registration(I_path, Im_path_t2, 1, 0, 0)
    num_it = 50
    it_vec = np.arange(1, num_it+1)

    plt.figure(figsize=(10, 10))
    plt.grid()
    plt.plot(it_vec, ex1, 'k')
    plt.plot(it_vec, ex2)
    plt.plot(it_vec, ex3)
    plt.xlabel('Number of iterations'); plt.ylabel('Cross-correlation')
    plt.xlim(0, num_it); plt.ylim(0, 1)
    plt.legend(['Cross-correlation of two T1 slices using rigid intensity-based registration',
                'Cross-correlation of two T1 slices using affine intensity-based registration',
                'Cross-correlation of a T1 and T2 slice using affine intensity-based registration'])
    plt.show()


def plotmi():
    I_path = '../data/image_data/1_1_t1.tif'
    Im_path = '../data/image_data/1_1_t1_d.tif'
    Im_path_t2 = '../data/image_data/1_1_t2.tif'
    ex4 = pro.intensity_based_registration(I_path, Im_path, 1, 1, 0)
    ex5 = pro.intensity_based_registration(I_path, Im_path_t2, 1, 1, 0)
    num_it = 50
    it_vec = np.arange(1, num_it + 1)

    plt.figure(figsize=(10, 10))
    plt.grid()
    plt.plot(it_vec, ex4, 'k')
    plt.plot(it_vec, ex5)
    plt.xlabel('Number of iterations');
    plt.ylabel('Mutual-information')
    plt.xlim(0, num_it);
    plt.ylim(0, 1)
    plt.legend(['Mutual information of two T1 slices using affine intensity-based registration',
                'Mutual information of a T1 and T2 slice using affine intensity-based registration'])
    plt.show()