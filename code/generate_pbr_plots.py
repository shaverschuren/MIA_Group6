"""
Code used for the generation of pbr plots
"""

import sys
sys.path.append("../code")
import registration_util as util
import registration as reg
import matplotlib.pyplot as plt
import numpy as np

# Choose images to be registered
I_path = '../data/image_data/3_2_t1.tif'
Im_path = '../data/image_data/3_2_t1_d.tif'

# Read out images
I = plt.imread(I_path)
Im = plt.imread(Im_path)

Im_t, X_t, X, Xm, T = reg.pbr(I_path,Im_path)

print(T)
T_aff = T[:2, :2]
print(T_aff)
t = T[:2, 2]
print(t)
# Compute Xm_t:
Xm_t = np.zeros([2, int(np.size(Xm)/2)])
for x in range(0,int(np.size(Xm)/2)):
    Xm_t[:, x] = T_aff.dot(np.transpose(Xm[:, x])) + t

# Display images
fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131)
ax1.axis('off')
im11 = ax1.imshow(I)
c_points1 = ax1.plot(*X, 'r.', "Markersize", 3)
ax1.legend(['Control points'])

ax2 = fig.add_subplot(132)
ax2.axis('off')
im21 = ax2.imshow(Im)
c_points2 = ax2.plot(*Xm, 'w.', "Markersize", 3)
ax2.legend(['Control points'])

ax3 = fig.add_subplot(133)
ax3.axis('off')
im31 = ax3.imshow(I)
im32 = ax3.imshow(Im_t, alpha=0.7)
c_points3_1 = ax3.plot(*X, 'r.',"Markersize",3)
c_points3_2 = ax3.plot(*Xm_t, 'w.',"Markersize",3)
ax3.legend(['Control points (I)', 'Control points (Im_t)'])

ax1.set_title('Fixed image (I) - 3_2_t1.tif')
ax2.set_title('Moving image (Im) - 3_2_t1_d.tif')
ax3.set_title('Overlay with I and transformed Im')
fig.suptitle("Visual representation of point-based registration", fontsize=16)

fig.show()
