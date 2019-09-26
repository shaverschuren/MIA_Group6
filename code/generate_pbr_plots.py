"""
Code used for the generation of pbr plots
"""

import sys
sys.path.append("../code")
import registration_util as util
import registration as reg
import matplotlib.pyplot as plt

# Choose images to be registered
I_path = '../data/image_data/3_2_t1.tif'
Im_path = '../data/image_data/3_2_t1_d.tif'

# Read out images
I = plt.imread(I_path)
Im = plt.imread(Im_path)

Im_t, X_t = reg.pbr(I_path,Im_path)

# Display images
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(131)
im11 = ax1.imshow(I)

ax2 = fig.add_subplot(132)
im21 = ax2.imshow(Im)

ax3 = fig.add_subplot(133)
im31 = ax3.imshow(I)
im32 = ax3.imshow(Im_t, alpha=0.7)

ax1.set_title('Original image (I)')
ax2.set_title('Warped image (Im)')
ax3.set_title('Overlay with I and transformed Im')

fig.display()

