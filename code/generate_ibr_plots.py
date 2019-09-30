"""
Code used for the generation of ibr plots

Author: S.H.A. Verschuren
        30/09/2019
"""

import sys
sys.path.append("../code")
import registration as reg
import matplotlib.pyplot as plt
import numpy as np
import timeit


def intensity_based_registration_no_vis(I_path, Im_path, r_a_switch=0, corr_mi_switch=0):
    # This is an edited ibr function that doesn't display any output.
    # It is also slightly optimised to be able to run a bit faster.
    #
    # Output: Similarity plot, I, Im_t

    # r_a_switch: 0 --> rigid (default)
    #             1 --> affine
    assert r_a_switch == 0 or r_a_switch == 1, "Error: input parameter r_a_switch must be either 0 or 1.. "

    # corr_mi_switch: 0 --> correlation (default)
    #                 1 --> mutual information
    assert corr_mi_switch == 0 or corr_mi_switch == 1, "Error: input parameter corr_mi_switch must be either 0 or 1.. "

    # read the fixed and moving images
    # change these in order to read different images
    I = plt.imread(I_path)
    Im = plt.imread(Im_path)

    # initial values for the parameters
    if r_a_switch == 0:
        x = np.array([0., 0., 0.]) # rotation,transx,transy
    elif r_a_switch == 1:
        x = np.array([0., 1., 1.,0.,0.,0.,0.]) # rotation,scalex,scaley,shearx,sheary,transx,transy
    else:
        print("ERROR.. r_a_switch must be either 0 or 1")

    # the similarity function
    # this line of code in essence creates a version of rigid_corr()
    # in which the first two input parameters (fixed and moving image)
    # are fixed and the only remaining parameter is the vector x with the
    # parameters of the transformation

    if r_a_switch == 0:
        assert corr_mi_switch == 0, "Error, only correlation similarity possible with rigid registration"
        sim_fun = reg.rigid_corr
    elif r_a_switch == 1:
        if corr_mi_switch == 0:
            sim_fun = reg.affine_corr
        else:
            sim_fun = reg.affine_mi
    else:
        print("ERROR.. r_a_switch must be either 0 or 1")

    fun = lambda x: (sim_fun(I, Im, x))[0]
    fun_full = lambda x: sim_fun(I, Im, x)

    if corr_mi_switch == 0:
        # the initial learning rate
        mu = 0.005
        # number of iterations
        num_iter = 50
    else:
        # the initial learning rate
        mu = 0.003
        # number of iterations
        num_iter = 30

    # Which results in the following formula for mu:
    fun_mu = lambda i: mu*np.exp(-5*i/num_iter)         # Which results in an initial mu at iteration 1 and a mu/200 at final iteration

    iterations = np.arange(1, num_iter+1)
    similarity = np.full((num_iter, 1), np.nan)

    # perform 'num_iter' gradient ascent updates
    i = 0
    for k in np.arange(num_iter):

        # gradient ascent
        g = reg.ngradient(fun, x)
        x += g*fun_mu(i)

        # for visualization of the result
        S, Im_t, _ = fun_full(x)

        # update 'learning' curve
        similarity[k] = S

        i = i+1

    return similarity, iterations, I, Im_t, x


# ###################### Main run ########################

# For the ibr exercises, we will have to register 5 pairs of pictures.
# These pairs are given by:
# 1: 1_1_t1.tif and 1_1_t1_d.tif (CC)
# 2: 1_1_t1.tif and 1_1_t1_d.tif (CC)
# 3: 1_1_t1.tif and 1_1_t2.tif (CC)
# 4: 1_1_t1.tif and 1_1_t1_d.tif (MI)
# 5: 1_1_t1.tif and 1_1_t2.tif (MI)
#
# This code computes the registration curves and plots all five of them in a single figure.

print("************************ INITIALISE PROGRAM ***********************\n")
start_tot = timeit.default_timer()

# Firstly, create a list of exercises including switches.
exercise_list = [['1_1_t1.tif', '1_1_t1_d.tif', 0, 0], ['1_1_t1.tif', '1_1_t1_d.tif', 1, 0], ['1_1_t1.tif', '1_1_t2.tif', 1, 0], ['1_1_t1.tif', '1_1_t1_d.tif', 1, 1], ['1_1_t1.tif', '1_1_t2.tif', 1, 1]]
image_path = '../data/image_data/'

# Loop over image names because typing it manually would give me hand cramps..
for i in range(len(exercise_list)):
    for j in range(2):
        exercise_list[i][j] = image_path + exercise_list[i][j]

print(len(exercise_list)," registrations prescribed..\n\n")

# Define some lists to store output vars
similarity_list = []
iterations_list = []
I_list = []
Im_t_list = []
x_list = []

# Now, actually run the program and loop over all exercises
i = 0
for exercise in exercise_list:
    print("*************** START REGISTRATION ",i+1," ***************\n")
    start = timeit.default_timer()
    print("Selected images: ",exercise[0]," and ",exercise[1],"\n")
    print("Registration type: ")
    if exercise[2] == 0:
        print("Rigid")
    else:
        print("Affine")
    if exercise[3] == 0:
        print("CC-based\n")
    else:
        print("MI-based\n")

    # Run program once
    OUTPUT = intensity_based_registration_no_vis(*exercise)
    # Store output in pre-defined lists
    similarity_list.append(OUTPUT[0])
    iterations_list.append(OUTPUT[1])
    I_list.append(OUTPUT[2])
    Im_t_list.append(OUTPUT[3])
    x_list.append(OUTPUT[4])

    print("Final similarity: ", OUTPUT[0][29],"\n")
    stop = timeit.default_timer()
    comp_time = stop - start
    print("Computing time: ",round(comp_time,3)," seconds\n")

    # Plot final registered picture:
    fig = plt.figure(i+1)
    ax = fig.add_subplot(111)

    im1 = ax.imshow(OUTPUT[2])
    im2 = ax.imshow(OUTPUT[3], alpha=0.7)
    ax.set_title(("Final registered image (exercise "+str(i+1)+")"))
    ax.text(0.1, 0.9,("Similarity: "+str(OUTPUT[0][29])),bbox={'facecolor': 'white', 'alpha': 1,'pad': 5},transform=ax.transAxes)
    ax.axis('off')
    fig.show()
    print("Registration complete... \n\n")

    i += 1
# And now for the actual plot:
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, xlim=(0, 30), ylim=(0, 1.1))
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Similarity [See legend]')
ax1.set_title("Similarity curves for intensity-based registrations")
ax1.grid()

for i in range(len(similarity_list)):
    ax1.plot(iterations_list[i], similarity_list[i])

ax1.legend(['Exercise 1 [CC]', 'Exercise 2 [CC]', 'Exercise 3 [CC]', 'Exercise 4 [MI]', 'Exercise 5 [MI]'])

stop_tot = timeit.default_timer()
comp_time_tot = stop_tot - start_tot
print("\nTotal computing time: ", round(comp_time_tot,3), " seconds\n")
print("************************** END PROGRAM *************************")

fig1.show()
