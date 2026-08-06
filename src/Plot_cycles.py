# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 16:33:10 2026

@author: md1tva
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.close("all")
cycle_total = []
# -----------------------------------------------------------
# Config
# -----------------------------------------------------------

DATES = ["29_12_25", "05_01_26", "12_01_26"]
MOUSE_NAMES = ["none", "1L", "1R"]
JOINTS = ["hip", "knee", "ankle", "abd"]

SPEED_NORM = mpl.colors.Normalize(vmin=0.25, vmax=0.4)
SPEED_CMAP = mpl.cm.get_cmap("plasma")

NORMALIZE = False # Remove the average shift from the mean curve for each cycle if True

# -----------------------------------------------------------
# Plot function 
# -----------------------------------------------------------

def plot_cycles(axs, data, speeds, outliers, X_old, min_consecutive, title_prefix):
    # Determine outliers curves
    bad_curves = []
    outliers_bool = np.isnan(outliers)
    outliers_index = np.where(outliers_bool==True)[0]
    outliers_index_number = np.unique(outliers_index)
    for index in outliers_index_number:
        if len(outliers_index[np.where(outliers_index==index)])>(min_consecutive-1):
            bad_curves.append(index)
    # Plot each good cycle
    cycle_index = 0
    for cycle, speed in zip(data, speeds):
        axs.plot(np.linspace(0, 100, 101),
                 cycle,
                 #color=SPEED_CMAP(SPEED_NORM(speed)))
                 )
    # Plot the bad curves
    for old_cycle in X_old:
        if cycle_index in bad_curves:
            """
            axs.plot(np.linspace(0, 100, 100),
                     old_cycle,
                     color= 'black')
            """
        #cycle_index+=1 
    axs.set_ylabel(title_prefix, fontsize = 20)
    axs.tick_params(axis='both', which='major', labelsize=20)
    if title_prefix == 'Hip abduction (°)':
        axs.set_xlabel("Stance phase (%)", fontsize = 20)
        axs.set_ylim([-40,40])
    else:
        axs.set_ylim([60,140])

# -----------------------------------------------------------
# Cycle check function 
# -----------------------------------------------------------
def reject_cycles_robust_zscore(X, z_thresh, min_consecutive):
    """
    X: (n_cycles, n_frames)
    Reject cycles with >= min_consecutive frames where |z| > z_thresh.

    Returns:
        X_clean
        X_outliers
        keep_mask
        n_removed
        z_scores (for debugging)
    """

    X = np.array(X)
    n_cycles, n_frames = X.shape

    # Robust statistics across cycles
    median = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - median), axis=0)
    mean_mad = np.median(mad)

    # Avoid division by zero
    #mad[mad == 0] = np.nan

    # Robust z-score
    z = 0.6745 * (X - median) / mean_mad
    
    # Save outliers matrix
    outliers = np.abs(z) > z_thresh
    X_outliers = X.copy()
    X_outliers[outliers] = np.nan

    keep_mask = np.ones(n_cycles, dtype=bool)

    for i in range(n_cycles):
        outside_number = len(z[i][np.abs(z[i]) > z_thresh])

        if outside_number >= min_consecutive:
            keep_mask[i] = False

    X_clean = X[keep_mask]
    return X_clean, X_outliers, keep_mask, X

# -----------------------------------------------------------
# Clean cycle function 
# -----------------------------------------------------------
def clean_joint_cycles(joint_cycles, z_thresh, min_consecutive):
    """
    joint_cycles: list of (100,) arrays
    Returns cleaned cycles + outliers + mask
    """
    if len(joint_cycles) == 0:
        return [], np.array([]), 0

    X = np.array(joint_cycles)
    X_clean, outliers, mask, X_old = reject_cycles_robust_zscore(
        X, z_thresh=z_thresh, min_consecutive=min_consecutive
    )
    return list(X_clean), outliers, mask, X_old

# -----------------------------------------------------------
# Loop over dates
# -----------------------------------------------------------

for date in DATES:

    # Load saved data
    with open(f"joint_angles_R_{date}.pkl", "rb") as f:
        DATA_R = pickle.load(f)

    with open(f"joint_angles_L_{date}.pkl", "rb") as f:
        DATA_L = pickle.load(f)

    with open(f"speed_R_{date}.pkl", "rb") as f:
        SPEED_R = pickle.load(f)

    with open(f"speed_L_{date}.pkl", "rb") as f:
        SPEED_L = pickle.load(f)
    
    CLEAN_DATA_R = {}
    CLEAN_DATA_L = {}
    CLEAN_SPEED_R = {}
    CLEAN_SPEED_L = {}

    # -------------------------------------------------------
    # Loop over mice
    # -------------------------------------------------------

    for mouse_idx, mouse in enumerate(MOUSE_NAMES):

        fig, axes = plt.subplots(4, 2, figsize=(18, 10), sharex=True)
        min_consecutive = 1
        z_thresh = 3.5
        
        # Colorbar
        """
        sm = mpl.cm.ScalarMappable(cmap=SPEED_CMAP, norm=SPEED_NORM)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.90, 0.25, 0.03, 0.5])
        cbar = fig.colorbar(sm, cax=cbar_ax, )
        cbar.set_label("Trotting speed (m/s)", fontsize = 20)
        cbar.ax.tick_params(labelsize=20)
        """

        # Extract cycles and speeds
        hip_R   = DATA_R[mouse]["hip"]
        knee_R  = DATA_R[mouse]["knee"]
        ankle_R = DATA_R[mouse]["ankle"]
        abd_R   = DATA_R[mouse]["abd"]

        hip_L   = DATA_L[mouse]["hip"]
        knee_L  = DATA_L[mouse]["knee"]
        ankle_L = DATA_L[mouse]["ankle"]
        abd_L   = DATA_L[mouse]["abd"]

        speed_R = SPEED_R[mouse_idx]
        speed_L = SPEED_L[mouse_idx]
        
        # Clean Right side
        hip_R, hip_R_out, mask_hip_R, hip_R_old = clean_joint_cycles(hip_R, z_thresh, min_consecutive)
        knee_R, knee_R_out, mask_knee_R, knee_R_old = clean_joint_cycles(knee_R, z_thresh, min_consecutive)
        ankle_R, ankle_R_out, mask_ankle_R, ankle_R_old = clean_joint_cycles(ankle_R, z_thresh, min_consecutive)
        abd_R, abd_R_out, mask_abd_R, abd_R_old = clean_joint_cycles(abd_R, z_thresh, min_consecutive)
    
        # Clean Left side
        hip_L, hip_L_out, mask_hip_L, hip_L_old = clean_joint_cycles(hip_L, z_thresh, min_consecutive)
        knee_L, knee_L_out, mask_knee_L, knee_L_old = clean_joint_cycles(knee_L, z_thresh, min_consecutive)
        ankle_L, ankle_L_out, mask_ankle_L, ankle_L_old = clean_joint_cycles(ankle_L, z_thresh, min_consecutive)
        abd_L, abd_L_out, mask_abd_L, abd_L_old = clean_joint_cycles(abd_L, z_thresh, min_consecutive)
        
        # Compute the number of minimum correct cycles
        n_cycles_R = min(len(hip_R), len(knee_R), len(ankle_R), len(abd_R))
        n_cycles_L = min(len(hip_L), len(knee_L), len(ankle_L), len(abd_L))
        
        ax_hip_R, ax_knee_R, ax_ankle_R, ax_abd_R = axes[:, 0]
        ax_hip_L, ax_knee_L, ax_ankle_L, ax_abd_L = axes[:, 1]
        
        # Recompute the speed for each joint (right)
        speed_R_hip = np.array(speed_R)[mask_hip_R]
        speed_R_knee = np.array(speed_R)[mask_knee_R]
        speed_R_ankle = np.array(speed_R)[mask_ankle_R]
        speed_R_abd = np.array(speed_R)[mask_abd_R]
        
        # Recompute the speed for each joint (left)
        speed_L_hip = np.array(speed_L)[mask_hip_L]
        speed_L_knee = np.array(speed_L)[mask_knee_L]
        speed_L_ankle = np.array(speed_L)[mask_ankle_L]
        speed_L_abd = np.array(speed_L)[mask_abd_L]
        
        # Normalize the data if needed
        if NORMALIZE == True:
            # Create list of intial data
            starting_list = 0
            joint_all = hip_R + knee_R + ankle_R + abd_R + hip_L + knee_L + ankle_L + abd_L
            joint_all_number = [len(hip_R), len(knee_R), len(ankle_R), len(abd_R), len(hip_L), len(knee_L), len(ankle_L), len(abd_L)]
            for joint_number in joint_all_number:
                data = joint_all[starting_list: starting_list + joint_number]
                cycle_index = 0
                # Determine the normalization offset foe each cycle
                offset = np.mean(data-np.mean(data, axis = 0), axis = 1)
                # Remove the offset from each cycle
                for cycle_number, cycle in enumerate(data):
                    cycle -= offset[cycle_index]
                    data[cycle_number] = cycle
                    cycle_index += 1
                # Store the normlized data in the list
                joint_all[starting_list: starting_list + joint_number] = data
                starting_list += joint_number
            
            # Redefine the value of each joint angle
            hip_R = joint_all[0: len(hip_R)]
            knee_R = joint_all[len(hip_R): len(hip_R) + len(knee_R)]
            ankle_R = joint_all[len(hip_R) + len(knee_R): len(hip_R) + len(knee_R)+ len(ankle_R)]
            abd_R = joint_all[len(hip_R) + len(knee_R)+ len(ankle_R): len(hip_R) + len(knee_R)+ len(ankle_R) + len(abd_R)]
            hip_L = joint_all[len(hip_R) + len(knee_R)+ len(ankle_R) + len(abd_R): len(hip_R) + len(knee_R)+ len(ankle_R) + len(abd_R) + len(hip_L)]
            knee_L = joint_all[len(hip_R) + len(knee_R)+ len(ankle_R) + len(abd_R) + len(hip_L): len(hip_R) + len(knee_R)+ len(ankle_R) + len(abd_R) + len(hip_L) + len(knee_L)]
            ankle_L = joint_all[len(hip_R) + len(knee_R)+ len(ankle_R) + len(abd_R) + len(hip_L) + len(knee_L): len(hip_R) + len(knee_R)+ len(ankle_R) + len(abd_R) + len(hip_L) + len(knee_L) + len(ankle_L)]
            abd_L = joint_all[len(hip_R) + len(knee_R)+ len(ankle_R) + len(abd_R) + len(hip_L) + len(knee_L) + len(ankle_L): len(hip_R) + len(knee_R)+ len(ankle_R) + len(abd_R) + len(hip_L) + len(knee_L) + len(ankle_L) + len(abd_L)]
        
        # Right
        plot_cycles(ax_hip_R, hip_R, speed_R_hip, hip_R_out, hip_R_old, min_consecutive, "Hip flexion (°)")
        plot_cycles(ax_knee_R, knee_R, speed_R_knee, knee_R_out, knee_R_old, min_consecutive, "Knee flexion (°)")
        plot_cycles(ax_ankle_R, ankle_R, speed_R_ankle, ankle_R_out, ankle_R_old, min_consecutive, "Ankle flexion (°)")
        plot_cycles(ax_abd_R, abd_R, speed_R_abd, abd_R_out, abd_R_old, min_consecutive, "Hip abduction (°)")

        # Left
        plot_cycles(ax_hip_L, hip_L, speed_L_hip, hip_L_out, hip_L_old, min_consecutive, "Hip flexion (°)")
        plot_cycles(ax_knee_L, knee_L, speed_L_knee, knee_L_out, knee_L_old, min_consecutive, "Knee flexion (°)")
        plot_cycles(ax_ankle_L, ankle_L, speed_L_ankle, ankle_L_out, ankle_L_old, min_consecutive, "Ankle flexion (°)")
        plot_cycles(ax_abd_L, abd_L, speed_L_abd, abd_L_out, abd_L_old, min_consecutive, "Hip abduction (°)")

        # Titles
        ax_hip_R.set_title(f"Right side ({n_cycles_R})")
        ax_hip_L.set_title(f"Left side ({n_cycles_L})")
        fig.suptitle(f"Joint angles – Mouse {mouse} – Date {date}")
        
        cycle_size = len(hip_R) + len(knee_R) + len(ankle_R) + len(abd_R) + len(hip_L) + len(knee_L) + len(ankle_L) + len(abd_L)
        cycle_total.append(cycle_size)
        
        plt.show()
        
        # Save cleaned joint data
        CLEAN_DATA_R[mouse] = {
            "hip": hip_R,
            "knee": knee_R,
            "ankle": ankle_R,
            "abd": abd_R,
        }
        
        CLEAN_DATA_L[mouse] = {
            "hip": hip_L,
            "knee": knee_L,
            "ankle": ankle_L,
            "abd": abd_L,
        }
        
        # Save cleaned speeds (per joint!)
        CLEAN_SPEED_R[mouse] = {
            "hip": speed_R_hip.tolist(),
            "knee": speed_R_knee.tolist(),
            "ankle": speed_R_ankle.tolist(),
            "abd": speed_R_abd.tolist(),
        }
        
        CLEAN_SPEED_L[mouse] = {
            "hip": speed_L_hip.tolist(),
            "knee": speed_L_knee.tolist(),
            "ankle": speed_L_ankle.tolist(),
            "abd": speed_L_abd.tolist(),
        }
        
    with open(f"joint_angles_R_clean_{date}.pkl", "wb") as f:
        pickle.dump(CLEAN_DATA_R, f)
    
    with open(f"joint_angles_L_clean_{date}.pkl", "wb") as f:
        pickle.dump(CLEAN_DATA_L, f)
    
    with open(f"speed_R_clean_{date}.pkl", "wb") as f:
        pickle.dump(CLEAN_SPEED_R, f)
    
    with open(f"speed_L_clean_{date}.pkl", "wb") as f:
        pickle.dump(CLEAN_SPEED_L, f)