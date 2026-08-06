# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 12:41:42 2026

@author: md1tva
"""

"""
Mouse hindlimb motion capture processing pipeline.

This script:
1. Loads C3D files from multiple acquisition dates and mice
2. Extracts and preprocesses marker trajectories
3. Detects stance phases
4. Computes joint kinematics and trotting speed
5. Normalizes gait cycles to 101 points
6. Saves processed joint angles and speed data

Author: Thomas VALERIO
Date: 2026-02-17
"""

# ============================================================================
# Imports
# ============================================================================

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import ezc3d

from customized_functions import (
    interpolate_nans,
    compute_angles_side,
    get_file_numbers,
    detect_stance_phases_foot,
    compute_velocity,
    hip_abduction_angle,
    process_hindlimb_markers,
)

plt.close("all")

# ============================================================================
# Configuration parameters
# ============================================================================

ACQUISITION_DATES = ["29_12_25", "05_01_26", "12_01_26"]
MOUSE_NAMES = ["1L", "1R"]

SAMPLING_RATE = 200  # Hz
FILTER_CUTOFF = 20   # Hz
FILTER_ORDER = 4
MIN_SPEED = 0.25   # m/s threshold for valid cycles

# Marker order in C3D file
MARKER_NAMES = [
    "RASI", "RightHIP", "RightKNEE", "RightANKLE", "RightMET",
    "LASI", "LEFT_HIP", "LeftKNEE", "LeftANKLE", "LeftMET"
]

# C3D files folder
os.chdir('c3d_files')

# ============================================================================
# Helper functions
# ============================================================================

def load_markers(c3d_file, marker_names):
    """Load marker trajectories from a C3D file."""
    c3d = ezc3d.c3d(c3d_file)
    data = c3d["data"]["points"]
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]

    xyz = {}
    for marker in marker_names:
        idx = labels.index(marker)
        xyz[marker] = data[:3, idx, :].T  # (Nframes, 3)
    return xyz


def normalize_cycle(signal, n_points=101):
    """Interpolate a time series to a fixed number of points."""
    old_x = np.linspace(0, 1, len(signal))
    new_x = np.linspace(0, 1, n_points)
    return np.interp(new_x, old_x, signal)


def process_side(xyz, frames, side_markers, contra_marker):
    """
    Compute joint angles, abduction, and speed for one limb side.

    Returns normalized cycles and speed list.
    """
    hip_all, knee_all, ankle_all, hip_abd_all, speeds = [], [], [], [], []

    for idx, (f0, f1) in enumerate(frames):

        hip, knee, ankle = compute_angles_side(
            xyz[side_markers[0]][f0:f1],
            xyz[side_markers[1]][f0:f1],
            xyz[side_markers[2]][f0:f1],
            xyz[side_markers[3]][f0:f1],
            xyz[side_markers[4]][f0:f1],
        )

        hip_abd = hip_abduction_angle(
            xyz[side_markers[0]][f0:f1],
            xyz[side_markers[1]][f0:f1],
            xyz[side_markers[2]][f0:f1],
            xyz[contra_marker][f0:f1],
        )

        # Skip cycles with NaNs
        if np.isnan(hip).any() or np.isnan(knee).any() or np.isnan(ankle).any() or np.isnan(hip_abd).any():
            print(f"Cycle {idx+1} contains NaNs → skipped")
            continue

        # Compute speed
        _, speed = compute_velocity(xyz[side_markers[1]][f0:f1], SAMPLING_RATE)
        speed = np.nanmean(speed) / 1000
        speed = np.round(speed, 2)

        if speed < MIN_SPEED:
            continue

        # Interpolate NaNs (just in case)
        hip = interpolate_nans(hip)
        knee = interpolate_nans(knee)
        ankle = interpolate_nans(ankle)
        hip_abd = interpolate_nans(hip_abd)

        # Convert to flexion convention
        hip = 180 - hip
        knee = 180 - knee
        ankle = 180 - ankle

        # Normalize to 100 points
        hip_i = normalize_cycle(hip)
        knee_i = normalize_cycle(knee)
        ankle_i = normalize_cycle(ankle)
        hip_abd_i = normalize_cycle(hip_abd)

        hip_all.append(hip_i)
        knee_all.append(knee_i)
        ankle_all.append(ankle_i)
        hip_abd_all.append(hip_abd_i)
        speeds.append(speed)

        print(f"Cycle {idx+1} speed = {speed} m/s")

    return hip_all, knee_all, ankle_all, hip_abd_all, speeds

# ============================================================================
# Main processing loop
# ============================================================================

for date in ACQUISITION_DATES:

    print(f"\nProcessing acquisition date: {date}")
    sR_boxplot, sL_boxplot = [], []
    n_cycles_R, n_cycles_L = [], []
    joint_data_R, joint_data_L = {}, {}

    for mouse_idx, mouse_name in enumerate(MOUSE_NAMES):
        
        # Get valid trial numbers
        trials = get_file_numbers(os.getcwd(), f"Cage6_{mouse_name}_{date}_trial_", ".c3d")

        hip_R, knee_R, ankle_R, abd_R = [], [], [], []
        hip_L, knee_L, ankle_L, abd_L = [], [], [], []
        
        speed_R_all, speed_L_all = [], []

        for trial in trials:

            c3d_file = f"Cage6_{mouse_name}_{date}_trial_{trial}.c3d"
            xyz = load_markers(c3d_file, MARKER_NAMES)

            xyz = process_hindlimb_markers(xyz)

            # Detect stance phases
            frames_R = detect_stance_phases_foot(xyz["RightMET"], SAMPLING_RATE)
            frames_L = detect_stance_phases_foot(xyz["LeftMET"], SAMPLING_RATE)

            # Process both sides
            side_R = MARKER_NAMES[0:5]
            side_L = MARKER_NAMES[5:10]

            hR, kR, aR, abdR, sR = process_side(xyz, frames_R, side_R, "LASI")
            hL, kL, aL, abdL, sL = process_side(xyz, frames_L, side_L, "RASI")

            hip_R += hR; knee_R += kR; ankle_R += aR; abd_R += abdR
            hip_L += hL; knee_L += kL; ankle_L += aL; abd_L += abdL

            speed_R_all += sR
            speed_L_all += sL
            sR_boxplot += sR
            sL_boxplot += sL
            
        # Store the data
        joint_data_R[mouse_name] = {"hip": hip_R, "knee": knee_R, "ankle": ankle_R, "abd": abd_R}
        joint_data_L[mouse_name] = {"hip": hip_L, "knee": knee_L, "ankle": ankle_L, "abd": abd_L}

        n_cycles_R.append(len(hip_R))
        n_cycles_L.append(len(hip_L))

    # ============================================================================
    # Save joint angles
    # ============================================================================

    with open(f"../joint_angles_R_{date}.pkl", "wb") as f:
        pickle.dump(joint_data_R, f)

    with open(f"../joint_angles_L_{date}.pkl", "wb") as f:
        pickle.dump(joint_data_L, f)

    # ============================================================================
    # Speed statistics plots
    # ============================================================================
    
    def plot_speed_boxplot(speed_list, n_cycles, side_label):
        speed_dict = {}
        start = 0
        for i, n in enumerate(n_cycles):
            speed_dict[i] = speed_list[start:start+n]
            start += n

        df = pd.DataFrame({
            "Speed (m/s)": sum(speed_dict.values(), []),
            "Mouse": sum([[f"{i+1} (n={n})"]*len(speed_dict[i]) for i, n in enumerate(n_cycles)], [])
        })

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x="Mouse", y="Speed (m/s)")
        plt.title(f"Trotting speed ({side_label}, date={date})")
        plt.show()
        
        plt.figure(figsize=(6, 4))
        plt.scatter(np.linspace(1,len(speed_dict[0]),len(speed_dict[0])), speed_dict[0], marker = '.')
        plt.scatter(np.linspace(1,len(speed_dict[1]),len(speed_dict[1])), speed_dict[1], marker = '*')
        plt.xlabel('Cycles number')
        plt.ylabel('Speed (m/s)')
        plt.title(f"Trotting speed over cycles ({side_label} Limb, Date = {date})")
        #plt.legend(['Mouse_1', 'Mouse_2')
        plt.show()

        return speed_dict

    speed_dict_R = plot_speed_boxplot(sR_boxplot, n_cycles_R, "Right")
    speed_dict_L = plot_speed_boxplot(sL_boxplot, n_cycles_L, "Left")

    with open(f"../speed_R_{date}.pkl", "wb") as f:
        pickle.dump(speed_dict_R, f)

    with open(f"../speed_L_{date}.pkl", "wb") as f:
        pickle.dump(speed_dict_L, f)
    
# ============================================================================
# Global Comparison Plots (Mouse x Date)
# ============================================================================

# 1. Gather all data into a master list for plotting
all_data = []
age = ['20 weeks', '21 weeks', '22 weeks']

for d, date in enumerate(ACQUISITION_DATES):
    for side in ["R", "L"]:
        # Load the saved speed dictionaries for this date
        with open(f"../speed_{side}_{date}.pkl", "rb") as f:
            speed_dict = pickle.load(f)
        
        for mouse_idx, speeds in speed_dict.items():
            for s in speeds:
                all_data.append({
                    "Speed (m/s)": s,
                    "Mouse": f"Mouse {mouse_idx + 1}",
                    "Date": age[d],
                    "Side": "Right" if side == "R" else "Left"
                })

df_total = pd.DataFrame(all_data)

# 2. Create the plots for each side
for side in ["Right", "Left"]:
    plt.figure(figsize=(12, 6))
    
    # Filter data for the specific side
    df_side = df_total[df_total["Side"] == side]
    
    # Create the boxplot
    # x="Mouse" groups the mice together
    # hue="Date" creates the 3 colored boxes per mouse
    sns.boxplot(
        data=df_side, 
        x="Mouse", 
        y="Speed (m/s)", 
        hue="Date", 
        palette="viridis"
    )
    
    plt.title(f"Speed Comparison by Mouse and Date - {side} Side")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Age", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
        
        
        
