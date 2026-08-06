# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:39:41 2026

@author: md1tva
"""

"""
Visualization of gait joint angles and SPM analysis.

This script:
- Loads joint angle and speed data
- Removes outliers using MAD
- Computes mean ± SD
- Plots:
    1) Each date for each mouse (right and left)
    2) Each date combined for the three mice (right and left)
    3) Right vs left for each mice
    4) Right vs left combined for the three mice
- Compute the SPM analysis and plot the significancy

Author: Thomas VALERIO
Date: 2026-02-17
"""

# ============================================================================
# Imports
# ============================================================================

import spm1d
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc

plt.close("all")

# ======================================================================
# Configuration
# ======================================================================

ACQUI_DATES = ["29_12_25", "05_01_26", "12_01_26"]
WEEK_LABELS = {
    "29_12_25": "20 weeks",
    "05_01_26": "21 weeks",
    "12_01_26": "22 weeks",
}

# Base color per week
WEEK_COLORS = {
    "29_12_25": "#2ca02c",  # 20 weeks - green
    "05_01_26": "#ff7f0e",  # 21 weeks - orange
    "12_01_26": "#9467bd",  # 22 weeks - purple
}

JOINTS = ["hip", "abd", "knee", "ankle"]
JOINTS_LABELS = {
    "hip": "Hip_flexion",
    "abd": "Hip_abduction",
    "knee": "Knee_flexion",
    "ankle": "Ankle_flexion",
}


MICE = ["none", "1L", "1R"]
MICE_LABELS = {
    "none": "1",
    "1L": "2",
    "1R": "3",
}

SIDES = ["R", "L"]

N_FRAMES = 101
FRAMES = np.linspace(0, 100, N_FRAMES)

YLIMS = {
    "hip": (-20, 40),
    "knee": (-40, 20),
    "ankle": (-20, 40),
    "abd": (-20, 40),
}

# Base color per mouse
MOUSE_COLORS = {
    "none": "#7f7f7f",
    "1L":   "#2ca02c",
    "1R":   "#1f77b4",
}

# ======================================================================
# Literature data from 2D video (Charles) and fluoroscopy (Bojados)
# ======================================================================
charles_data = {
    "hip": {
        "mean": np.array([100,105,110,115,120,125,127.5,130,127.5,125]),
        "std":  50*np.ones((10))
    },
    "knee": {
        "mean": np.array([100,98,96,94,92,90,89,88,87,85]),
        "std":  20*np.ones((10))
    },
    "ankle": {
        "mean": np.array([75,77.5,80,80,80,90,100,110,110,110]),
        "std":  5*np.ones((10))
    }
}

bojados_data = {
    "hip": {
        "mean": np.array([45,47,49,51,60,70,80,81,82,82]),
        "std":  20*np.ones((10))
    },
    "knee": {
        "mean": np.array([90,85,80,78,76,75,75,78,80,78]),
        "std":  10*np.ones((10))
    },
    "ankle": {
        "mean": np.array([95,92.5,90,92,94,95,105,115,120,120]),
        "std":  20*np.ones((10))
    }
}

# ======================================================================
# Color utilities
# ======================================================================

def adjust_color(color, factor=1.0):
    c = np.array(mc.to_rgb(color))
    if factor > 1:
        return tuple(1 - (1 - c) / factor)
    else:
        return tuple(c * factor)

def get_side_colors(mouse):
    base = MOUSE_COLORS[mouse]
    return {
        "R": adjust_color(base, 0.8),   # darker
        "L": adjust_color(base, 1.6),   # lighter
    }

# ======================================================================
# Data loading
# ======================================================================

def load_all_data():
    data = {side: {} for side in SIDES}
    speed = {side: {} for side in SIDES}

    for date in ACQUI_DATES:
        for side in SIDES:
            with open(f"joint_angles_{side}_clean_{date}.pkl", "rb") as f:
                data[side][date] = pickle.load(f)

            with open(f"speed_{side}_clean_{date}.pkl", "rb") as f:
                speed[side][date] = pickle.load(f)

    return data, speed

# ======================================================================
# Data normalization 
# ======================================================================

def normalize_data(data):
    for side in SIDES:     
        for date in ACQUI_DATES:
            for mouse in MICE:
                for joint in JOINTS:
                    n_cycles = len(data[side][date][mouse][joint])
                    array_temp = np.zeros((n_cycles,N_FRAMES))
                    for i in range(n_cycles):
                        array_temp[i,:] = data[side][date][mouse][joint][i]
                    offset = np.mean(array_temp[:,0])
                    data[side][date][mouse][joint] = data[side][date][mouse][joint] - offset
                    
    return data

# ======================================================================
# Statistics utilities
# ======================================================================

def stack_trials(trials):
    if len(trials) == 0:
        return np.empty((0, N_FRAMES))
    return np.vstack(trials)

def mean_std_without_outliers(X, z_thresh=1e8):
    median = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - median), axis=0)
    mad[mad == 0] = np.nan

    z = 0.6745 * (X - median) / mad
    X_clean = X.copy()
    X_clean[np.abs(z) > z_thresh] = np.nan

    return np.nanmean(X_clean, axis=0), np.nanstd(X_clean, axis=0)

def compute_mean_speed(speed_dict, date, mouse):
    """
    Returns mean speed per cycle averaged across joints.
    Handles unequal number of cycles safely.
    """

    speeds = [
        np.array(speed_dict[date][mouse][j])
        for j in JOINTS
    ]

    # Find minimal number of cycles
    n_min = min(len(s) for s in speeds)

    # Truncate all to same length
    speeds_truncated = [s[:n_min] for s in speeds]

    S = np.vstack(speeds_truncated)

    return np.nanmean(S, axis=0)

def compute_stats(data_dict, speed_dict, side, date, mouse, joint):
    trials = data_dict[side][date][mouse][joint]
    X = stack_trials(trials)

    mean, std = mean_std_without_outliers(X)
    n = len(trials)

    speed_cycles = compute_mean_speed(speed_dict[side], date, mouse)
    speed_cycles = speed_cycles[:n]

    return mean, std, n, np.nanmean(speed_cycles), np.nanstd(speed_cycles)

# ======================================================================
# Plot utilities
# ======================================================================

def plot_curve(ax, mean, std, label, color, linestyle='-'):
    ax.plot(FRAMES, mean, label=label, color=color, linewidth=2, linestyle=linestyle)
    ax.fill_between(FRAMES, mean - std, mean + std,
                    color=color, alpha=0.25)

def format_axes(axes):
    for ax, joint in zip(axes, JOINTS):
        ax.set_ylim(*YLIMS[joint])
        ax.set_ylabel(f"{joint.capitalize()} angle (°)")
        ax.spines[['top', 'right']].set_visible(False)

    axes[-1].set_xlabel("Stance phase (%)")
    axes[-1].legend(frameon=False)

# ======================================================================
# Plotting functions
# ======================================================================

def plot_right_left(mouse, date, DATA, SPEED):
    colors = get_side_colors(mouse)

    fig, axes = plt.subplots(2, 2, figsize=(8, 10), sharex=True)

    for joint, ax in zip(JOINTS, axes):

        for side, linestyle in zip(SIDES, ['-', '--']):

            mean, std, n, s_mean, s_sd = compute_stats(
                DATA, SPEED, side, date, mouse, joint
            )

            label = f"{side} (n={n})" if joint != "abd" else \
                f"{side} (n={n}, {s_mean:.2f}±{s_sd:.2f} m/s)"

            plot_curve(ax, mean, std, label,
                       colors[side], linestyle)

    format_axes(axes)
    fig.suptitle(f"Right vs Left – Mouse {mouse} – {WEEK_LABELS[date]}")
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------

def plot_mouse_across_weeks(mouse, side, DATA, SPEED):

    base_color = MOUSE_COLORS[mouse]

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    for date in ACQUI_DATES:

        week_label = WEEK_LABELS[date]
        color = adjust_color(base_color, 1 + 0.3 * ACQUI_DATES.index(date))

        for joint, ax in zip(JOINTS, axes):

            mean, std, n, s_mean, s_sd = compute_stats(
                DATA, SPEED, side, date, mouse, joint
            )

            label = week_label if joint != "abd" else \
                f"{week_label} (n={n})"

            plot_curve(ax, mean, std, label, color)

    format_axes(axes)
    fig.suptitle(f"Mouse {mouse} – Side {side}")
    plt.tight_layout()
    plt.show()

def plot_mouse_all_weeks_RL(mouse, DATA, SPEED, mouse_label):

    # 2 rows × 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    axes = axes.flatten()   # Convert 2x2 array to flat list

    for date in ACQUI_DATES:

        week_label = WEEK_LABELS[date]
        base_color = WEEK_COLORS[date]

        # Darker = Right, Lighter = Left
        color_R = adjust_color(base_color, 0.8)
        color_L = adjust_color(base_color, 1.6)

        for joint, ax in zip(JOINTS, axes):

            # ---- RIGHT ----
            mean_R, std_R, nR, sR, sdR = compute_stats(
                DATA, SPEED, "R", date, mouse, joint
            )
            
            label_R = f"{week_label} R"

            plot_curve(ax, mean_R, std_R,
                       label_R,
                       color_R,
                       linestyle='-')

            # ---- LEFT ----
            mean_L, std_L, nL, sL, sdL = compute_stats(
                DATA, SPEED, "L", date, mouse, joint
            )

            label_L = f"{week_label} L"

            plot_curve(ax, mean_L, std_L,
                       label_L,
                       color_L,
                       linestyle='--')

    # Formatting
    for ax, joint in zip(axes, JOINTS):
        joint_label = JOINTS_LABELS[joint]
        ax.set_ylim(*YLIMS[joint])
        ax.set_title(f"{joint_label}", fontsize=15)
        ax.set_ylabel("Angle (°)", fontsize=15)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', labelsize=15)

    # Only bottom row gets x-label
    axes[2].set_xlabel("Stance phase (%)", fontsize=15)
    axes[3].set_xlabel("Stance phase (%)", fontsize=15)

    # Single legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='center right',
               frameon=False)

    fig.suptitle(f"Relative joint angles for mouse {mouse_label}",
                 fontsize=20)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()
    
# ======================================================================
# Create dataframe for SPM analysis
# ======================================================================

def create_spm_dataframe(DATA, SPEED, filename="gait_data_spm.csv"):

    rows = []

    for mouse in MICE:
        mouse_label = MICE_LABELS[mouse]

        for date in ACQUI_DATES:
            age = WEEK_LABELS[date].split()[0]   # 20 / 21 / 22

            for side in SIDES:

                for joint in JOINTS:

                    mean, std, n, s_mean, s_sd = compute_stats(
                        DATA, SPEED, side, date, mouse, joint
                    )

                    row = {
                        "Mouse": mouse_label,
                        "Age": age,
                        "Side": side,
                        "Joint": JOINTS_LABELS[joint]
                    }

                    # add frames
                    for i in range(N_FRAMES):
                        row[f"Frame{i}"] = mean[i]

                    rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(filename, index=False)

    print(f"\nSPM dataframe saved to: {filename}")
    print(f"Shape: {df.shape}")

    return df

# ----------------------------------------------------------------------
# Average curves across mice
# ----------------------------------------------------------------------

def plot_average_across_mice(DATA, SPEED):

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    for date in ACQUI_DATES:

        week_label = WEEK_LABELS[date]
        base_color = WEEK_COLORS[date]

        color_R = adjust_color(base_color, 0.8)
        color_L = adjust_color(base_color, 1.6)

        for joint, ax in zip(JOINTS, axes):

            for side, color, linestyle in zip(
                ["R", "L"],
                [color_R, color_L],
                ["-", "--"]
            ):

                mouse_curves = []

                for mouse in MICE:

                    mean, std, n, s_mean, s_sd = compute_stats(
                        DATA, SPEED, side, date, mouse, joint
                    )

                    mouse_curves.append(mean)

                mouse_curves = np.vstack(mouse_curves)

                mean_all = np.mean(mouse_curves, axis=0)
                std_all = np.std(mouse_curves, axis=0)

                label = f"{week_label} {side} (n={len(MICE)} mice)"

                plot_curve(ax, mean_all, std_all,
                           label,
                           color,
                           linestyle)

    # Formatting
    for ax, joint in zip(axes, JOINTS):
        joint_label = JOINTS_LABELS[joint]
        ax.set_ylim(*YLIMS[joint])
        ax.set_title(joint_label, fontsize=20)
        ax.set_ylabel("Angle (°)", fontsize=20)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', labelsize=20)

    axes[2].set_xlabel("Stance phase (%)", fontsize=20)
    axes[3].set_xlabel("Stance phase (%)", fontsize=20)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=False)

    fig.suptitle("Mean joint kinematics across mice", fontsize=20)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()

# ----------------------------------------------------------------------
# Plot Right vs Left for each mouse independently (averaging across dates)
# ----------------------------------------------------------------------
def plot_right_left_per_mouse_all_dates(DATA, SPEED):
    for mouse in MICE:
        mouse_label = MICE_LABELS[mouse]
        colors = get_side_colors(mouse)

        fig, axes = plt.subplots(2, 2, figsize=(8, 10), sharex=True)
        axes = axes.flatten()

        for joint, ax in zip(JOINTS, axes):
            for side, linestyle in zip(SIDES, ['-', '--']):
                # collect all cycles across dates
                curves = []
                for date in ACQUI_DATES:
                    mean, _, _, _, _ = compute_stats(DATA, SPEED, side, date, mouse, joint)
                    curves.append(mean)
                curves = np.vstack(curves)
                mean_all = np.mean(curves, axis=0)
                std_all = np.std(curves, axis=0)

                label = f"{side} (n={len(ACQUI_DATES)} dates)"
                plot_curve(ax, mean_all, std_all, label, colors[side], linestyle)

        # Formatting
        for ax, joint in zip(axes, JOINTS):
            joint_label = JOINTS_LABELS[joint]
            ax.set_ylim(*YLIMS[joint])
            ax.set_title(f"{joint_label}", fontsize=16)
            ax.set_ylabel("Angle (°)")
            ax.spines[['top', 'right']].set_visible(False)

        axes[-2].set_xlabel("Stance phase (%)")
        axes[-1].set_xlabel("Stance phase (%)")
        axes[0].legend(frameon=False)
        fig.suptitle(f"Right vs Left – Mouse {mouse_label} (mean across dates)", fontsize=18)
        plt.tight_layout()
        plt.show()

# ----------------------------------------------------------------------
# Plot Right vs Left averaged across all mice
# ----------------------------------------------------------------------
def plot_right_left_average_all_mice(DATA, SPEED):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    for joint, ax in zip(JOINTS, axes):
        for side, linestyle in zip(SIDES, ['-', '--']):
            all_mouse_curves = []
            for mouse in MICE:
                curves = []
                for date in ACQUI_DATES:
                    mean, _, _, _, _ = compute_stats(DATA, SPEED, side, date, mouse, joint)
                    curves.append(mean)
                curves = np.vstack(curves)
                mean_all_dates = np.mean(curves, axis=0)
                all_mouse_curves.append(mean_all_dates)
            all_mouse_curves = np.vstack(all_mouse_curves)
            mean_all = np.mean(all_mouse_curves, axis=0)
            std_all = np.std(all_mouse_curves, axis=0)

            color = 'tab:blue' if side=='R' else 'tab:orange'
            label = f"{side}"
            plot_curve(ax, mean_all, std_all, label, color, linestyle)

    # Formatting
    for ax, joint in zip(axes, JOINTS):
        joint_label = JOINTS_LABELS[joint]
        ax.set_ylim(*YLIMS[joint])
        ax.set_title(joint_label, fontsize=15)
        ax.set_ylabel("Angle (°)", fontsize=15)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', labelsize=15)

    axes[2].set_xlabel("Stance phase (%)", fontsize=15)
    axes[3].set_xlabel("Stance phase (%)", fontsize=15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=False, fontsize=15)
    fig.suptitle("Right vs Left – Mean across all mice", fontsize=15)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()

# ----------------------------------------------------------------------
# Plot all ages for each mouse independently (averaging across sides)
# ----------------------------------------------------------------------
def plot_age_per_mouse_all_sides(DATA, SPEED):
    for mouse in MICE:
        mouse_label = MICE_LABELS[mouse]

        fig, axes = plt.subplots(2, 2, figsize=(8, 10), sharex=True)
        axes = axes.flatten()

        for joint, ax in zip(JOINTS, axes):
            for date in ACQUI_DATES:
                week_label = WEEK_LABELS[date]
                color = adjust_color(WEEK_COLORS[date], 1.0)

                curves = []
                for side in SIDES:
                    mean, _, _, _, _ = compute_stats(DATA, SPEED, side, date, mouse, joint)
                    curves.append(mean)
                curves = np.vstack(curves)
                mean_all_sides = np.mean(curves, axis=0)
                std_all_sides = np.std(curves, axis=0)

                label = f"{week_label} (n={len(SIDES)} sides)"
                plot_curve(ax, mean_all_sides, std_all_sides, label, color, linestyle='-')

        # Formatting
        for ax, joint in zip(axes, JOINTS):
            joint_label = JOINTS_LABELS[joint]
            ax.set_ylim(*YLIMS[joint])
            ax.set_title(f"{joint_label}", fontsize=16)
            ax.set_ylabel("Angle (°)")
            ax.spines[['top', 'right']].set_visible(False)

        axes[-2].set_xlabel("Stance phase (%)")
        axes[-1].set_xlabel("Stance phase (%)")
        axes[0].legend(frameon=False)
        fig.suptitle(f"Joint angles across ages – Mouse {mouse_label} (mean across sides)", fontsize=18)
        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# Plot all ages averaged across all mice
# ----------------------------------------------------------------------
def plot_age_average_all_mice(DATA, SPEED):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    for joint, ax in zip(JOINTS, axes):
        for date in ACQUI_DATES:
            week_label = WEEK_LABELS[date]
            color = adjust_color(WEEK_COLORS[date], 1.0)

            all_mouse_curves = []
            for mouse in MICE:
                curves = []
                for side in SIDES:
                    mean, _, _, _, _ = compute_stats(DATA, SPEED, side, date, mouse, joint)
                    curves.append(mean)
                curves = np.vstack(curves)
                mean_all_sides = np.mean(curves, axis=0)
                all_mouse_curves.append(mean_all_sides)

            all_mouse_curves = np.vstack(all_mouse_curves)
            mean_all = np.mean(all_mouse_curves, axis=0)
            std_all = np.std(all_mouse_curves, axis=0)

            label = f"{week_label} (n={len(MICE)} mice)"
            plot_curve(ax, mean_all, std_all, label, color, linestyle='-')

    # Formatting
    for ax, joint in zip(axes, JOINTS):
        joint_label = JOINTS_LABELS[joint]
        ax.set_ylim(*YLIMS[joint])
        ax.set_title(joint_label, fontsize=20)
        ax.set_ylabel("Angle (°)")
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', labelsize=14)

    axes[2].set_xlabel("Stance phase (%)")
    axes[3].set_xlabel("Stance phase (%)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=False)
    fig.suptitle("Joint angles across ages – Mean across all mice", fontsize=20)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()
    
# ----------------------------------------------------------------------
# Plot effect of mouse (averaged across dates and sides)
# ----------------------------------------------------------------------
def plot_mouse_effect(DATA, SPEED):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    for joint, ax in zip(JOINTS, axes):
        for mouse in MICE:
            curves = []

            for date in ACQUI_DATES:
                for side in SIDES:
                    mean, _, _, _, _ = compute_stats(DATA, SPEED, side, date, mouse, joint)
                    curves.append(mean)

            curves = np.vstack(curves)
            mean_all = np.mean(curves, axis=0)
            std_all = np.std(curves, axis=0)

            label = f"Mouse {MICE_LABELS[mouse]}"
            color = MOUSE_COLORS[mouse]

            plot_curve(ax, mean_all, std_all, label, color, linestyle='-')

    # Formatting
    for ax, joint in zip(axes, JOINTS):
        ax.set_ylim(*YLIMS[joint])
        ax.set_title(JOINTS_LABELS[joint], fontsize=20)
        ax.set_ylabel("Relative Angle (°)", fontsize=16)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', labelsize=14)

    axes[2].set_xlabel("Stance phase (%)", fontsize=16)
    axes[3].set_xlabel("Stance phase (%)", fontsize=16)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=False, fontsize = 13)

    fig.suptitle("Relative joint angles for each mouse", fontsize=20)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()

# ----------------------------------------------------------------------
# SPM analysis
# ----------------------------------------------------------------------
def spm_analysis(DATAFRAME, JOINT):
    
    # Load dataframe
    df = pd.read_csv(DATAFRAME)
    
    # Specify the joint
    joint = JOINT
    
    df_joint = df[df["Joint"] == joint]
    
    # waveform matrix
    Y = df_joint.iloc[:, 4:].values
    
    # Factorize for ANOVA calculations
    # Note: Using pd.factorize returns numerical codes, 
    # but we will extract original text for clean plot legends.
    mouse = pd.factorize(df_joint["Mouse"])[0]
    age_codes, age_labels = pd.factorize(df_joint["Age"])
    side_codes, side_labels = pd.factorize(df_joint["Side"])
    

    # 1️⃣ SIDE EFFECT (paired t-test)
    df_R = df_joint[df_joint["Side"] == "R"]
    df_L = df_joint[df_joint["Side"] == "L"]
    
    Y_R = df_R.iloc[:, 4:].values
    Y_L = df_L.iloc[:, 4:].values
    
    # --- Plot Time Series for Side ---
    plt.figure(figsize=(6, 6))
    
    # Right Side (Blue line, light blue shadow)
    spm1d.plot.plot_mean_sd(Y_R, 
                            linecolor="blue", 
                            facecolor="blue", 
                            edgecolor="none", 
                            alpha=0.15, 
                            label="Right Side")
    
    # Left Side (Red line, light red shadow)
    spm1d.plot.plot_mean_sd(Y_L, 
                            linecolor="red", 
                            facecolor="red", 
                            edgecolor="none", 
                            alpha=0.15, 
                            label="Left Side")
    
    plt.title(f"{joint} – Time Series (R vs L)", fontsize=15)
    plt.xlabel("Stance phase (%)", fontsize=15)
    plt.ylabel("Angle (°)", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
    
    # --- SPM Inference ---
    t = spm1d.stats.ttest_paired(Y_R, Y_L)
    ti = t.inference(alpha=0.05)
    
    plt.figure(figsize=(6, 6))
    ti.plot()
    plt.title(f"{joint} – Side effect (R vs L)", fontsize=15)
    plt.xlabel("Stance phase (%)", fontsize=15)
    plt.ylabel("SPM{t}", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
    
    # 2️⃣ AGE EFFECT (repeated measures ANOVA) 
    # --- Plot Time Series for Age ---
    plt.figure(figsize=(6, 6))
    colors = ["teal", "orange", "purple", "green"] 
    
    for i, label in enumerate(age_labels):
        Y_age = Y[age_codes == i]
        current_color = colors[i % len(colors)]
        
        spm1d.plot.plot_mean_sd(Y_age, 
                                linecolor=current_color, 
                                facecolor=current_color, 
                                edgecolor="none", 
                                alpha=0.15, 
                                label=str(label))
        
    plt.title(f"{joint} – Time Series across Age", fontsize=15)
    plt.xlabel("Stance phase (%)", fontsize=15)
    plt.ylabel("Angle (°)", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.show()
    
    # --- SPM Inference ---
    F_age = spm1d.stats.anova1rm(Y, age_codes, mouse)
    Fi_age = F_age.inference(alpha=0.05)
    
    plt.figure(figsize=(6, 6))
    Fi_age.plot()
    plt.title(f"{joint} – Age effect", fontsize=15)
    plt.xlabel("Stance phase (%)", fontsize=15)
    plt.ylabel("SPM{F}", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()


    # 3️⃣ AGE × SIDE INTERACTION
    # --- Plot Time Series for Age × Side Combinations ---
    plt.figure(figsize=(6, 6))
    # Combining styles to differentiate groups (e.g., solid line for R, dashed for L)
    line_styles = {0: "-", 1: "--"} # Assuming 2 sides 
    
    for a_idx, a_label in enumerate(age_labels):
        for s_idx, s_label in enumerate(side_labels):
            # Mask to find specific Age and Side combo
            mask = (age_codes == a_idx) & (side_codes == s_idx)
            Y_combo = Y[mask]
            
            # Skip if subset is empty
            if len(Y_combo) == 0:
                continue
                
            # Plotting mean line with unique style/color combos
            mean = Y_combo.mean(axis=0)
            plt.plot(mean, label=f"{a_label} ({s_label})", 
                     color=colors[a_idx % len(colors)], 
                     linestyle=line_styles.get(s_idx, "-"))
            
    plt.title(f"{joint} – Time Series Interaction (Age × Side)", fontsize=15)
    plt.xlabel("Stance phase (%)", fontsize=15)
    plt.ylabel("Angle (°)", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.show()

    # --- SPM Inference ---
    F_inter = spm1d.stats.anova2rm(Y, age_codes, side_codes, mouse)
    Fi_inter = F_inter.inference(alpha=0.05)
    
    plt.figure(figsize=(6, 6))
    Fi_inter.plot()
    plt.title("Interaction")
    plt.xlabel("Stance phase (%)")
    plt.ylabel("SPM{F}")
    plt.show()
    
    return ti
# ----------------------------------------------------------------------
# Calculate global mean for literature comparison
# ----------------------------------------------------------------------
def collect_all_trials(DATA, joint):

    trials = []

    for side in SIDES:
        for date in ACQUI_DATES:
            for mouse in MICE:

                joint_trials = DATA[side][date][mouse][joint]

                for trial in joint_trials:
                    trials.append(trial)

    return np.vstack(trials)

def compute_global_stats(DATA):

    results = {}

    for joint in ["hip", "knee", "ankle", "abd"]:

        X = collect_all_trials(DATA, joint)

        mean = np.mean(X, axis=0)
        std  = np.std(X, axis=0)

        results[joint] = {
            "mean": mean,
            "std": std
        }

    return results

def interpolate_literature(lit_data):

    lit_interp = {}

    x_lit = np.linspace(0,100,10)
    x_full = np.linspace(0,100,101)

    for joint in lit_data:

        mean_interp = np.interp(x_full, x_lit, lit_data[joint]["mean"])
        std_interp  = np.interp(x_full, x_lit, lit_data[joint]["std"])

        lit_interp[joint] = {
            "mean": mean_interp,
            "std": std_interp
        }

    return lit_interp

def plot_vs_literature(results, lit_interp_1, lit_interp_2):

    fig, axes = plt.subplots(4,1, figsize=(7,9), sharex=True)

    joints = ["hip","knee","ankle", "abd"]

    for joint, ax in zip(joints, axes):

        mean = results[joint]["mean"]
        std  = results[joint]["std"]

        # Your data
        ax.plot(FRAMES, mean, label="This study (3D motion capture)", color="black", linewidth=2)
        ax.fill_between(FRAMES, mean-std, mean+std,
                        color="black", alpha=0.25)
        
        if joint != "abd": # No abduction values in literature on mice
            # Literature 1
            lit_mean_1 = lit_interp_1[joint]["mean"]
            lit_std_1  = lit_interp_1[joint]["std"]
            ax.plot(FRAMES, lit_mean_1, label="Charles et al., 2018 (2D motion capture)", color="red", linewidth=2)
            ax.fill_between(FRAMES, lit_mean_1-lit_std_1, lit_mean_1+lit_std_1,
                            color="tomato", alpha=0.25)
            
            
            # Literature 2
            lit_mean_2 = lit_interp_2[joint]["mean"]
            lit_std_2  = lit_interp_2[joint]["std"]
            ax.plot(FRAMES, lit_mean_2, label="Bojados et al., 2013 (2D fluoroscopy)", color="blue", linewidth=2)
            ax.fill_between(FRAMES, lit_mean_2-lit_std_2, lit_mean_2+lit_std_2,
                            color="cornflowerblue", alpha=0.25)
    
        ax.set_ylabel(f"{JOINTS_LABELS[joint].capitalize()}(°)", fontsize = 15)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='both', labelsize=15)

    axes[-1].set_xlabel("Stance phase (%)", fontsize = 15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.38, 0.25), frameon=False)
    axes[0].set_title('Population level curves', fontsize = 15)
    
    plt.tight_layout()
    plt.show()

# ======================================================================
# Main execution
# ======================================================================

if __name__ == "__main__":

    DATA, SPEED = load_all_data()
    DATA = normalize_data(DATA)

    # ---------------------------------------------------
    # Create dataframe for SPM statistics
    # ---------------------------------------------------
    df_spm = create_spm_dataframe(DATA, SPEED)

    for mouse in MICE:
        mouse_label = MICE_LABELS[mouse]
        plot_mouse_all_weeks_RL(mouse, DATA, SPEED, mouse_label)
        
    plot_average_across_mice(DATA, SPEED)
    plot_right_left_per_mouse_all_dates(DATA, SPEED)
    plot_right_left_average_all_mice(DATA, SPEED)
    plot_age_per_mouse_all_sides(DATA, SPEED)
    plot_age_average_all_mice(DATA, SPEED)
    plot_mouse_effect(DATA, SPEED)
    t = spm_analysis("gait_data_spm.csv", "Ankle_flexion")
    
    results = compute_global_stats(DATA)
    lit_interp_1 = interpolate_literature(charles_data)
    lit_interp_2 = interpolate_literature(bojados_data)
    plot_vs_literature(results, lit_interp_1, lit_interp_2)