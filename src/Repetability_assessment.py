# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 13:58:45 2026

@author: md1tva
"""

import pickle
import numpy as np
import pandas as pd
from scipy.stats import linregress

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

DATES = ["29_12_25", "05_01_26", "12_01_26"]
MOUSE_NAMES = ["1L", "1R"]
JOINTS = ["hip", "knee", "ankle", "abd"]

# -----------------------------------------------------------
# MEAN WITHOUT OUTLIERS FUNCTION
# -----------------------------------------------------------

def detect_outliers(X, z_thresh=1000000000000):
    """MAD-based outlier rejection."""
    median = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - median), axis=0)
    mad[mad == 0] = np.nan

    z = 0.6745 * (X - median) / mad
    outliers = np.abs(z) > z_thresh

    X_clean = X.copy()
    X_clean[outliers] = np.nan
    
    return X_clean

# -----------------------------------------------------------
# LINEAR FIT FUNCTION
# -----------------------------------------------------------

def compute_linear_fit(cycles):
    """
    Compute the Linear fitting (Lfit) across cycles.
    
    cycles: np.array of shape (n_cycles, n_timepoints)
    
    Returns:
        a0: scalar, offset
        a1: amplitude scaling
        R2: r-square of the linear regression 
    """
    X = np.array(cycles)
    n_cycles, n_points = X.shape

    if n_cycles < 2:
        return np.nan

    # Remove the outliers
    X_clean = detect_outliers(X)

    # Linear regression
    a0_list, a1_list, R2_list = [],[],[]
    for cycle in range(n_cycles):
        # Compute the leave-one-out-mean (loo) (mean by excluding the current cycle)
        loo_matrix = X_clean.copy()
        np.delete(loo_matrix, cycle, axis = 0)
        loo_mean = np.nanmean(loo_matrix, axis=0)
        # Linear fitting
        reg = linregress(loo_mean, X_clean[cycle,:])
        a0_list.append(reg.intercept)
        a1_list.append(reg.slope)
        R2_list.append(reg.rvalue**2)
    a0_mean, a0_std = np.mean(a0_list), np.std(a0_list)
    a1_mean, a1_std = np.mean(a1_list), np.std(a1_list)
    R2_mean, R2_std = np.mean(R2_list), np.std(R2_list)
    
    return a0_mean, a1_mean, R2_mean, a0_std, a1_std, R2_std

# -----------------------------------------------------------
# CV FUNCTION
# -----------------------------------------------------------

def compute_cv(cycles):
    """
    Compute the Coefficient of Variation (Cv) across cycles.
    
    cycles: np.array of shape (n_cycles, n_timepoints)
    
    Returns:
        cv: scalar, coefficient of variation (%)
    """
    X = np.array(cycles)
    n_cycles, n_points = X.shape

    if n_cycles < 2:
        return np.nan

    # Remove the outliers
    X_clean = detect_outliers(X)

    # Mean waveform across cycles
    mean_wave = np.nanmean(X_clean, axis=0)

    # Standard deviation across cycles
    sd_wave = np.nanstd(X_clean, axis=0)
    
    # Coefficient of variation
    cv = np.mean(sd_wave/mean_wave)*100

    return cv

# -----------------------------------------------------------
# CMC FUNCTION
# -----------------------------------------------------------

def compute_cmc(cycles):
    """
    Compute the Coefficient of Multiple Correlation (CMC) across cycles.
    
    cycles: np.array of shape (n_cycles, n_timepoints)
    
    Returns:
        cmc: scalar, waveform reproducibility (0-1)
    """
    X = np.array(cycles)
    n_cycles, n_points = X.shape

    if n_cycles < 2:
        return np.nan

    # Remove the outliers
    X_clean = detect_outliers(X)

    # Step 1: Mean waveform across cycles
    mean_wave = np.nanmean(X_clean, axis=0)

    # Step 2: Sum of squares of residuals (deviation of each cycle from mean)
    ss_error = np.nansum((X_clean - mean_wave) **2)

    # Step 3: Total sum of squares (deviation of each cycle from grand mean)
    grand_mean = np.nanmean(X_clean)
    ss_total = np.nansum((X_clean - grand_mean) **2)

    # Step 4: Compute CMC
    cmc = np.sqrt(1 - ss_error / ss_total)

    return cmc

# -----------------------------------------------------------
# MAV_RMSE FUNCTION
# -----------------------------------------------------------
def compute_mav_rmse(cycles):
    """
    Compute the Mean absolute value (MAV) and root mean square error (RMSE) across cycles.
    
    cycles: np.array of shape (n_cycles, n_timepoints)
    
    Returns:
        mav, rmse: scalars, (% of range)
    """
    X = np.array(cycles)
    n_cycles, n_points = X.shape
    
    # Remove the outliers
    X_clean = detect_outliers(X)
    
    # Compute the range of motion for normalization (RoM = angleMax-angleMin)
    RoM = abs(np.nanmax(np.nanmean(X_clean, axis=0))-np.nanmin(np.nanmean(X_clean, axis=0)))
    
    # Compute MAV and RMSE
    mav_absolute_mean = np.mean(abs(np.nanmax(X_clean, axis=0)-np.nanmin(X_clean, axis=0)))
    mav_absolute_std = np.std(abs(np.nanmax(X_clean, axis=0)-np.nanmin(X_clean, axis=0)))
    rmse_absolute_mean = np.sqrt(np.nanmean((X_clean-np.nanmean(X_clean, axis=0))**2))
    rmse_absolute_std = np.sqrt(np.nanstd((X_clean-np.nanmean(X_clean, axis=0))**2))
    
    # Normalize MAV and RMSE
    mav_mean = (mav_absolute_mean/RoM)*100
    mav_std = (mav_absolute_std/RoM)*100
    rmse_mean = (rmse_absolute_mean/RoM)*100
    rmse_std = (rmse_absolute_std/RoM)*100
    
    return mav_absolute_mean, mav_absolute_std, rmse_absolute_mean, rmse_absolute_std
    

# -----------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------

summary_results = []

for date in DATES:

    print(f"\nProcessing date: {date}")

    with open(f"joint_angles_R_clean_{date}.pkl", "rb") as f:
        DATA_R = pickle.load(f)

    with open(f"joint_angles_L_clean_{date}.pkl", "rb") as f:
        DATA_L = pickle.load(f)

    for mouse in MOUSE_NAMES:

        for side_label, DATA in zip(["R", "L"], [DATA_R, DATA_L]):

            for joint in JOINTS:

                cycles = DATA[mouse][joint]

                if len(cycles) < 2:
                    print(f"Skipping {mouse} | {side_label} | {joint} (not enough cycles)")
                    continue

                X = np.array(cycles)  # shape: (n_cycles, n_timepoints)

                # Compute CMC, MAV, RMSE, CV, and linear fitting across full waveform
                cmc_value = compute_cmc(X)
                mav_mean, mav_std, rmse_mean, rmse_std = compute_mav_rmse(X)
                cv = compute_cv(X)
                a0_mean, a1_mean, R2_mean, a0_std, a1_std, R2_std = compute_linear_fit(X)
                
                print(f"{date} | Mouse {mouse} | {side_label} | {joint} "
                      f"| CMC={cmc_value:.2f} | MAV={mav_mean:.2f}±{mav_std:.2f}° | RMSE={rmse_mean:.2f}±{rmse_std:.2f}°"
                      f"| a0={a0_mean:.2f}±{a0_std:.2f} | a1={a1_mean:.2f}±{a1_std:.2f} | R2={R2_mean:.2f}±{R2_std:.2f}")
                
                summary_results.append({
                    "date": date,
                    "mouse": mouse,
                    "side": side_label,
                    "joint": joint,
                    "n_cycles": len(cycles),
                    "cmc": cmc_value,
                    "mav_mean": mav_mean,
                    "mav_std": mav_std,
                    "rmse_mean": rmse_mean,
                    "rmse_std": rmse_std,
                    "cv": cv,
                    "a0_mean": a0_mean,
                    "a0_std": a0_std,
                    "a1_mean": a1_mean,
                    "a1_std": a1_std,
                    "R2_mean": R2_mean,
                    "R2_std": R2_std,
                })
                
# -----------------------------------------------------------
# SAVE SUMMARY
# -----------------------------------------------------------

df_summary = pd.DataFrame(summary_results)
df_summary.to_csv("reproductibility_summary.csv", index=False)

print("\nSaved Reproductibility summary to Reproductibility_summary.csv")



import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------
# Compute mean ± SD for each joint
# -----------------------------------------------------------

joint_order = ["hip", "knee", "ankle", "abd"]

stats = (
    df_summary.groupby("joint")[["cmc", "mav_mean"]]
    .agg(["mean", "std"])
    .reindex(joint_order)
)

cmc_mean = stats[("cmc", "mean")].values
cmc_std  = stats[("cmc", "std")].values

mav_mean = stats[("mav_mean", "mean")].values
mav_std  = stats[("mav_mean", "std")].values


# -----------------------------------------------------------
# Literature values (edit or leave as None)
# -----------------------------------------------------------

lit_cmc_mean = [0.996, 0.994, 0.975, 0.964]
lit_cmc_std  = [0.003, 0.005, 0.018, 0.03]

# -----------------------------------------------------------
# Plot
# -----------------------------------------------------------

x = np.arange(len(joint_order))
width = 0.35

fig, ax = plt.subplots(figsize=(7,5))

ax.bar(
    x - width/2,
    cmc_mean,
    width,
    yerr=cmc_std,
    capsize=5,
    label="Current study"
)

ax.bar(
    x + width/2,
    lit_cmc_mean,
    width,
    yerr=lit_cmc_std,
    capsize=5,
    label="Previous literature on human gait [1]", color = 'crimson'
)

ax.set_xticks(x)
ax.set_xticklabels(["Hip flexion","Knee flexion","Ankle flexion","Hip abduction"], fontsize = 15)
ax.set_ylabel("CMC", fontsize = 15)
ax.set_ylim(0,1.5)
ax.legend(fontsize = 11, loc='upper left')
ax.set_title("Coefficient of Multiple Correlation (CMC)", fontsize = 15)
ax.tick_params(axis='both', labelsize=15)

plt.tight_layout()
plt.show()

lit_mav_mean = [19.3, 6.7, 20.4, 7.1]
lit_mav_std  = [1.6, 1.3, 8, 1.5]

fig, ax = plt.subplots(figsize=(7,5))

ax.bar(
    x - width/2,
    mav_mean,
    width,
    yerr=mav_std,
    capsize=5,
    label="Current study"
)

ax.bar(
    x + width/2,
    lit_mav_mean,
    width,
    yerr=lit_mav_std,
    capsize=5,
    label="Previous literature on human gait [2]", color = 'crimson'
)

ax.set_xticks(x)
ax.set_xticklabels(["Hip flexion","Knee flexion","Ankle flexion","Hip abduction"], fontsize = 15)
ax.set_ylabel("MAV (°)", fontsize = 15)
ax.legend(fontsize = 11, loc='upper left')
ax.set_title("Mean Absolute Variability (MAV)", fontsize = 15)
ax.tick_params(axis='both', labelsize=15)

plt.tight_layout()
plt.show()
