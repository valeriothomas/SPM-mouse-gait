# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 17:11:19 2025

@author: md1tva
"""

import os
import re
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.interpolate import PchipInterpolator

def get_file_numbers(folder, prefix, suffix):
    pattern = re.compile(rf"{re.escape(prefix)}(\d+){re.escape(suffix)}$")

    numbers = []

    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            numbers.append(int(match.group(1)))

    return sorted(numbers)

def butter_lowpass_filter_1d(x, cutoff, fs, order=4):
    """
    Low-pass Butterworth filter for 1D signal x.
    NaNs in x are preserved: we linearly interpolate for filtering then
    restore NaNs to their original positions.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    nan_mask = np.isnan(x)
    if np.all(nan_mask):
        return x.copy()

    # Interpolate over NaNs for filtering
    valid_idx = np.flatnonzero(~nan_mask)
    if valid_idx.size == 0:
        return x.copy()

    interp_x = x.copy()
    # if first or last values are NaN, fill using nearest valid (so interpolation works)
    if valid_idx[0] != 0:
        interp_x[:valid_idx[0]] = x[valid_idx[0]]
    if valid_idx[-1] != n - 1:
        interp_x[valid_idx[-1] + 1:] = x[valid_idx[-1]]

    # linear interpolation for internal NaNs
    interp_x[nan_mask] = np.interp(
        np.flatnonzero(nan_mask),
        valid_idx,
        x[valid_idx])

    # design filter
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # filtfilt
    filtered = filtfilt(b, a, interp_x)

    # restore NaNs
    filtered[nan_mask] = np.nan
    return filtered

def butter_lowpass_filter_1d_2(x, cutoff, fs, order=4):
    x = np.asarray(x, dtype=float)
    nan_mask = np.isnan(x)

    if np.all(nan_mask):
        return x.copy()

    valid_idx = np.flatnonzero(~nan_mask)
    interp_x = x.copy()

    # edge handling
    interp_x[:valid_idx[0]] = x[valid_idx[0]]
    interp_x[valid_idx[-1]+1:] = x[valid_idx[-1]]

    # PCHIP interpolation
    interp = PchipInterpolator(valid_idx, x[valid_idx])
    interp_x[nan_mask] = interp(np.flatnonzero(nan_mask))

    # filter
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    filtered = filtfilt(b, a, interp_x)

    # restore NaNs
    filtered[nan_mask] = np.nan
    return filtered

def normalize(v):
    """Normalize each vector in an array of shape (n,3)."""
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norm

def sagittal_projection(v, plane='xy'):
    """
    Project vectors onto the sagittal plane.
    Assuming mouse walks in XY plane.
    Remove X (lateral axis) -> keep Y and Z.
    """
    v_proj = v.copy()
    #v_proj[:, 0] = 0   # remove X component
    return v_proj

def angle_between(u, v):
    """Compute angle between two sets of vectors using dot product."""
    # Normalize the vectors
    u_norm = u / np.linalg.norm(u, axis=1, keepdims=True)
    v_norm = v / np.linalg.norm(v, axis=1, keepdims=True)
    
    # Dot product
    dot = np.sum(u_norm * v_norm, axis=1)
    dot = np.clip(dot, -1.0, 1.0)  # numerical safety
    
    return np.arccos(dot)          # radians # radians

def compute_pelvis_rotation(Spine, TAIL):
    # Segment vectors
    Pelvis = Spine - TAIL
    
    # Ground reference (-y)
    n_frame = len(Spine)
    Ground_vector = np.zeros((n_frame,3))
    Ground_vector[:,1] = -1
    
    # Compute angle
    Pelvis_angle = angle_between(Pelvis,Ground_vector)
    
    # Convert to degrees
    return np.degrees(Pelvis_angle)

def compute_angles_side(LAsis, FH, Knee, Ankle, MT):
    # Segment vectors
    Femur =   Knee  - FH     # hip → knee
    Tibia =   Ankle - Knee   # knee → ankle
    Foot  =   MT    - Ankle  # ankle → toe

    # Hip reference vector (pelvis → femoral head)
    PelvisCenter = LAsis
    PelvisToHip = FH - PelvisCenter

    # Project to sagittal plane
    Femur_s = sagittal_projection(Femur)
    Tibia_s = sagittal_projection(Tibia)
    Foot_s  = sagittal_projection(Foot)
    PelvisToHip_s = sagittal_projection(PelvisToHip)

    # Compute joint angles
    hip_angle   = angle_between(PelvisToHip_s, Femur_s)
    knee_angle  = angle_between(Femur_s, Tibia_s)
    ankle_angle = angle_between(Tibia_s, Foot_s)

    # Convert to degrees
    return np.degrees(hip_angle), np.degrees(knee_angle), np.degrees(ankle_angle)

def interpolate_nans(y):
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    y_interp = np.interp(x, x[mask], y[mask])
    return y_interp

def interpolate_internal_nans(arr):
    """
    Interpolate only internal NaNs in each column of an array.
    Leading or trailing NaNs are kept as NaN.
    
    Parameters
    ----------
    arr : ndarray, shape (N, M)
        Input array with NaNs to interpolate.
    
    Returns
    -------
    out : ndarray, shape (N, M)
        Copy of arr with internal NaNs replaced by linear interpolation.
    """
    arr = np.asarray(arr, float)
    out = arr.copy()
    
    n, m = arr.shape

    for j in range(m):  # for each coordinate (x, y, z)
        col = arr[:, j]
        mask = np.isnan(col)

        # if no NaNs or all NaNs, skip
        if mask.sum() == 0 or mask.sum() == n:
            continue

        valid_idx = np.where(~mask)[0]    # indices of non-NaN values
        nan_idx   = np.where(mask)[0]     # indices of NaN values

        # If the first valid index is not 0, leave leading NaNs unchanged
        first_valid = valid_idx[0]
        # If the last valid index is not n-1, leave trailing NaNs unchanged
        last_valid  = valid_idx[-1]

        # Only interpolate NaNs between first_valid and last_valid
        internal_nan_mask = (nan_idx > first_valid) & (nan_idx < last_valid)
        internal_nan_idx = nan_idx[internal_nan_mask]

        if internal_nan_idx.size > 0:
            out[internal_nan_idx, j] = np.interp(
                internal_nan_idx,          # x: positions to fill
                valid_idx,                 # xp: known good positions
                col[valid_idx]             # fp: known values
            )

    return out

def detect_stance_phases_foot(foot_xyz: np.ndarray, fs: float):
    """
    Detect stance phases using filtered 3D foot marker coordinates AND
    exclude those with displacement below a given threshold.

    Parameters
    ----------
    foot_xyz : ndarray, shape (N, 3)
        Filtered 3D marker coordinates of the foot.
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    stance_events : ndarray, shape (M, 2)
        Each row is [initial_contact_index, toe_off_index].
    """

    # --------- 1. EXTRACT VERTICAL COORDINATE ---------
    y = foot_xyz[:, 1]
    
    # --------- 2. COMPUTE VERTICAL VELOCITY ---------
    vy = np.gradient(y, 1/fs)

    # --------- 3. LOW VERTICAL SPEED CRITERION ---------
    speed_thresh = np.nanmean(np.abs(vy))
    low_speed = np.abs(vy) < speed_thresh

    # --------- 4. STANCE = LOW VERTICAL SPEED ---------
    stance = low_speed

    # --------- 5. FIND TRANSITIONS ---------
    stance_diff = np.diff(stance.astype(int))

    IC = np.where(stance_diff == 1)[0] + 1      # 0 → 1
    TO = np.where(stance_diff == -1)[0] + 1     # 1 → 0

    # --------- 6. ALIGN IC AND TO ---------
    if len(TO) > 0 and len(IC) > 0:
        if TO[0] < IC[0]:
            TO = TO[1:]

    min_len = min(len(IC), len(TO))
    IC = IC[:min_len]
    TO = TO[:min_len]

    stance_events = np.column_stack((IC, TO))
    
    # --------- 8. FILTER STANCE EVENTS BY LENGTH TO ELIMINATE STATIC POSITIONS ---------
    valid_events = []

    for start, end in stance_events:
        if (end-start<30) & (end-start>9):
            valid_events.append([start, end])

    return np.array(valid_events)

def compute_velocity(marker_xyz, sampling_rate):
    """
    Compute velocity from a marker trajectory.
    
    Parameters
    ----------
    marker_xyz : array (nFrames, 3)
        XYZ coordinates of the marker (e.g., femoral head).
    sampling_rate : float
        Motion capture sampling rate (Hz).
    
    Returns
    -------
    vel_vector : array (nFrames, 3)
        Instantaneous velocity vector (m/s).
    speed : array (nFrames,)
        Instantaneous speed (m/s).
    """
    dt = 1.0 / sampling_rate

    # Compute time derivative using central differences
    vel_vector = np.zeros_like(marker_xyz)
    vel_vector[1:-1] = (marker_xyz[2:] - marker_xyz[:-2]) / (2 * dt)

    # Use forward/backward difference at edges
    vel_vector[0] = (marker_xyz[1] - marker_xyz[0]) / dt
    vel_vector[-1] = (marker_xyz[-1] - marker_xyz[-2]) / dt

    # Speed = magnitude of velocity vector
    speed = np.linalg.norm(vel_vector, axis=1)

    return vel_vector, speed

def hip_abduction_angle(right_crest, hip, knee, left_crest):
    """
    Compute hip abduction angle (deg) for each frame.
    
    Parameters
    ----------
    right_crest : ndarray (n_frames, 3)
    left_crest  : ndarray (n_frames, 3)
    hip         : ndarray (n_frames, 3)
    knee        : ndarray (n_frames, 3)

    Returns
    -------
    angles_deg : ndarray (n_frames,)
        Hip abduction/adduction angle in degrees.
    """

    # Femur vector (hip → knee)
    femur = knee - hip   # shape (n,3)

    # Crest line vector (right → left)
    crest = left_crest - right_crest

    # Calculate the angle
    abduction = angle_between(crest,femur)

    return np.degrees(abduction)-90

def process_hindlimb_markers(
    markers,                 # dict: {'IC': Nx3, 'HIP': Nx3, ...}
    fs=200,                  # sampling frequency (Hz)
    v_max=1000,              # max marker velocity (mm/s)
    length_tol=3,            # standard deviation of segment length tolerance (sd)
    min_valid_run=1,         # minimum valid segment (frames)
    max_interp_gap=20,       # max gap to interpolate (frames)
    cutoff=20,               # low-pass cutoff (Hz)
    filter_order=4
):
    """
    Full MoCap post-processing pipeline:
    - velocity-based outlier rejection
    - bone-length consistency enforcement
    - short valid run removal
    - PCHIP interpolation (short gaps only)
    - zero-lag low-pass filtering

    NaNs are handled safely at every step.
    """

    # ---------- Helper functions ----------

    def reject_velocity_spikes(marker):
        marker = marker.copy()
        dt = 1 / fs
        n = len(marker)

        for i in range(1, n - 1):
            if np.any(np.isnan(marker[i])):
                continue

            v_prev = 0
            v_next = 0

            if not np.any(np.isnan(marker[i - 1])):
                v_prev = np.linalg.norm(marker[i] - marker[i - 1]) / dt
            if not np.any(np.isnan(marker[i + 1])):
                v_next = np.linalg.norm(marker[i + 1] - marker[i]) / dt

            if max(v_prev, v_next) > v_max:
                marker[i, :] = np.nan

        return marker

    def enforce_bone_length(m1, m2):
        d = np.linalg.norm(m1 - m2, axis=1)
        d_ref = np.nanmedian(d)
        d_std = np.nanstd(d)

        bad = np.abs(d - d_ref) > length_tol * d_std
        m1[bad] = np.nan
        m2[bad] = np.nan
        return m1, m2

    def remove_short_valid_runs(marker):
        valid = ~np.isnan(marker[:, 0])
        idx = np.where(valid)[0]
        if len(idx) == 0:
            return marker

        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for r in runs:
            if len(r) < min_valid_run:
                marker[r] = np.nan
        return marker

    def interpolate_short_gaps(marker):
        marker = marker.copy()
        n = len(marker)

        for dim in range(3):
            x = marker[:, dim]
            nan = np.isnan(x)

            if np.all(nan):
                continue

            idx = np.where(~nan)[0]
            interp = PchipInterpolator(idx, x[idx])

            gap_starts = np.where((nan[:-1] == False) & (nan[1:] == True))[0] + 1
            gap_ends   = np.where((nan[:-1] == True) & (nan[1:] == False))[0] + 1

            if nan[0]:
                gap_starts = np.r_[0, gap_starts]
            if nan[-1]:
                gap_ends = np.r_[gap_ends, n]

            for s, e in zip(gap_starts, gap_ends):
                if e - s <= max_interp_gap:
                    x[s:e] = interp(np.arange(s, e))

            marker[:, dim] = x

        return marker

    def butter_lowpass(marker):
        marker_f = marker.copy()
        nyq = 0.5 * fs
        b, a = butter(filter_order, cutoff / nyq, btype='low')

        for dim in range(3):
            x = marker[:, dim]
            valid = ~np.isnan(x)

            if np.sum(valid) < filter_order * 3:
                continue

            x_f = x.copy()
            x_f[~valid] = np.interp(
                np.where(~valid)[0],
                np.where(valid)[0],
                x[valid]
            )

            x_f = filtfilt(b, a, x_f)
            x_f[~valid] = np.nan
            marker_f[:, dim] = x_f

        return marker_f

    # ---------- PIPELINE ----------

    # 1. Velocity-based rejection
    for k in markers:
        markers[k] = reject_velocity_spikes(markers[k])

    # 2. Bone-length consistency
    chain = ['RASI','RightHIP','RightKNEE','RightANKLE','RightMET']
    for a, b in zip(chain[:-1], chain[1:]):
        markers[a], markers[b] = enforce_bone_length(markers[a], markers[b])
    
    chain_left = ['LASI', 'LEFT_HIP','LeftKNEE','LeftANKLE','LeftMET']
    for a, b in zip(chain_left[:-1], chain_left[1:]):
        markers[a], markers[b] = enforce_bone_length(markers[a], markers[b])

    # 3. Remove short valid bursts
    for k in markers:
        markers[k] = remove_short_valid_runs(markers[k])

    # 4. Interpolate short gaps
    for k in markers:
        markers[k] = interpolate_short_gaps(markers[k])

    # 5. Low-pass filter
    for k in markers:
        markers[k] = butter_lowpass(markers[k])

    return markers


    
    