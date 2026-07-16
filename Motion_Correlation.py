# =========================================================
# INTERACTION ANALYSIS PIPELINE
# Following, synchrony, and interaction-conditioned coupling
# =========================================================

import pandas as pd
import numpy as np
import os
from tqdm import tqdm


# =========================================================
# DATA PATHS
# =========================================================
# csv_folder:
#   Contains DeepLabCut tracking output (one file per dyad)
#
# petri_folder:
#   Contains calibration files defining spatial reference
#
# output_file:
#   Final dataset for statistical analysis (e.g., GLMM in R)

csv_folder = "PATH_TO_FILTERED_TRACKING_DATA"
petri_folder = "PATH_TO_PETRI_CALIBRATION_DATA"
output_file = "bee_interaction_analysis.xlsx"


# =========================================================
# FILE IDENTIFICATION FUNCTION
# =========================================================
# Extracts a clean base name from tracking files
# Used to match each video with its corresponding Petri calibration file

def get_base(filename):
    base = os.path.basename(filename).split(".csv")[0]

    # Remove DeepLabCut suffix if present
    if "DLC" in base:
        base = base.split("DLC")[0]

    return base.rstrip("_")


# =========================================================
# LOAD PETRI DISH CENTERS
# =========================================================
# Each experiment contains two spatial reference points (two dishes)
# These define the coordinate system for spatial normalization

def load_petri(base):
    file_path = os.path.join(petri_folder, base + "_petri_circles.csv")

    if not os.path.exists(file_path):
        return None, None

    df = pd.read_csv(file_path)

    # First row = dish 1 center
    # Second row = dish 2 center
    c1 = (df.iloc[0]["Center_X"], df.iloc[0]["Center_Y"])
    c2 = (df.iloc[1]["Center_X"], df.iloc[1]["Center_Y"])

    return c1, c2


# =========================================================
# CARTESIAN → POLAR TRANSFORMATION
# =========================================================
# Converts spatial coordinates into a circular coordinate system
#
# r (radial distance):
#   distance from dish center → spatial position in arena
#
# theta (angular position):
#   direction around center → directional locomotion state

def compute_polar(x, y, cx, cy):
    x_rel = x - cx
    y_rel = y - cy

    r = np.sqrt(x_rel**2 + y_rel**2)
    theta = np.arctan2(y_rel, x_rel)

    return r, theta


# =========================================================
# ANGULAR VELOCITY (ROTATIONAL MOVEMENT)
# =========================================================
# Computes change in angular position across frames
# Circular correction ensures continuity at ±π boundary

def angular_velocity(theta):
    dtheta = np.diff(theta)

    # Wrap angular differences to [-π, π]
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

    return dtheta


# =========================================================
# INTERACTION-CONDITIONED FOLLOWING ANALYSIS
# =========================================================
# Estimates directional coupling between individuals using:
#   - temporal lag correlation of angular trajectories
#
# IMPORTANT:
# Analysis is restricted to frames where bees are interacting:
#   1. physical proximity (distance threshold)
#   2. similar radial position (same spatial zone)
#
# This reduces false positives from independent movement.

def compute_following(theta1, theta2, r1, r2, dist,
                       max_lag=20,
                       dist_thresh=50,
                       radius_thresh=30):

    best_corr = -np.inf
    best_lag = np.nan

    # ---------------------------------------------------------
    # INTERACTION FILTER
    # ---------------------------------------------------------
    # Select frames where bees are sufficiently close
    # and occupy similar radial zones within the arena

    spatial_mask = (
        (dist < dist_thresh) &
        (np.abs(r1 - r2) < radius_thresh)
    )

    # Require minimum interaction data for reliable estimation
    if np.sum(spatial_mask) < 10:
        return np.nan, np.nan

    theta1_f = theta1[spatial_mask]
    theta2_f = theta2[spatial_mask]

    # ---------------------------------------------------------
    # LAGGED CROSS-CORRELATION
    # ---------------------------------------------------------
    # Tests whether one bee's angular movement predicts the other
    # at different temporal shifts (lead–lag structure)

    for lag in range(-max_lag, max_lag + 1):

        if lag < 0:
            t1 = theta1_f[:lag]
            t2 = theta2_f[-lag:]

        elif lag > 0:
            t1 = theta1_f[lag:]
            t2 = theta2_f[:-lag]

        else:
            t1 = theta1_f
            t2 = theta2_f

        if len(t1) < 5:
            continue

        corr = np.corrcoef(t1, t2)[0, 1]

        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag

    return best_lag, best_corr


# =========================================================
# STORAGE STRUCTURE
# =========================================================
# results:
#   final dataset for statistical modeling (GLMM in R)
#
# Each row = one dyad × one dish × one video

results = []


# =========================================================
# MAIN ANALYSIS LOOP
# =========================================================
# Processes both experimental groups:
#   CN = control
#   GR = germ-reduced

for group in ["CN", "GR"]:

    files = [
        f for f in os.listdir(csv_folder)
        if f.startswith(group) and f.endswith(".csv")
    ]

    print(f"\nProcessing group: {group}")

    for f in tqdm(files):

        path = os.path.join(csv_folder, f)
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        base = get_base(f)
        c1, c2 = load_petri(base)

        if c1 is None:
            continue

        # -----------------------------------------------------
        # EXTRACT TRACKING DATA
        # -----------------------------------------------------
        # Abdomen coordinates are used as proxy for body position

        x1 = df[[c for c in df.columns if "B1_ABD" in c and c.endswith("_x")][0]].values
        y1 = df[[c for c in df.columns if "B1_ABD" in c and c.endswith("_y")][0]].values
        x2 = df[[c for c in df.columns if "B2_ABD" in c and c.endswith("_x")][0]].values
        y2 = df[[c for c in df.columns if "B2_ABD" in c and c.endswith("_y")][0]].values

        # -----------------------------------------------------
        # DATA CLEANING
        # -----------------------------------------------------
        # Remove frames with missing tracking values for either bee

        mask = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(x2) & np.isfinite(y2)

        x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]

        # Ensure sufficient temporal length for reliable statistics
        if len(x1) < 20:
            continue

        # -----------------------------------------------------
        # ANALYSIS PER PETRI DISH
        # -----------------------------------------------------
        # Each recording is analyzed twice:
        #   - relative to dish 1 center
        #   - relative to dish 2 center

        for dish_name, center in [("dish1", c1), ("dish2", c2)]:

            # Convert to polar coordinate system
            r1, theta1 = compute_polar(x1, y1, *center)
            r2, theta2 = compute_polar(x2, y2, *center)

            # Spatial interaction measures
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            radius_diff = np.abs(r1 - r2)

            # Angular alignment (circular difference)
            dtheta = theta1 - theta2
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

            # Movement dynamics
            w1 = angular_velocity(theta1)
            w2 = angular_velocity(theta2)

            min_len = min(len(w1), len(w2))
            w_corr = np.corrcoef(w1[:min_len], w2[:min_len])[0, 1]

            # -------------------------------------------------
            # FOLLOWING ANALYSIS (INTERACTION-CONDITIONED)
            # -------------------------------------------------
            lag, lag_corr = compute_following(
                theta1, theta2,
                r1, r2,
                dist
            )

            # -------------------------------------------------
            # STORE RESULTS
            # -------------------------------------------------
            results.append({
                "File": f,
                "Group": group,
                "Dish": dish_name,

                "Mean_Bee_Distance": np.mean(dist),
                "Std_Bee_Distance": np.std(dist),

                "Mean_Radius_Diff": np.mean(radius_diff),
                "Mean_Angle_Diff": np.mean(np.abs(dtheta)),

                "Angular_Velocity_Corr": w_corr,

                "Following_Lag": lag,
                "Following_Strength": lag_corr
            })


# =========================================================
# EXPORT FOR STATISTICAL ANALYSIS (GLMM / R)
# =========================================================

df_out = pd.DataFrame(results)
df_out.to_excel(output_file, index=False)

print("Analysis completed. Dataset exported.")