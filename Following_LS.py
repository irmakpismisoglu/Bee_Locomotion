# %%
# =========================================
# MEAN FOLLOWING LAG + FOLLOWING STRENGTH
# =========================================
#
# This script calculates the mean following lag across multiple
# following-like events within each video.
#
# It does not assign a leader or follower.
#
# Output:
#   Mean_Following_Lag
#   Mean_Following_Strength
#   Number_Following_Events
#
# Required packages:
#   pip install numpy pandas tqdm openpyxl
#
# =========================================

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==============================
# PATHS
# ==============================
csv_folder = r"I:\sailor_moon\DATA\filtered_csv"

petri_folder = r"I:\sailor_moon\DATA\Petri_Border_dimensions"

output_file = r"I:\sailor_moon\DATA\mean_following_lag_strength_results.xlsx"

# ==============================
# SETTINGS
# ==============================
groups = ["CN", "GR"]

max_lag = 10

dist_thresh = 50

radius_thresh = 30

window_size = 30

step_size = 10

min_points = 20

min_strength = 0.3

# ==============================
# HELPER FUNCTIONS
# ==============================
def get_base(filename):

    base = os.path.basename(filename).split(".csv")[0]

    if "DLC" in base:
        base = base.split("DLC")[0]

    return base.rstrip("_")


def load_petri(base):

    petri_file = os.path.join(
        petri_folder,
        base + "_petri_circles.csv"
    )

    if not os.path.exists(petri_file):
        print("Missing Petri file:", petri_file)
        return None, None

    df = pd.read_csv(petri_file)

    center1 = (
        df.iloc[0]["Center_X"],
        df.iloc[0]["Center_Y"]
    )

    center2 = (
        df.iloc[1]["Center_X"],
        df.iloc[1]["Center_Y"]
    )

    return center1, center2


def compute_polar(x, y, cx, cy):

    x_rel = x - cx
    y_rel = y - cy

    radius = np.sqrt(x_rel**2 + y_rel**2)

    theta = np.arctan2(y_rel, x_rel)

    return radius, theta


def circular_angle_diff(theta):

    dtheta = np.diff(theta)

    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

    return dtheta


def best_lag_in_window(
    dtheta1,
    dtheta2,
    interaction_mask,
    max_lag=10,
    min_points=20
):

    best_lag = np.nan
    best_strength = np.nan
    best_points = 0

    for lag in range(0, max_lag + 1):

        if lag == 0:

            signal1 = dtheta1
            signal2 = dtheta2
            mask = interaction_mask

            signal1 = signal1[mask]
            signal2 = signal2[mask]

            if len(signal1) >= min_points:

                if np.std(signal1) > 0 and np.std(signal2) > 0:

                    corr = np.corrcoef(signal1, signal2)[0, 1]

                    if np.isfinite(corr):
                        best_lag = lag
                        best_strength = corr
                        best_points = len(signal1)

        else:

            signal1_a = dtheta1[lag:]
            signal2_a = dtheta2[:-lag]
            mask_a = interaction_mask[lag:] & interaction_mask[:-lag]

            signal1_a = signal1_a[mask_a]
            signal2_a = signal2_a[mask_a]

            if len(signal1_a) >= min_points:

                if np.std(signal1_a) > 0 and np.std(signal2_a) > 0:

                    corr_a = np.corrcoef(signal1_a, signal2_a)[0, 1]

                    if np.isfinite(corr_a):

                        if np.isnan(best_strength) or corr_a > best_strength:
                            best_lag = lag
                            best_strength = corr_a
                            best_points = len(signal1_a)

            signal1_b = dtheta1[:-lag]
            signal2_b = dtheta2[lag:]
            mask_b = interaction_mask[:-lag] & interaction_mask[lag:]

            signal1_b = signal1_b[mask_b]
            signal2_b = signal2_b[mask_b]

            if len(signal1_b) >= min_points:

                if np.std(signal1_b) > 0 and np.std(signal2_b) > 0:

                    corr_b = np.corrcoef(signal1_b, signal2_b)[0, 1]

                    if np.isfinite(corr_b):

                        if np.isnan(best_strength) or corr_b > best_strength:
                            best_lag = lag
                            best_strength = corr_b
                            best_points = len(signal1_b)

    return best_lag, best_strength, best_points


def compute_mean_following(
    theta1,
    theta2,
    r1,
    r2,
    dist,
    max_lag=10,
    dist_thresh=50,
    radius_thresh=30,
    window_size=30,
    step_size=10,
    min_points=20,
    min_strength=0.3
):

    dtheta1 = circular_angle_diff(theta1)
    dtheta2 = circular_angle_diff(theta2)

    interaction_mask = (
        (dist < dist_thresh) &
        (np.abs(r1 - r2) < radius_thresh)
    )

    interaction_mask = interaction_mask[1:]

    event_lags = []
    event_strengths = []
    event_points = []

    n = len(dtheta1)

    for start in range(0, n - window_size + 1, step_size):

        end = start + window_size

        dtheta1_window = dtheta1[start:end]
        dtheta2_window = dtheta2[start:end]
        mask_window = interaction_mask[start:end]

        lag, strength, points = best_lag_in_window(
            dtheta1_window,
            dtheta2_window,
            mask_window,
            max_lag=max_lag,
            min_points=min_points
        )

        if np.isfinite(lag) and np.isfinite(strength):

            if strength >= min_strength:

                event_lags.append(lag)
                event_strengths.append(strength)
                event_points.append(points)

    if len(event_lags) == 0:

        return np.nan, np.nan, 0, np.nan

    mean_lag = np.mean(event_lags)

    mean_strength = np.mean(event_strengths)

    total_events = len(event_lags)

    mean_points = np.mean(event_points)

    return mean_lag, mean_strength, total_events, mean_points


# ==============================
# MAIN ANALYSIS
# ==============================
results = []

for group in groups:

    files = [
        f for f in os.listdir(csv_folder)
        if f.startswith(group) and f.endswith(".csv")
    ]

    print(f"\nProcessing {group}: {len(files)} files")

    for file in tqdm(files, desc=f"{group} files"):

        file_path = os.path.join(csv_folder, file)

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        base = get_base(file)

        center1, center2 = load_petri(base)

        if center1 is None:
            continue

        try:
            x1_col = [
                c for c in df.columns
                if "B1_ABD" in c and c.endswith("_x")
            ][0]

            y1_col = [
                c for c in df.columns
                if "B1_ABD" in c and c.endswith("_y")
            ][0]

            x2_col = [
                c for c in df.columns
                if "B2_ABD" in c and c.endswith("_x")
            ][0]

            y2_col = [
                c for c in df.columns
                if "B2_ABD" in c and c.endswith("_y")
            ][0]

        except IndexError:
            print("Missing bee coordinate columns:", file)
            continue

        x1 = df[x1_col].values
        y1 = df[y1_col].values

        x2 = df[x2_col].values
        y2 = df[y2_col].values

        valid_mask = (
            np.isfinite(x1) &
            np.isfinite(y1) &
            np.isfinite(x2) &
            np.isfinite(y2)
        )

        x1 = x1[valid_mask]
        y1 = y1[valid_mask]
        x2 = x2[valid_mask]
        y2 = y2[valid_mask]

        if len(x1) < window_size + max_lag + 1:
            continue

        for dish_name, center in [
            ("dish1", center1),
            ("dish2", center2)
        ]:

            r1, theta1 = compute_polar(x1, y1, *center)
            r2, theta2 = compute_polar(x2, y2, *center)

            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            mean_lag, mean_strength, n_events, mean_points = compute_mean_following(
                theta1,
                theta2,
                r1,
                r2,
                dist,
                max_lag=max_lag,
                dist_thresh=dist_thresh,
                radius_thresh=radius_thresh,
                window_size=window_size,
                step_size=step_size,
                min_points=min_points,
                min_strength=min_strength
            )

            results.append({
                "File": file,
                "Group": group,
                "Dish": dish_name,
                "Mean_Following_Lag": mean_lag,
                "Mean_Following_Strength": mean_strength,
                "Number_Following_Events": n_events,
                "Mean_Points_Per_Event": mean_points,
                "Window_Size": window_size,
                "Step_Size": step_size,
                "Min_Strength": min_strength
            })


# ==============================
# SAVE RESULTS
# ==============================
df_results = pd.DataFrame(results)

df_results.to_excel(output_file, index=False)

print("\nFinished!")
print("Results saved to:")
print(output_file)
