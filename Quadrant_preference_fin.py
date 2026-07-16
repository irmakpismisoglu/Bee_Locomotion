import pandas as pd
import numpy as np
import os

# -----------------------------
# PATHS (REPLACE WITH YOUR DATA DIRECTORY)
# -----------------------------
csv_folder = "PATH_TO_TRACKING_DATA"
petri_folder = "PATH_TO_PETRI_CIRCLE_DATA"

bees = ['B1', 'B2']
bodypart = 'ABD'

# -----------------------------
# Extract base name for matching Petri dish file
# -----------------------------
def get_base_for_petri(filename):
    base = os.path.basename(filename).split(".csv")[0]
    if "DLC" in base:
        base = base.split("DLC")[0]
    return base.rstrip("_")

# -----------------------------
# Load Petri dish centers
# -----------------------------
def load_petri(base):
    file = os.path.join(petri_folder, base + "_petri_circles.csv")
    if not os.path.exists(file):
        return None, None

    df = pd.read_csv(file)
    c1 = (df.iloc[0]["Center_X"], df.iloc[0]["Center_Y"])
    c2 = (df.iloc[1]["Center_X"], df.iloc[1]["Center_Y"])
    return c1, c2

# -----------------------------
# Quadrant classification
# -----------------------------
def classify_quadrant(x, y, cx, cy):
    if x < cx and y < cy:
        return "TopLeft"
    elif x >= cx and y < cy:
        return "TopRight"
    elif x < cx and y >= cy:
        return "BottomLeft"
    else:
        return "BottomRight"

# -----------------------------
# Analyze one video file
# -----------------------------
def analyze_video(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    base_petri = get_base_for_petri(file_path)
    center1, center2 = load_petri(base_petri)

    if center1 is None:
        return None

    bee_counts = {
        'B1': {"dish1": {q: 0 for q in ["TopLeft","TopRight","BottomLeft","BottomRight"]},
               "dish2": {q: 0 for q in ["TopLeft","TopRight","BottomLeft","BottomRight"]}},
        'B2': {"dish1": {q: 0 for q in ["TopLeft","TopRight","BottomLeft","BottomRight"]},
               "dish2": {q: 0 for q in ["TopLeft","TopRight","BottomLeft","BottomRight"]}}
    }

    x_b1_col = [c for c in df.columns if 'B1_ABD' in c and c.endswith('_x')]
    y_b1_col = [c for c in df.columns if 'B1_ABD' in c and c.endswith('_y') or '.1_y' in c]
    x_b2_col = [c for c in df.columns if 'B2_ABD' in c and c.endswith('_x')]
    y_b2_col = [c for c in df.columns if 'B2_ABD' in c and c.endswith('_y') or '.1_y' in c]

    if not (x_b1_col and y_b1_col and x_b2_col and y_b2_col):
        return None

    coords = [
        ('B1', df[x_b1_col[0]].values, df[y_b1_col[0]].values),
        ('B2', df[x_b2_col[0]].values, df[y_b2_col[0]].values)
    ]

    for bee, x_vals, y_vals in coords:
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_vals, y_vals = x_vals[mask], y_vals[mask]

        for xi, yi in zip(x_vals, y_vals):
            d1 = abs(xi - center1[0]) + abs(yi - center1[1])
            d2 = abs(xi - center2[0]) + abs(yi - center2[1])

            dish = "dish1" if d1 <= d2 else "dish2"
            cx, cy = (center1 if dish == "dish1" else center2)

            q = classify_quadrant(xi, yi, cx, cy)
            bee_counts[bee][dish][q] += 1

    return bee_counts

# -----------------------------
# Process group of videos
# -----------------------------
def collect_group_data(folder, prefix):
    files = [f for f in os.listdir(folder)
             if f.endswith(".csv") and f.startswith(prefix)]

    results = []
    file_names = []

    for f in files:
        res = analyze_video(os.path.join(folder, f))
        if res:
            results.append(res)
            file_names.append(f)

    return results, file_names

# -----------------------------
# RUN ANALYSIS
# -----------------------------
cn_counts, cn_files = collect_group_data(csv_folder, "CN")
gr_counts, gr_files = collect_group_data(csv_folder, "GR")

# -----------------------------
# EXPORT RESULTS
# -----------------------------
data = []

for group_name, counts_list, files in zip(
    ['CN', 'GR'],
    [cn_counts, gr_counts],
    [cn_files, gr_files]
):
    for video_counts, file_path in zip(counts_list, files):
        dyad = os.path.splitext(os.path.basename(file_path))[0]

        for bee in ['B1', 'B2']:
            for dish in ['dish1', 'dish2']:

                total = sum(video_counts[bee][dish].values())

                row = {
                    'Group': group_name,
                    'Dyad': dyad,
                    'Bee': bee,
                    'Dish': dish
                }

                for q in ['TopLeft','TopRight','BottomLeft','BottomRight']:
                    row[q] = (video_counts[bee][dish][q] / total * 100) if total > 0 else 0

                data.append(row)

df_out = pd.DataFrame(data)
df_out.to_excel("quadrant_preference_per_bee.xlsx", index=False)

print("Analysis complete: quadrant preference exported.")