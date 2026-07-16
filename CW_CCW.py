
"""
Rotation Preference Analysis Script
===================================

This script analyzes rotational movement preference from filtered bee tracking
CSV files.

It calculates whether each bee moves more clockwise or counterclockwise around
the Petri dish centers.

The script uses filtered coordinate files, Petri dish center positions, and
then creates:

1. An Excel file containing clockwise and counterclockwise rotation values.
2. A mixed-effects statistical model.
3. Bar plots comparing clockwise and counterclockwise movement.
4. A rotation bias plot.

Required Python packages
------------------------

Install the required packages before running the script:

    pip install pandas numpy statsmodels seaborn matplotlib tqdm openpyxl

Folder structure
----------------

The script expects these folders to exist:

    filtered_csv
    Petri_Border_dimensions

By default, these paths are written as:

    csv_folder = r"filtered_csv"
    petri_folder = r"Petri_Border_dimensions"

This means the script expects both folders to be in the same location where
the Python script is being run.

Example folder structure:

    project_folder/
    ├── rotation_analysis.py
    ├── filtered_csv/
    └── Petri_Border_dimensions/

If your folders are somewhere else, change these paths.

Example:

    csv_folder = r"C:\Users\YourName\Documents\bee_project\filtered_csv"
    petri_folder = r"C:\Users\YourName\Documents\bee_project\Petri_Border_dimensions"

Input files
-----------

1. Filtered coordinate CSV files

The script reads filtered tracking CSV files from:

    filtered_csv

These files should contain abdomen coordinates for two bees.

By default, the script searches for columns containing:

    B1_ABD
    B2_ABD

and ending with:

    _x
    _y

Example expected column names:

    B1_ABD_x
    B1_ABD_y
    B2_ABD_x
    B2_ABD_y

If your coordinate files use different column names, change this section:

    x_b1 = [
        c for c in df.columns
        if 'B1_ABD' in c and c.endswith('_x')
    ][0]

    y_b1 = [
        c for c in df.columns
        if 'B1_ABD' in c and '_y' in c
    ][0]

    x_b2 = [
        c for c in df.columns
        if 'B2_ABD' in c and c.endswith('_x')
    ][0]

    y_b2 = [
        c for c in df.columns
        if 'B2_ABD' in c and '_y' in c
    ][0]

For example, if your columns are:

    B1_x
    B1_y
    B2_x
    B2_y

you can change the code to:

    x_b1 = "B1_x"
    y_b1 = "B1_y"
    x_b2 = "B2_x"
    y_b2 = "B2_y"

2. Petri dish CSV files

The script reads Petri dish border files from:

    Petri_Border_dimensions

Each Petri file should be named like this:

    video_name_petri_circles.csv

For example, if the tracking file is:

    CN01_ABD_filtered_border.csv

the script will try to match it to:

    CN01_ABD_filtered_border_petri_circles.csv

Important:
If your Petri files are named using the original video name, but your filtered
CSV files include extra text such as "_ABD_filtered_border", you may need to
edit the get_base_for_petri() function so the names match correctly.

Each Petri file must contain:

    Center_X
    Center_Y

for both:

    Petri_Dish_1
    Petri_Dish_2

Example Petri CSV:

    ,Center_X,Center_Y,Radius
    Petri_Dish_1,250,300,180
    Petri_Dish_2,650,300,178

Output files
------------

The main output file is:

    rotation_preference.xlsx

This Excel file contains, for each bee and dish:

    Bee
    Dish
    CW
    CCW
    CW_percent
    CCW_percent
    Group
    Dyad
    Subject

Column meanings
---------------

Bee:
    Bee identity, usually B1 or B2.

Dish:
    Which Petri dish center was used for rotation calculation.

CW:
    Number of clockwise angular movements.

CCW:
    Number of counterclockwise angular movements.

CW_percent:
    Percentage of movements that were clockwise.

CCW_percent:
    Percentage of movements that were counterclockwise.

Group:
    Experimental group, based on filename prefix.

Dyad:
    Video or pair name.

Subject:
    Unique bee identifier made from dyad name and bee name.

Group names
-----------

By default, the script analyzes two groups:

    CN
    GR

This is controlled by:

    groups = ["CN","GR"]

The script only processes files that start with these prefixes.

If your groups have different names, change this line.

Example:

    groups = ["Control", "Treatment"]

The script also uses these colors for plotting:

    palette = {
        "CN":"#1f77b4",
        "GR":"#d62728"
    }

If you change the group names, also change the palette.

Example:

    palette = {
        "Control":"#1f77b4",
        "Treatment":"#d62728"
    }

How rotation is calculated
--------------------------

For each bee, the script calculates its position relative to a Petri dish
center.

It then converts each x/y coordinate into an angle using:

    np.arctan2(y_rel, x_rel)

The script compares the angle from one frame to the next.

If the angle change is negative, the movement is counted as clockwise.

If the angle change is positive, the movement is counted as counterclockwise.

The script corrects angle wrapping so movement across -pi and +pi is handled
properly.

Mixed-effects model
-------------------

The script creates a long-format dataframe and runs this model:

    Percentage ~ Group * Dish * Direction

with:

    Subject

as the grouping

# Import required libraries
import pandas as pd                     # Data handling
import numpy as np                      # Numerical operations
import os                               # File management
import statsmodels.formula.api as smf   # Mixed-effects statistical models
import seaborn as sns                   # Statistical plotting
import matplotlib.pyplot as plt         # Visualization
from tqdm import tqdm                   # Progress bar

# ================================
# Define file paths
# ================================
csv_folder = r"filtered_csv"

# Folder containing Petri dish dimensions
petri_folder = r"Petri_Border_dimensions"

# Output Excel file
output_excel = r"rotation_preference.xlsx"

# ================================
# Helper function:
# extract base file name
# ================================
def get_base_for_petri(filename):

    # Remove CSV extension
    base = os.path.basename(filename).split(".csv")[0]

    # Remove DLC suffix if present
    if "DLC" in base:
        base = base.split("DLC")[0]

    return base.rstrip("_")

# ================================
# Load Petri dish center positions
# ================================
def load_petri(base):

    # Build Petri file path
    petri_file = os.path.join(
        petri_folder,
        base + "_petri_circles.csv"
    )

    # Check whether file exists
    if not os.path.exists(petri_file):

        print("Petri CSV not found:", petri_file)

        return None, None

    # Load Petri dimensions
    df = pd.read_csv(petri_file)

    # Extract center of first dish
    center1 = (
        df.loc[df.iloc[:,0]=="Petri_Dish_1", "Center_X"].values[0],
        df.loc[df.iloc[:,0]=="Petri_Dish_1", "Center_Y"].values[0]
    )

    # Extract center of second dish
    center2 = (
        df.loc[df.iloc[:,0]=="Petri_Dish_2", "Center_X"].values[0],
        df.loc[df.iloc[:,0]=="Petri_Dish_2", "Center_Y"].values[0]
    )

    return center1, center2

# ================================
# Compute rotational movement
# ================================
def compute_rotation_vec(x, y, cx, cy):

    # Convert coordinates relative to dish center
    x_rel = x - cx
    y_rel = y - cy

    # Compute angular position
    angles = np.arctan2(y_rel, x_rel)

    # Compute angular change between frames
    dtheta = np.diff(angles)

    # Correct angle wrapping
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

    # Count clockwise movement
    cw = np.sum(dtheta < 0)

    # Count counterclockwise movement
    ccw = np.sum(dtheta > 0)

    return cw, ccw

# ================================
# Analyze one video
# ================================
def analyze_video_both_dishes(file_path):

    # Load coordinate data
    df = pd.read_csv(file_path)

    # Get matching Petri file
    base_petri = get_base_for_petri(file_path)

    # Load Petri centers
    center1, center2 = load_petri(base_petri)

    # Skip if missing
    if center1 is None:
        return None

    # -----------------------------
    # Find bee coordinate columns
    # -----------------------------
    x_b1 = [
        c for c in df.columns
        if 'B1_ABD' in c and c.endswith('_x')
    ][0]

    y_b1 = [
        c for c in df.columns
        if 'B1_ABD' in c and '_y' in c
    ][0]

    x_b2 = [
        c for c in df.columns
        if 'B2_ABD' in c and c.endswith('_x')
    ][0]

    y_b2 = [
        c for c in df.columns
        if 'B2_ABD' in c and '_y' in c
    ][0]

    results = []

    # -----------------------------
    # Analyze both bees
    # -----------------------------
    for bee, x_vals, y_vals in [

        ('B1', df[x_b1].values, df[y_b1].values),

        ('B2', df[x_b2].values, df[y_b2].values)

    ]:

        # Remove missing coordinates
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)

        x_vals = x_vals[mask]
        y_vals = y_vals[mask]

        # Skip if too few points
        if len(x_vals) < 2:
            continue

        # =========================
        # Analyze rotation around
        # dish 1
        # =========================
        cw1, ccw1 = compute_rotation_vec(
            x_vals,
            y_vals,
            *center1
        )

        total1 = cw1 + ccw1

        results.append({

            "Bee": bee,

            "Dish": "dish1",

            "CW": cw1,

            "CCW": ccw1,

            # Percentage clockwise
            "CW_percent":
                cw1 / total1 * 100
                if total1 > 0 else 0,

            # Percentage counterclockwise
            "CCW_percent":
                ccw1 / total1 * 100
                if total1 > 0 else 0
        })

        # =========================
        # Analyze rotation around
        # dish 2
        # =========================
        cw2, ccw2 = compute_rotation_vec(
            x_vals,
            y_vals,
            *center2
        )

        total2 = cw2 + ccw2

        results.append({

            "Bee": bee,

            "Dish": "dish2",

            "CW": cw2,

            "CCW": ccw2,

            "CW_percent":
                cw2 / total2 * 100
                if total2 > 0 else 0,

            "CCW_percent":
                ccw2 / total2 * 100
                if total2 > 0 else 0
        })

    return results

# ================================
# Collect all rotation data
# ================================
data = []

groups = ["CN","GR"]

# Process each experimental group
for group_prefix in groups:

    # Find matching CSV files
    files = [
        f for f in os.listdir(csv_folder)
        if f.endswith(".csv")
        and f.startswith(group_prefix)
    ]

    print(f"\nProcessing {len(files)} videos for group {group_prefix}...")

    # Loop through files
    for f in tqdm(files, desc=f"{group_prefix} videos"):

        # Analyze video
        res = analyze_video_both_dishes(
            os.path.join(csv_folder,f)
        )

        # Store results
        if res:

            for r in res:

                # Clean file name
                full_name = f.replace(".csv","")

                if "DLC" in full_name:
                    full_name = full_name.split("DLC")[0]

                full_name = full_name.rstrip("_")

                # Add metadata
                r["Group"] = group_prefix
                r["Dyad"] = full_name

                # Unique bee identifier
                r["Subject"] = f"{full_name}_{r['Bee']}"

                data.append(r)

# ================================
# Save results to Excel
# ================================
df_out = pd.DataFrame(data)

df_out.to_excel(output_excel, index=False)

print("Saved rotation data with both dishes!")

# ================================
# Convert to long format
# for statistics and plotting
# ================================
df_long = df_out.melt(

    id_vars=[
        "Group",
        "Dyad",
        "Bee",
        "Dish",
        "Subject"
    ],

    value_vars=[
        "CW_percent",
        "CCW_percent"
    ],

    var_name="Direction",

    value_name="Percentage"
)

# Rename directions
df_long["Direction"] = df_long["Direction"].map({

    "CW_percent":"Clockwise",

    "CCW_percent":"Counterclockwise"

})

# ================================
# Mixed-effects statistical model
# ================================
md = smf.mixedlm(

    "Percentage ~ Group * Dish * Direction",

    df_long,

    groups=df_long["Subject"]

)

# Fit model
mdf = md.fit()

# Print summary
print("\nMIXED EFFECTS MODEL SUMMARY")

print(mdf.summary())

# ================================
# Plot CW vs CCW rotation
# ================================
palette = {

    "CN":"#1f77b4",

    "GR":"#d62728"

}

# Create grouped bar plots
g = sns.catplot(

    data=df_long,

    x="Direction",

    y="Percentage",

    hue="Group",

    col="Dish",

    kind="bar",

    errorbar="sd",

    palette=palette,

    height=5,

    aspect=1
)

# Figure title
g.fig.suptitle(

    "Rotation Preference by Dish (CW vs CCW)",

    fontsize=16,

    weight='bold',

    y=1.05
)

# Axis formatting
for ax in g.axes.flat:

    ax.set_ylim(0,100)

    ax.set_ylabel("Percentage (%)")

    ax.set_xlabel("Direction")

plt.tight_layout()

plt.show()

# ================================
# Compute rotation bias
# ================================
df_out["Bias"] = (

    (df_out["CCW"] - df_out["CW"])

    /

    (df_out["CCW"] + df_out["CW"])

)

# ================================
# Plot rotation bias
# ================================
plt.figure(figsize=(6,5))

sns.barplot(

    data=df_out,

    x="Group",

    y="Bias",

    errorbar="sd",

    palette=palette
)

# Reference line at zero
plt.axhline(

    0,

    linestyle="--",

    color="black"
)

# Labels
plt.title(

    "Rotation Bias (CCW - CW)",

    fontsize=14,

    weight='bold'
)

plt.ylabel("Bias Index")

plt.xlabel("Group")

sns.despine()

plt.tight_layout()

plt.show()


# %%
