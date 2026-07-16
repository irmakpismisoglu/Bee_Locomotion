# Bee Position Heatmap Script

This script creates normalized heatmaps from bee tracking CSV files.

It reads tracked x/y positions, matches each video to Petri dish border information, 
normalizes all positions by Petri dish radius, and saves both individual and group heatmaps.

---

## Required Python Packages

Install the required packages before running the script:

```bash
pip install numpy pandas matplotlib

<span style="color:red">Important text</span>

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap


# PATHS

# Change this to the folder containing your tracking CSV files.
csv_folder = r"I:\sailor_moon\DATA\filtered_csv"

# Change this to the folder containing your Petri dish border CSV files.
petri_folder = r"I:\sailor_moon\DATA\Petri_Border_dimensions"

# The script will create a Heatmaps folder inside csv_folder.
output_folder = os.path.join(csv_folder, "Heatmaps")

# The script will save one heatmap per video inside this folder.
individual_folder = os.path.join(output_folder, "Individual")

# Create output folders if they do not already exist.
os.makedirs(output_folder, exist_ok=True)
os.makedirs(individual_folder, exist_ok=True)


# PARAMETERS

# Size of the output heatmap image.
IMG_SIZE = 800

# Normalized radius of the Petri dish in the output image.
NORM_RADIUS = 150


# Get Petri filename

def get_base_for_petri(filename):

    # Keep only the file name, not the full folder path.
    base = os.path.basename(filename)

    # Remove the .csv extension.
    if base.endswith(".csv"):
        base = base[:-4]

    # If the filename contains DeepLabCut text, remove everything after DLC.
    if "DLC" in base:
        base = base.split("DLC")[0]

    # Remove extra underscores from the end of the filename.
    return base.rstrip("_")


# Load Petri information

def load_petri(base):

    # Build the expected Petri dish filename.
    file = os.path.join(
        petri_folder,
        base + "_petri_circles.csv"
    )

    # If the Petri file does not exist, skip this video.
    if not os.path.exists(file):
        print("Missing:", file)
        return None, None, None

    # Read the Petri dish CSV file.
    df = pd.read_csv(file)

    # Read the center of Petri dish 1 from the first row.
    center1 = (
        df.iloc[0]["Center_X"],
        df.iloc[0]["Center_Y"]
    )

    # Read the center of Petri dish 2 from the second row.
    center2 = (
        df.iloc[1]["Center_X"],
        df.iloc[1]["Center_Y"]
    )

    # Use the average radius from the Petri file.
    radius = np.mean(df["Radius"])

    return center1, center2, radius


# Create one video's normalized heatmap

def process_video(file_path):

    # Create an empty heatmap.
    heatmap = np.zeros((IMG_SIZE, IMG_SIZE))

    # Center coordinate of the output image.
    center_img = IMG_SIZE // 2

    # Read the tracking CSV file.
    df = pd.read_csv(file_path)

    # Remove accidental spaces from column names.
    df.columns = df.columns.str.strip()

    # Get the base filename used to find the matching Petri file.
    base = get_base_for_petri(file_path)

    # Load Petri dish center and radius information.
    center1, center2, radius = load_petri(base)

    # Skip this file if the Petri information is missing.
    if center1 is None:
        return None

    # Find x and y coordinate columns for bee 1.
    x_b1 = [c for c in df.columns if "B1_ABD" in c and c.endswith("_x")]
    y_b1 = [c for c in df.columns if "B1_ABD" in c and c.endswith("_y")]

    # Find x and y coordinate columns for bee 2.
    x_b2 = [c for c in df.columns if "B2_ABD" in c and c.endswith("_x")]
    y_b2 = [c for c in df.columns if "B2_ABD" in c and c.endswith("_y")]

    # Skip this file if the required columns are missing.
    if not (x_b1 and y_b1 and x_b2 and y_b2):
        print("Missing columns:", file_path)
        return None

    # Store both bees' coordinate data.
    bees = [
        (df[x_b1[0]].values, df[y_b1[0]].values),
        (df[x_b2[0]].values, df[y_b2[0]].values)
    ]

    # Process each bee.
    for xvals, yvals in bees:

        # Keep only rows where both x and y are valid numbers.
        mask = np.isfinite(xvals) & np.isfinite(yvals)

        xvals = xvals[mask]
        yvals = yvals[mask]

        # Process every tracked point.
        for x, y in zip(xvals, yvals):

            # Distance from this point to Petri dish 1.
            d1 = np.sqrt((x-center1[0])**2 + (y-center1[1])**2)

            # Distance from this point to Petri dish 2.
            d2 = np.sqrt((x-center2[0])**2 + (y-center2[1])**2)

            # Decide which Petri dish contains this point.
            if d1 <= radius:
                cx, cy = center1

            elif d2 <= radius:
                cx, cy = center2

            else:
                # Ignore points outside both Petri dishes.
                continue

            # Normalize the position relative to Petri center and radius.
            xn = (x-cx)/radius
            yn = (y-cy)/radius

            # Convert normalized coordinates to heatmap pixel coordinates.
            xp = int(center_img + xn*NORM_RADIUS)
            yp = int(center_img + yn*NORM_RADIUS)

            # Add one count to this heatmap pixel.
            if 0 <= xp < IMG_SIZE and 0 <= yp < IMG_SIZE:
                heatmap[yp, xp] += 1

    return heatmap


# Plot heatmap

def save_heatmap(heatmap, title, outfile):

    # Create a white figure background.
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
    ax.set_facecolor("white")

    # Keep empty regions white instead of coloring zero values.
    plotted_heatmap = np.ma.masked_where(heatmap == 0, heatmap)

    # Use the 99th percentile as the color maximum.
    # This prevents a few very dense pixels from making the rest too faint.
    nonzero = heatmap[heatmap > 0]
    vmax = np.percentile(nonzero, 99) if nonzero.size else 1

    # Create a publication-style density color map.
    # Low density = pale blue, medium density = yellow/orange, high density = red.
    density_cmap = LinearSegmentedColormap.from_list(
        "academic_density",
        [
            "#d7efff",  # very low density: pale blue
            "#67a9cf",  # low density: blue
            "#fff3b0",  # medium density: pale yellow
            "#fdae61",  # high density: orange
            "#d7191c"   # highest density: red
        ]
    )

    # Make masked zero-count areas white.
    density_cmap.set_bad(color="white")

    # Draw the heatmap.
    im = ax.imshow(
        plotted_heatmap,
        origin="lower",
        cmap=density_cmap,
        interpolation="gaussian",
        norm=colors.PowerNorm(gamma=0.6, vmin=1, vmax=vmax)
    )

    # Draw the normalized Petri dish border.
    circle = plt.Circle(
        (IMG_SIZE//2, IMG_SIZE//2),
        NORM_RADIUS,
        edgecolor="black",
        fill=False,
        linewidth=2
    )

    ax.add_patch(circle)

    # Add title and remove axes.
    ax.set_title(title, fontsize=12)
    ax.axis("off")

    # Add a colorbar showing position density.
    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04
    )
    cbar.set_label("Position density", rotation=270, labelpad=15)

    # Adjust spacing.
    fig.tight_layout()

    # Save the heatmap.
    fig.savefig(
        outfile,
        dpi=300,
        facecolor="white",
        bbox_inches="tight"
    )

    # Close the figure to save memory.
    plt.close(fig)


# GROUP HEATMAPS

# Empty group heatmap for CN files.
CN_group = np.zeros((IMG_SIZE, IMG_SIZE))

# Empty group heatmap for GR files.
GR_group = np.zeros((IMG_SIZE, IMG_SIZE))

# Find all CSV files in the tracking folder.
files = sorted(
    [f for f in os.listdir(csv_folder) if f.endswith(".csv")]
)

# Process every CSV file.
for file in files:

    print(file)

    # Create heatmap for this video.
    heatmap = process_video(
        os.path.join(csv_folder, file)
    )

    # Skip files that could not be processed.
    if heatmap is None:
        continue

    # Save individual heatmap.
    save_heatmap(
        heatmap,
        file,
        os.path.join(
            individual_folder,
            file.replace(".csv", ".png")
        )
    )

    # Add this heatmap to the CN group if the filename starts with CN.
    if file.startswith("CN"):
        CN_group += heatmap

    # Add this heatmap to the GR group if the filename starts with GR.
    elif file.startswith("GR"):
        GR_group += heatmap


# Save group heatmaps

save_heatmap(
    CN_group,
    "CN Group Heatmap",
    os.path.join(
        output_folder,
        "CN_Group_Heatmap.png"
    )
)

save_heatmap(
    GR_group,
    "GR Group Heatmap",
    os.path.join(
        output_folder,
        "GR_Group_Heatmap.png"
    )
)

print("\nFinished!")
print(output_folder)
