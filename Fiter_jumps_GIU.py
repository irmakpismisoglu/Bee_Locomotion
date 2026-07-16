README
"""
Petri Dish Border Filtering Script
==================================

This script filters DeepLabCut tracking data so that only bee abdomen
coordinates located inside the Petri dishes are kept.

The script does three main things:

1. Finds each DeepLabCut CSV file.
2. Matches it with the corresponding video file.
3. Keeps only the tracked bee abdomen points that fall inside the manually
   marked Petri dish borders.

Points outside the Petri dishes are replaced with NaN.



Install the required packages before running the script


Folder structure
----------------

Before running the script, users must set the main folder path here:

    main_folder = r""

For example:

    main_folder = r"C:\Users\YourName\Documents\bee_project"

Inside this main folder, the script expects these folders:

    video_folder
    rawdata_folder
    Petri_Border_dimensions
    filtered_csv

Example folder structure:

    bee_project/
    ├── video_folder/
    ├── rawdata_folder/
    ├── Petri_Border_dimensions/
    └── filtered_csv/

The video files should be placed inside:

    video_folder

The raw DeepLabCut CSV files should be placed inside:

    rawdata_folder

The script will automatically create these output folders if they do not exist:

    Petri_Border_dimensions
    filtered_csv

Input files
-----------

The script expects DeepLabCut CSV files in the rawdata_folder.

The script also expects each video to have the same base name as the
corresponding CSV file before the word "DLC".

Example:

    CSV file:
    CN01DLC_resnet50_project.csv

    Matching video:
    CN01.mp4

The script extracts "CN01" from the CSV filename and looks for:

    CN01.mp4

If the matching video is missing, that file is skipped.

Manual Petri dish marking
-------------------------

For each video, the script loads the first frame and asks the user to mark
two Petri dish borders.

An OpenCV window opens for each Petri dish.

Use the sliders to adjust:

    Center X
    Center Y
    Radius

When the circle fits the Petri dish, press:

    q

This confirms the selected circle.

The script saves the Petri dish border information as a CSV file in:

    Petri_Border_dimensions

The saved file is named:

    video_name_petri_circles.csv

Example:

    CN01_petri_circles.csv

If this file already exists, the script loads it automatically and does not
ask the user to mark the Petri dishes again.

Petri output file
-----------------

Each saved Petri file contains:

    Center_X
    Center_Y
    Radius

for:

    Petri_Dish_1
    Petri_Dish_2

Filtering tracking data
-----------------------

The script reads abdomen coordinates for two individuals:

    B1
    B2

It expects the DeepLabCut body part name to be:

    ABD

For each bee, the script extracts:

    x coordinate
    y coordinate

Then it checks whether each point is inside either Petri dish.

If the point is inside a Petri dish, it is kept.

If the point is outside both Petri dishes, it is replaced with NaN.

Important names users may need to change
----------------------------------------

If your individuals are not named B1 and B2, change this line:

    for individual in ['B1','B2']:

For example:

    for individual in ['Bee1','Bee2']:

If your body part is not named ABD, change these lines:

    x = dlc_data[scorer_name][individual]['ABD']['x']
    y = dlc_data[scorer_name][individual]['ABD']['y']

For example, if your body part is called abdomen:

    x = dlc_data[scorer_name][individual]['abdomen']['x']
    y = dlc_data[scorer_name][individual]['abdomen']['y']

Output files
------------

Filtered coordinate files are saved in:

    filtered_csv

Each output file is named:

    video_name_ABD_filtered_border.csv

Example:

    CN01_ABD_filtered_border.csv

The output CSV contains:

    B1_x
    B1_y
    B2_x
    B2_y

Only coordinates inside the Petri dishes are kept.
Coordinates outside the Petri dishes are saved as NaN.

Summary
-------

This script is useful for cleaning DeepLabCut tracking data before making
heatmaps or other position-based analyses. It removes tracking points outside
the experimental arena and keeps only positions inside the Petri dishes.
"""


# Import required libraries
import cv2                  # Video processing and GUI tools
import pandas as pd         # Data handling
import numpy as np          # Numerical operations
import os                   # File and folder management
from glob import glob       # Finding files in folders

# -----------------------------
# Define folder paths
main_folder = r""

video_folder = os.path.join(main_folder, "video_folder")
dlc_folder = os.path.join(main_folder, "rawdata_folder")
petri_dim_folder = os.path.join(main_folder, "Petri_Border_dimensions")
filtered_csv_folder = os.path.join(main_folder, "filtered_csv")

# Create output folders if they do not exist
os.makedirs(petri_dim_folder, exist_ok=True)
os.makedirs(filtered_csv_folder, exist_ok=True)

# -----------------------------
# Function to manually mark Petri dish borders
def mark_circle_trackbar(frame, dish_idx):

    # Initial circle position and radius
    h, w = frame.shape[:2]
    cx, cy, r = w//2, h//2, min(h,w)//4

    # Empty callback for trackbars
    def nothing(x): 
        pass

    # Create interactive OpenCV window
    cv2.namedWindow(f"Circle Petri Dish {dish_idx+1}")

    # Create sliders for circle adjustment
    cv2.createTrackbar("Center X", f"Circle Petri Dish {dish_idx+1}", cx, w, nothing)
    cv2.createTrackbar("Center Y", f"Circle Petri Dish {dish_idx+1}", cy, h, nothing)
    cv2.createTrackbar("Radius", f"Circle Petri Dish {dish_idx+1}", r, min(h,w)//2, nothing)

    # Update circle interactively
    while True:

        # Read slider positions
        cx = cv2.getTrackbarPos("Center X", f"Circle Petri Dish {dish_idx+1}")
        cy = cv2.getTrackbarPos("Center Y", f"Circle Petri Dish {dish_idx+1}")
        r = cv2.getTrackbarPos("Radius", f"Circle Petri Dish {dish_idx+1}")

        # Draw preview circle
        disp = frame.copy()
        cv2.circle(disp, (cx, cy), r, (0,0,255), 2)

        # Show preview
        cv2.imshow(f"Circle Petri Dish {dish_idx+1}", disp)

        # Press "q" to confirm
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break

    # Close window
    cv2.destroyWindow(f"Circle Petri Dish {dish_idx+1}")

    # Return selected circle parameters
    return (cx, cy, r)

# -----------------------------
# Function to keep coordinates only inside Petri dishes
def cut_strictly_inside(x, y, circles):

    # Combine x and y coordinates
    coords = np.vstack((x, y)).T

    # Create empty boolean mask
    inside_any = np.zeros(len(coords), dtype=bool)

    # Check whether points are inside any circle
    for (cx, cy, r) in circles:

        # Calculate distance from circle center
        dist = np.sqrt((coords[:,0]-cx)**2 + (coords[:,1]-cy)**2)

        # Mark points inside the circle
        inside_any |= dist <= r

    # Keep inside points, replace outside with NaN
    x_cut = np.where(inside_any, x, np.nan)
    y_cut = np.where(inside_any, y, np.nan)

    return x_cut, y_cut

# -----------------------------
# Find all DLC CSV files
dlc_files = glob(os.path.join(dlc_folder, "*.csv"))

# Check whether files exist
if not dlc_files:
    print("No files found")

else:

    # Process each DLC file
    for dlc_file in dlc_files:

        # Extract file name
        full_csv_name = os.path.basename(dlc_file)

        # Extract dyad name before "DLC"
        dyad_name = full_csv_name.split("DLC")[0]

        # Match corresponding video file
        video_path = os.path.join(video_folder, f"{dyad_name}.mp4")

        # Skip if video is missing
        if not os.path.exists(video_path):
            print(f"Video not found for {dyad_name}, skipping.")
            continue

        print(f"Processing {dyad_name}...")

        # Read DeepLabCut CSV with multi-level header
        dlc_data = pd.read_csv(dlc_file, header=[0,1,2,3], index_col=0)

        # Get scorer name
        scorer_name = dlc_data.columns.levels[0][0]

        # -----------------------------
        # Load first video frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        # Skip if frame cannot be read
        if not ret:
            print(f"Cannot read video frame for {dyad_name}, skipping.")
            continue

        # -----------------------------
        # Load saved Petri dish circles if available
        petri_file = os.path.join(
            petri_dim_folder,
            f"{dyad_name}_petri_circles.csv"
        )

        if os.path.exists(petri_file):

            # Read saved circle dimensions
            df_circles = pd.read_csv(petri_file, index_col=0)

            # Store circles as tuples
            dish_circles = [
                (
                    int(df_circles.loc["Petri_Dish_1","Center_X"]),
                    int(df_circles.loc["Petri_Dish_1","Center_Y"]),
                    int(df_circles.loc["Petri_Dish_1","Radius"])
                ),
                (
                    int(df_circles.loc["Petri_Dish_2","Center_X"]),
                    int(df_circles.loc["Petri_Dish_2","Center_Y"]),
                    int(df_circles.loc["Petri_Dish_2","Radius"])
                )
            ]

        else:

            # Manually mark Petri dishes
            dish_circles = [
                mark_circle_trackbar(frame, i)
                for i in range(2)
            ]

            # Save circle dimensions
            df_circles = pd.DataFrame(
                dish_circles,
                columns=['Center_X','Center_Y','Radius']
            )

            df_circles.index = ['Petri_Dish_1','Petri_Dish_2']

            df_circles.to_csv(petri_file)

            print(f"Petri circles saved to {petri_file}")

        # -----------------------------
        # Filter abdomen coordinates
        filtered_coords = {}

        for individual in ['B1','B2']:

            # Extract ABD coordinates
            x = dlc_data[scorer_name][individual]['ABD']['x']
            y = dlc_data[scorer_name][individual]['ABD']['y']

            # Keep only points inside dishes
            x_cut, y_cut = cut_strictly_inside(x, y, dish_circles)

            # Store filtered coordinates
            filtered_coords[individual] = (x_cut, y_cut)

        # -----------------------------
        # Save filtered coordinates
        save_csv_path = os.path.join(
            filtered_csv_folder,
            f"{dyad_name}_ABD_filtered_border.csv"
        )

        # Create output dataframe
        df_save = pd.DataFrame()

        # Add filtered coordinates
        for individual, (x, y) in filtered_coords.items():
            df_save[f"{individual}_x"] = x
            df_save[f"{individual}_y"] = y

        # Save CSV
        df_save.to_csv(save_csv_path, index=False)

        print(f"Filtered data saved: {save_csv_path}")

# -----------------------------
print("done")
