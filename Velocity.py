#done with ipython, copy paste ready. 


import os                     # interacting with the operating system (windows, linux etc.)
import numpy as np            # Import NumPy for numerical operations such as arrays and mathematical calculations
import pandas as pd           # Import Pandas for reading, processing, and saving data (CSV/Excel files)

def calculate_velocity(x, y): # Function to calculate velocity from x and y coordinate arrays
    dx = np.diff(x)           #  x-coordinates between consecutive frames
    dy = np.diff(y)           #  y-coordinates between consecutive frames
    dist = np.sqrt(dx**2 + dy**2)   # Euclidean distance 
    dist[dist > 70] = np.nan  # Remove unrealistic jumps (greater than 70 pixels) by marking them as missing values (NaN)
    return dist               # Return the velocity values (distance per frame)

def process_folder(folder_path, group_label):  # function to process all CSV files 
    all_records = []          # Initialize a list to store per-bee velocity results for all files
    b1_velocities = []        # Initialize a list to store mean velocities of Bee1 across files
    b2_velocities = []        # Initialize a list to store mean velocities of Bee2 across files

    for filename in os.listdir(folder_path):   # Loop through every file in the given folder
        if not filename.endswith(".csv"):      # Skip files that are not CSV (only process CSV files)
            continue

        file_path = os.path.join(folder_path, filename)   # Construct the full file path

        try:
            df = pd.read_csv(file_path, header=[0, 1, 2, 3])  # Load the CSV file with a 4-level header from DLC file formats
        except Exception as e:   # If loading fails, handle the error
            print(f"Failed to read {filename}: {e}")  
            continue             

        dyad_id = filename.split("DLC")[0].replace(".csv", "")  # Extract the Dyad ID from the filename

        try:
            x_b1 = df[('DLC_dlcrnetms5_New_DraftJul3shuffle1_200000', 'b1', 'head', 'x')].values
            y_b1 = df[('DLC_dlcrnetms5_New_DraftJul3shuffle1_200000', 'b1', 'head', 'y')].values
            x_b2 = df[('DLC_dlcrnetms5_New_DraftJul3shuffle1_200000', 'b2', 'head', 'x')].values
            y_b2 = df[('DLC_dlcrnetms5_New_DraftJul3shuffle1_200000', 'b2', 'head', 'y')].values
        except KeyError:
            print(f"Skipping {filename} due to missing columns.")
            continue

        vel_b1 = calculate_velocity(x_b1, y_b1)
        vel_b2 = calculate_velocity(x_b2, y_b2)

        # to ensure ID of each bee is documented, the data contains the full name of each bee in the format of group+dyad number + petri position (bee 1 or bee 2)

        mean_b1 = np.nanmean(vel_b1)
        mean_b2 = np.nanmean(vel_b2)

        b1_velocities.append(mean_b1)
        b2_velocities.append(mean_b2)

        all_records.append({
            "Dyad ID": f"{dyad_id}",
            "Bee ID": "Bee1",
            "Average Velocity": mean_b1
        })

        all_records.append({
            "Dyad ID": f"{dyad_id}",
            "Bee ID": "Bee2",
            "Average Velocity": mean_b2
        })

    return all_records, b1_velocities, b2_velocities

def save_velocities_to_excel(records, filename):
    df = pd.DataFrame(records)
    df.to_excel(filename, index=False)
    print(f"Excel file saved to: {filename}")

def main():
    # Define folders
    cn_folder = r"#yourfilename"
    gr_folder = r"#yourfilename"

    # Process both groups
    cn_records, cn_b1, cn_b2 = process_folder(cn_folder, group_label="CN")
    gr_records, gr_b1, gr_b2 = process_folder(gr_folder, group_label="GR")

    # Print group-level means
    print(f"Control Bee1 Mean Velocity: {np.nanmean(cn_b1):.2f}")
    print(f"Control Bee2 Mean Velocity: {np.nanmean(cn_b2):.2f}")
    print(f"GR Bee1 Mean Velocity: {np.nanmean(gr_b1):.2f}")
    print(f"GR Bee2 Mean Velocity: {np.nanmean(gr_b2):.2f}")

    # Save to Excel
    output_path = r"individual_group_mean_velocities.xlsx"
    save_velocities_to_excel(cn_records + gr_records, output_path)

if __name__ == "__main__":
    main()
  
