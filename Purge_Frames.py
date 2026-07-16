# %%
import os
import pandas as pd

%Paths

data_folder = r"YOUR_RAW_DATA_DIR"
timesheet_path = r"YOUR_START_END_TIME_EXCEL_DIR"
output_folder = r"YOUR_OUTPUT_FILE_DIR"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

%read excel

timesheet = pd.read_excel(timesheet_path)
print("Timesheet loaded successfully.\n")


# video processing


for _, row in timesheet.iterrows():

    video_id = str(row["ID"])
    start_frame = int(row["start_frame"])
    end_frame = int(row["end_frame"])

    print("========================================")
    print(f"Processing: {video_id}")
    print(f"Frame range: {start_frame} to {end_frame}")

    
    # Find matching CSV
   

    matching_files = [
        f for f in os.listdir(data_folder)
        if f.startswith(video_id) and f.endswith(".csv")
    ]

    if len(matching_files) == 0:
        print("⚠ No matching CSV file found.")
        continue

    original_filename = matching_files[0]
    csv_path = os.path.join(data_folder, original_filename)

    print(f"Using file: {original_filename}")

  
    # Read DLC CSV 
   

    df = pd.read_csv(csv_path, header=[0,1,2,3])

    
    # Slice desired frames
    

    df_corrected = df.iloc[start_frame:end_frame+1].copy()

    print(f"New file will contain {len(df_corrected)} frames.")

    
    # Save new CSV
  

    new_filename = original_filename.replace(".csv", "_framecorrected.csv")
    save_path = os.path.join(output_folder, new_filename)

    df_corrected.to_csv(save_path, index=False)

    print(f"Saved to: {save_path}\n")

print("✅ All files exported successfully.")

