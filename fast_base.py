1   import os
2   import numpy as np
3   import pandas as pd
4
5   
6   # Settings
7   
8   fps = 30  # Frames per second 
9   missing_threshold = 0.2  # Maximum allowed missing proportion per bee 
10  max_jump = 70  # Maximum distance (pixels/frame) to consider as valid, larger jumps are tracking errors.
11
12 
13  # Function to calculate velocity
14 
15  def calculate_velocity(x, y):
16      """
17      Compute frame-by-frame velocity (pixels/frame) of a bee's head.
18      Large jumps due to tracking errors (> max_jump) are removed  and saved as NaN to see proportion of missing data.
19      """
20      dx = np.diff(x)  # Frame-to-frame x displacement
21      dy = np.diff(y)  # Frame-to-frame y displacement
22      dist = np.sqrt(dx**2 + dy**2)  # Euclidean distance per frame
23      dist[dist > max_jump] = np.nan  # Remove unrealistic jumps
24      return dist  # Returns velocity in pixels/frame
25
26 
27  # Function to process all CSV files in a folder
28  
29  def process_folder(folder_path, group_name):
30      """
31      Processes CSV files for a group:
32      - Skips files if either bee has > missing_threshold missing data
33      - Computes combined velocity for each frame
34      - Assigns 'fast' if velocity > mean + 2*SD, else 'baseline'
35      """
36      all_results = []  # Store results for all files
37
38      for filename in os.listdir(folder_path):
39          if not filename.endswith(".csv"):  # Only CSV files
40              continue
41
42          file_path = os.path.join(folder_path, filename)
43          df = pd.read_csv(file_path, header=[0,1,2,3])  # DLC output with multi-level headers
44
45          
46          # Extract coordinates for both bees
47        
48          try:
49              x_b1 = df[('DLC_dlcrnetms5_New_DraftJul3shuffle1_200000', 'b1', 'head', 'x')].values
50              y_b1 = df[('DLC_dlcrnetms5_New_DraftJul3shuffle1_200000', 'b1', 'head', 'y')].values
51              x_b2 = df[('DLC_dlcrnetms5_New_DraftJul3shuffle1_200000', 'b2', 'head', 'x')].values
52              y_b2 = df[('DLC_dlcrnetms5_New_DraftJul3shuffle1_200000', 'b2', 'head', 'y')].values
53          except KeyError:
54              print(f"Skipping {filename} due to missing columns.")
55              continue
56
57         
58          # Calculate missing data percentages
59         
60          missing_b1 = (np.isnan(x_b1).sum() + np.isnan(y_b1).sum()) / (2 * len(x_b1))
61          missing_b2 = (np.isnan(x_b2).sum() + np.isnan(y_b2).sum()) / (2 * len(x_b2))
62
63          # Skip file if either bee has > 20% missing
64          if missing_b1 > missing_threshold or missing_b2 > missing_threshold:
65              print(f"Skipping {filename} due to >20% missing data in one of the bees.")
66              continue
67
68          
69          # Calculate velocities
70         
71          vel_b1 = calculate_velocity(x_b1, y_b1)
72          vel_b2 = calculate_velocity(x_b2, y_b2)
73
74          # Combined velocity (average of two bees) per frame
75          combined_vel = np.nanmean([vel_b1, vel_b2], axis=0)
76
77          # Remove NaNs for statistics
78          valid_vel = combined_vel[~np.isnan(combined_vel)]
79          mean_vel = np.mean(valid_vel)  # Mean velocity
80          sd_vel = np.std(valid_vel)     # Standard deviation
81
82          # Print mean and SD 
83          print(f"{filename} - Mean velocity: {mean_vel:.2f}, SD: {sd_vel:.2f}")
84
85          
86          # Assign movement type
87         
88          # "fast" if above mean + 2*SD, else "baseline"
89          labels = np.where(combined_vel > mean_vel + 2*sd_vel, "fast", "baseline")
90
91          
92          # Save results for this file
93          
94          df_result = pd.DataFrame({
95              "File": filename,
96              "Group": group_name,
97              "Velocity": combined_vel,
98              "Movement_Type": labels
99          })
100         all_results.append(df_result)
101
102     # Combine all files into a single DataFrame
103     if all_results:
104         return pd.concat(all_results, ignore_index=True)
105     else:
106         return pd.DataFrame(columns=["File", "Group", "Velocity", "Movement_Type"])
107
108 
109 # Main analysis
110 
111 def main():
112     gr_folder =#!replace with your file! file that contains GR coordination cvs
113     cn_folder = #same for CN
114   #to see if all well show "processing"  
115     print("Processing GR group...")
116     gr_results = process_folder(gr_folder, group_name="GR")
117     print("Processing CN group...")
118     cn_results = process_folder(cn_folder, group_name="CN")
119
120     
121     # Save combined results to Excel
122     
123     save_path = "/home/irmak.pismisoglu/New_Draft-Irmak -2025-07-03/A/bee_velocity_movement_summary.xlsx"
124     combined = pd.concat([gr_results, cn_results], ignore_index=True)
125     combined.to_excel(save_path, index=False)
126     print(f"Results saved to {save_path}")
127
128 if __name__ == "__main__":
129     main()

