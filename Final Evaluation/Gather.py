import pandas as pd
import os

# Define the base path and the paths to the CSV files
base_path = "Final Evaluation"
csv_files = {
    "Med-Halt": os.path.join(base_path, "Med-Halt", "averages.csv"),
    "Halu_Bench": os.path.join(base_path, "Halu_Bench", "averages.csv"),
    "Truthful_QA": os.path.join(base_path, "Truthful_QA", "averages.csv")
}

# Initialize an empty list to hold the DataFrames
dataframes = []

# Loop through the CSV files and read them
for path in csv_files.values():
    df = pd.read_csv(path)
    dataframes.append(df)

# Concatenate all DataFrames into one
merged_df = pd.concat(dataframes, ignore_index=True)

# Save the merged DataFrame to a new CSV file
output_path = os.path.join(base_path, 'merged_averages.csv')
merged_df.to_csv(output_path, index=False)

print(f"Merged CSV saved as '{output_path}'")