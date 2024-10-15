import os
import pandas as pd

# Define the directories
directories = ['Final Evaluation/Truthful_QA/Base_model', 'Final Evaluation/Truthful_QA/QLoRA', 'Final Evaluation/Truthful_QA/LoRA']

# Initialize an empty dataframe
all_data = pd.DataFrame()

# Loop through all directories
for directory in directories:
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the filename contains 'result' and is a CSV file
        if 'result' in filename and filename.endswith('.csv'):
            # Construct the full file path
            file_path = os.path.join(directory, filename)
            # Read the file into a dataframe
            df = pd.read_csv(file_path)
            # Append the data to the all_data dataframe
            all_data = all_data.append(df, ignore_index=True)

# Write the combined data to a new CSV file
all_data.to_csv('combined_results.csv', index=False)