import os
import pandas as pd
import evaluate

# Define the directory containing the CSV files
directory = "Final Evaluation/Halueval"

# Initialize an empty list to hold the results
results = []

# Loop through the files in the directory
for root, _, files in os.walk(directory):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")
            try:
                df = pd.read_csv(file_path)
                
                # Skip the last 10 rows
                df = df.iloc[:-10]
                
                # Combine the required metrics
                metrics = evaluate.combine(['accuracy', 'f1', 'precision', 'recall'])
                scores = metrics.compute(predictions=df['base_model'], references=df['targets'])
                
                # Store the scores in the results list
                for metric, score in scores.items():
                    results.append({
                        'File Name': file_path,
                        'Metric': metric,
                        'Average Value': score
                    })
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Pivot the DataFrame to have files as rows and metrics as columns
pivot_results = results_df.pivot(index='File Name', columns='Metric', values='Average Value').reset_index()

# Rename columns for better readability
pivot_results.columns.name = None

# Define the output file path
output_file = os.path.join(directory, 'averages.csv')

# Write the results to a CSV file
print(f"Writing results to {output_file}")
pivot_results.to_csv(output_file, index=False)
print("Done!")