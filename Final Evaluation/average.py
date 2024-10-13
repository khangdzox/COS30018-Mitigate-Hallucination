import os
import pandas as pd

def calculate_averages(input_dir, output_file):
    # Create a DataFrame to store the results
    results = pd.DataFrame(columns=['File Name', 'Metric', 'Average Value'])
    print(f"Scanning directory: {input_dir}")

    # Columns to calculate averages for
    columns_of_interest = ['meteor_score', 'bleu_score', 'rouge_1', 'rouge_l', 'Accuracy_pred', 'Accuracy_ref']

    # Walk through the directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv') and file != 'averages.csv':
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                try:
                    df = pd.read_csv(file_path)
                    
                    # Calculate the average for each column of interest
                    for column in columns_of_interest:
                        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                            average = df[column].mean()
                            results = pd.concat([results, pd.DataFrame({'File Name': [file], 'Metric': [column], 'Average Value': [average]})], ignore_index=True)
                        else:
                            print(f"Skipping column: {column} (not found or non-numeric)")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Pivot the DataFrame to have files as rows and metrics as columns
    pivot_results = results.pivot(index='File Name', columns='Metric', values='Average Value').reset_index()

    # Rename columns for better readability
    pivot_results.columns.name = None

    # Write the results to a CSV file
    print(f"Writing results to {output_file}")
    pivot_results.to_csv(output_file, index=False)
    print("Done!")

# Define the input directory and output file
input_dir = './Final Evaluation/'
output_file = './Final Evaluation/averages.csv'

# Calculate the averages and write to the output file
calculate_averages(input_dir, output_file)