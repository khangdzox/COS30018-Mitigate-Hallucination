import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def main():
    data = pd.read_csv('../../../medical_3/Deep Cleaning/Version_1.csv')
    
    rows = []
    for i in data.iloc:
        rows.append(
            {
                "question": i["question"],
                "option_a": i['opa'],
                "option_b": i['opb'],
                "option_c": i['opc'],
                "option_d": i['opd'],
                "cop": i['cop'],
                "exp": i["exp"],
                "subject": i["subject_name"],
            }
        )
        
    df = pd.DataFrame(rows)
    
    # Replace specified characters with space in all columns
    for column in df.columns:
        df[column] = df[column].str.replace(r'<\\p>|<p>|&;|&;s|<img alt="" src="/>', '', regex=True)

    # Count rows containing specified characters after replacement
    # count = df.apply(lambda x: x.str.contains(r'<\\p>|<p>|&;|<img alt="" src="/>').any(), axis=1).sum()
    # print(f"Rows containing specified characters after replacement: {count}")
    
    # Check if the question is exactly same as answer then drop that raw
    df = df[~df.apply(lambda x: x['exp'] in [x['option_a'], x['option_b'], x['option_c'], x['option_d']], axis=1)]

    # Check if the question is 90% similar to other then compare the explanation and keep the longest
    # Create a similarity matrix
    similarity_matrix = [[similar(a, b) for b in df['question']] for a in df['question']]

    # Create a mask for questions that are 90% similar
    mask = [[val > 0.9 for val in row] for row in similarity_matrix]

    # Apply the mask to the DataFrame
    df = df[mask]

    # Group by 'cop' and keep only the row with the longest 'exp' in each group
    df = df.loc[df.groupby('cop')['exp'].idxmax()]

    # Verify replacement
    print(df)

    # Save cleaned data to csv
    df.to_csv('../../../medical_3/Deep Cleaning/final.csv', index=False)

if __name__ == '__main__':
    main()

