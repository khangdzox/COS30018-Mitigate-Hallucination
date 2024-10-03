import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def main():
    data = pd.read_csv('../medical_3/Deep Cleaning/final.csv')
    
    rows = []
    for i in data.iloc:
        rows.append(
            {
                "question": i["question"],
                "option_a": i['option_a'],
                "option_b": i['option_b'],
                "option_c": i['option_c'],
                "option_d": i['option_d'],
                "cop": i['cop'],
                "exp": i["exp"],
            }
        )
        
    df = pd.DataFrame(rows)
    
    # Replace specified characters with space in all columns
    # for column in df.columns:
        # df[column] = df[column].str.replace(r'<\\p>|<p>|&;|&;s|<img alt="" src="/>', '', regex=True)

    # Count rows containing specified characters after replacement
    # count = df.apply(lambda x: x.str.contains(r'<\\p>|<p>|&;|<img alt="" src="/>').any(), axis=1).sum()
    # print(f"Rows containing specified characters after replacement: {count}")
    
    # Check if the question is exactly same as answer then drop that raw
    # df = df[~df.apply(lambda x: x['exp'] in [x['option_a'], x['option_b'], x['option_c'], x['option_d']], axis=1)]

    def replace_cop(row):
        if row['cop'] == 'A':
            return row['option_a']
        elif row['cop'] == 'B':
            return row['option_b']
        elif row['cop'] == 'C':
            return row['option_c']
        elif row['cop'] == 'D':
            return row['option_d']
        return row['cop']
    
    # Apply the function to the cop column
    df['cop'] = df.apply(replace_cop, axis=1)

    # Verify replacement
    print(df)

    # Save cleaned data to csv
    df.to_csv('../medical_3/Deep Cleaning/final.csv', index=False)

if __name__ == '__main__':
    main()

