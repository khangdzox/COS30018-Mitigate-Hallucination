import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher
import string
import re

def main():
    data = pd.read_csv('../medical_3/Deep Cleaning/final_2.csv')
    
    # Remove all occurrences of "Ans. " in the 'exp' column
    data['exp'] = data['exp'].str.replace('Ans. ', '', regex=False)
    
    rows = []
    for i in data.itertuples(index=False):
        rows.append(
            {
                "question": i.question,
                "option_a": i.option_a,
                "option_b": i.option_b,
                "option_c": i.option_c,
                "option_d": i.option_d,
                "cop": i.cop,
                "exp": i.exp,
            }
        )
        
    df = pd.DataFrame(rows)
    
    # Save cleaned data to csv
    df.to_csv('../medical_3/Deep Cleaning/final_2_cleaned.csv', index=False)

if __name__ == '__main__':
    main()