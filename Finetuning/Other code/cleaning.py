import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher
import string

def main():
    data = pd.read_csv('../medical_3/Deep Cleaning/version_2.csv')
    
    # Lowercase the 'clean_question' column
    data['clean_question'] = data['clean_question'].str.lower()
    
    # Remove all spaces from the 'clean_question' column
    data['clean_question'] = data['clean_question'].str.replace(" ", "")
    
    # Remove punctuation from the 'clean_question' column
    data['clean_question'] = data['clean_question'].str.replace(f"[{string.punctuation}]", "", regex=True)
    
    # Count and print the number of duplicate rows based on the 'clean_question' column
    num_duplicates = data.duplicated(subset=['clean_question']).sum()
    print(f"Number of duplicate rows: {num_duplicates}")
    
    # Remove duplicated rows based on the 'clean_question' column
    data = data.drop_duplicates(subset=['clean_question'])
    
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
                "clean_question": i.clean_question
            }
        )
        
    df = pd.DataFrame(rows)
    
    # Save cleaned data to csv
    df.to_csv('../medical_3/Deep Cleaning/final.csv', index=False)

if __name__ == '__main__':
    main()