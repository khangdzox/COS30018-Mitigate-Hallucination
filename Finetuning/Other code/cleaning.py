import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Test

def main():
    data = pd.read_csv('../medical_3/Deep Cleaning/Version_1.csv')
    
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
        df[column] = df[column].str.replace(r'<\\p>|<p>|&;|<img alt="" src="/>', ' ', regex=True)
    
    # Count rows containing specified characters after replacement
    count = df.apply(lambda x: x.str.contains(r'<\\p>|<p>|&;|<img alt="" src="/>').any(), axis=1).sum()
    print(f"Rows containing specified characters after replacement: {count}")
    
    # Verify replacement
    print(df)
    
if __name__ == '__main__':
    main()

