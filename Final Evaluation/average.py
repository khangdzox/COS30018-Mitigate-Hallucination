import pandas as pd

# Load the CSV file
df = pd.read_csv('Halu_Bench/LoRA/L10_last_model_2.csv')

# Calculate the averages
avg_meteor_score = df['meteor_score'].mean()
avg_bleu_score = df['bleu_score'].mean()
avg_rouge_1 = df['rouge_1'].mean()
avg_rouge_l = df['rouge_l'].mean()
avg_Accuracy_pred = df['Accuracy_pred'].mean()
avg_Accuracy_ref = df['Accuracy_ref'].mean()

# Create a new DataFrame with the averages
averages = pd.DataFrame({
    'avg_Accuracy_pred': [avg_Accuracy_pred],
    'avg_Accuracy_ref': [avg_Accuracy_ref],
    'avg_bleu_score': [avg_bleu_score],
    'avg_meteor_score': [avg_meteor_score],
    'avg_rouge_1': [avg_rouge_1],
    'avg_rouge_l': [avg_rouge_l],
})

# Append the new DataFrame to averages.csv
averages.to_csv('averages.csv', mode='a', index=False)