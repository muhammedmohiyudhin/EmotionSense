import pandas as pd
import re

input_file = "../data/emotions.csv"
output_file = "../data/cleaned_emotions.csv"

df = pd.read_csv(input_file)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

df.to_csv(output_file, index=False)

print("Data preprocessing completed. cleaned_emotions.csv created.")
