import pandas as pd

input_file = "../data/train.txt"
output_file = "../data/emotions.csv"

texts = []
emotions = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if ";" in line:
            text, emotion = line.strip().rsplit(";", 1)
            texts.append(text)
            emotions.append(emotion)

df = pd.DataFrame({"text": texts, "emotion": emotions})
df.to_csv(output_file, index=False)

print("Done! emotions.csv created.")
