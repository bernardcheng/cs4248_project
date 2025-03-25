import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load data from WinoBias/wino/data directory
def load_sentences_from_file(filepath, label):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip().replace('[', '').replace(']', '')  # Remove brackets such as []
        data.append({'sentence': line, 'label': label})
    return data


# Folder path
folder = 'datasets/WinoBias/data'

# File-label mapping: 1 = gender biased, 0 = gender neutral
file_labels = {
    'anti_stereotyped_type1.txt': 0,
    'anti_stereotyped_type2.txt': 0,
    'pro_stereotyped_type1.txt': 1,
    'pro_stereotyped_type2.txt': 1,
}

all_data = []
for fname, label in file_labels.items():
    path = os.path.join(folder, fname)
    all_data.extend(load_sentences_from_file(path, label))

df = pd.DataFrame(all_data)
print(df.head())

# Optionally save dataset
# df.to_csv('word_bias_dataset.csv', index=False)

# 2. Preprocessing steps
model = SentenceTransformer('all-MiniLM-L6-v2')
X = model.encode(df['sentence'], show_progress_bar=True)
y = df['label']

# 3. Convert words to features (TF-IDF Vectorizer)
# vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4))
# vectorizer = TfidfVectorizer()
# X_vec = vectorizer.fit_transform(X)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the model
classifier = LogisticRegression(max_iter = 1000)
classifier.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# 7. Make predictions
# def predict_bias(word):
#     word_vec = vectorizer.transform([word])
#     prediction = classifier.predict(word_vec)
#     return "Gender Biased" if prediction[0] == 1 else "Gender Neutral"

# # Example
# print(predict_bias("nurse"))
# print(predict_bias("engineer"))
# print(predict_bias("teacher"))
# print(predict_bias("programmer"))

