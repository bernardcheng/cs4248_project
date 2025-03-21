import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load data here
# Replace with name of dataset
df = pd.read_csv('word_bias_dataset.csv')  # e.g., {'word': 'nurse', 'label': 1}

# 2. Preprocessing steps
X = df['word']
y = df['label']

# 3. Convert words to features (TF-IDF Vectorizer)
# Using character n-grams here 
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4)) 
# vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# 5. Train the model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# 6. Evaluate the model
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# 7. Make predictions
def predict_bias(word):
    word_vec = vectorizer.transform([word])
    prediction = classifier.predict(word_vec)
    return "Gender Biased" if prediction[0] == 1 else "Gender Neutral"

# Example
print(predict_bias("nurse"))
print(predict_bias("engineer"))
