# NLP-practical-2
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    'text': [
        'I love this product, it is amazing!',
        'This is the worst experience ever.',
        'Absolutely fantastic service!',
        'I am very disappointed with the quality.',
        'Great value for money.',
        'Terrible, will not buy again.'
    ],
    'label': [
        'positive', 'negative',
        'positive', 'negative',
        'positive', 'negative'
    ]
}

df = pd.DataFrame(data)

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict new sentence
new_text = ["The product is excellent and I love it."]
new_vector = vectorizer.transform(new_text)
prediction = model.predict(new_vector)

print("\nPrediction:", prediction[0]) 