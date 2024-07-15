import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
# Download NLTK resources

# Load the Sentiment140 dataset
# Replace 'path_to_sentiment140.csv' with the actual path to your downloaded CSV file
df = pd.read_csv('path_to_sentiment140.csv', encoding='ISO-8859-1', header=None, names=["target", "id", "date", "flag", "user", "text"])

# Keep only the 'target' (sentiment) and 'text' columns
df = df[['target', 'text']]

# Map target labels to 0 and 1 (0 for negative, 4 for positive)
df['target'] = df['target'].map({0: 0, 4: 1})

# Shuffle the dataset
df = shuffle(df)

# Sample a subset of the dataset for faster execution (adjust as needed)
df = df.sample(10000)

# Preprocess the text data
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

df['text'] = df['text'].apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)
# # Save the trained SVM model
# joblib.dump(svm_classifier, 'svm_model.pkl')
#
# # Save the TF-IDF vectorizer
# joblib.dump(vectorizer, 'vectorizer.pkl')
