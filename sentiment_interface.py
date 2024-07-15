import tkinter as tk
from tkinter import messagebox
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained SVM model and vectorizer
svm_classifier = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)


def predict_sentiment():
    user_input = entry.get()
    if user_input:
        preprocessed_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([preprocessed_text])
        prediction = svm_classifier.predict(vectorized_text)[0]

        if prediction == 1:
            result = "Positive"
        else:
            result = "Negative"

        messagebox.showinfo("Sentiment Analysis Result", f"The sentiment is {result}")
    else:
        messagebox.showwarning("Input Error", "Please enter a sentiment.")


# GUI setup
root = tk.Tk()
root.title("Sentiment Analysis App")

# Set initial dimensions
initial_width = 400
initial_height = 200
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_position = (screen_width - initial_width) // 2
y_position = (screen_height - initial_height) // 2
root.geometry(f"{initial_width}x{initial_height}+{x_position}+{y_position}")

# Label
label = tk.Label(root, text="Enter a sentiment:")
label.pack(pady=10)

# Input field
entry = tk.Entry(root, width=40)
entry.pack(pady=10)

# Button for sentiment analysis
analyze_button = tk.Button(root, text="Analyze Sentiment", command=predict_sentiment)
analyze_button.pack(pady=20)

# Run the GUI
root.mainloop()