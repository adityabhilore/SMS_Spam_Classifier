# SMS Spam Classifier

A machine learning project that classifies text messages (SMS) as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP).

---

## Project Overview

This project uses a trained ML model to classify whether a given message is spam or not. It includes:
- Data preprocessing (cleaning, tokenization)
- TF-IDF Vectorization
- Model Training (e.g., Naive Bayes / Logistic Regression)
- A simple UI built with Python for prediction

---

## Project Structure
sms-spam-classifier/
│
├── app.py # UI file to take input and predict
├── model.py # Script to train the model
├── model.pkl # Trained ML model
├── vectorizer.pkl # TF-IDF vectorizer
├── spam.csv # Dataset used for training
├── README.md # Project documentation

How it Works
The dataset is cleaned and preprocessed.
Text is transformed using TF-IDF Vectorizer.
Model is trained on labeled data using a classifier like Multinomial Naive Bayes.
The trained model and vectorizer are saved using pickle.
The app.py loads them and takes user input to predict.

Technologies Used
Python
Pandas, NumPy
Scikit-learn
Pickle
[Optional: Streamlit / Tkinter / Flask] for UI
