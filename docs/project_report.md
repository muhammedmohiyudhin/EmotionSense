PROJECT TITLE
Automated Detection of Different Emotions from Textual Comments and Feedback

INTRODUCTION
This project detects human emotions from text using deep learning. It helps understand customer feedback automatically

PROBLEM STATEMENT
Manually analyzing emotions from large text data is slow and inefficient.

OBJECTIVES
1.Detect emotions from text
2.Train an AI model
3.Build a web interface
4.Evaluate performance

DATASET DESCRIPTION
Dataset collected from Kaggle. Contains text and emotion labels such as sadness, joy, anger, etc.

DATA PREPROCESSING
Converted TXT to CSV, cleaned text, removed symbols, and normalized data.

MODEL DESCRIPTION
BERT transformer model was used for emotion classification.

TRAINING PROCESS
Model trained using PyTorch and HuggingFace Transformers for 2 epochs.

EVALUATION AND RESULTS
Accuracy: 95.7%

Classification Report:
              precision    recall  f1-score   support

     sadness       0.98      0.99      0.98       598
       anger       0.95      0.95      0.95       261
        love       0.98      0.82      0.89       161
    surprise       0.84      0.89      0.86        70
        fear       0.95      0.90      0.92       247
         joy       0.95      0.99      0.97       663

    accuracy                           0.96      2000
   macro avg       0.94      0.92      0.93      2000
weighted avg       0.96      0.96      0.96      2000

USER INTERFACE
Streamlit web app used to enter text and display predicted emotion.

TESTING
Test cases, test scenarios, and test design documents were created in CSV format.

TOOLS AND TECHNOLOGIES USED
Python
PyTorch
Transformers
BERT
Streamlit
VS Code

CHALLENGES FACED
Library installation issues
Dataset conversion
Model training time

FUTURE ENHANCEMENTS
Multilingual support
Voice emotion detection
Mobile app

CONCLUSION
The system successfully detects emotions with high accuracy.

REFERENCES
Kaggle Dataset
HuggingFace Documentation
PyTorch Documentation