# Project Design

Simple flow:
Input text --> Preprocess (clean, remove stopwords with NLTK) --> DistilBERT (classify 6 emotions: sadness, joy, love, anger, fear, surprise) --> Output (label + mental suggestion in Streamlit).
