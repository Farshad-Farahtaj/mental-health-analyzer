import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from sklearn.model_selection import train_test_split

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def clean_text(text):
    """
    Clean text by removing stopwords and converting to lowercase
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Join tokens back into text
    return ' '.join(tokens)

def main():
    # Ensure data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Load the dataset
    try:
        train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        print("Dataset loaded successfully.")
        print(f"Original shape: {train_df.shape}")
    except FileNotFoundError:
        print("Error: train.csv not found in data folder!")
        return

    # Check if we need to split the data
    if 'test.csv' not in os.listdir(data_dir):
        print("No test.csv found. Splitting train.csv into train and test sets...")
        train_df, test_df = train_test_split(
            train_df,
            test_size=0.2,
            random_state=42,
            stratify=train_df['label'] if 'label' in train_df.columns else None
        )
    else:
        print("Loading existing test.csv...")
        test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # Clean the text data
    print("Cleaning training data...")
    train_df['processed_text'] = train_df['text'].apply(clean_text)
    
    print("Cleaning test data...")
    test_df['processed_text'] = test_df['text'].apply(clean_text)
    
    # Save processed datasets
    train_output_path = os.path.join(data_dir, 'train_processed.csv')
    test_output_path = os.path.join(data_dir, 'test_processed.csv')
    
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"\nPreprocessing complete. Files saved as:")
    print(f"- {train_output_path}")
    print(f"- {test_output_path}")
    
    # Print sample of processed text
    print("\nSample of processed text:")
    print("Original:", train_df['text'].iloc[0][:100], "...")
    print("Processed:", train_df['processed_text'].iloc[0][:100], "...")

if __name__ == "__main__":
    main()