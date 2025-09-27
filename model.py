import pandas as pd
import numpy as np
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import os
from sklearn.metrics import accuracy_score, classification_report
import evaluate

# Define emotion labels
EMOTIONS = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

def load_data():
    """Load and prepare the data for training"""
    # Load processed datasets
    train_df = pd.read_csv('data/train_processed.csv')
    test_df = pd.read_csv('data/test_processed.csv')
    
    # Use processed_text column for training
    text_column = 'processed_text'
    
    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])
    test_df['label_encoded'] = label_encoder.transform(test_df['label'])
    
    # Save label encoder classes for later use
    label_classes = label_encoder.classes_
    print(f"Labels: {label_classes}")
    
    return train_df, test_df, text_column, label_classes

def prepare_datasets(train_df, test_df, text_column, tokenizer):
    """Convert dataframes to HuggingFace datasets"""
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    return train_dataset, test_dataset

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Generate classification report
    report = classification_report(
        labels, 
        predictions, 
        target_names=[EMOTIONS[i] for i in range(len(EMOTIONS))],
        output_dict=True
    )
    
    # Extract metrics from the report
    metrics = {
        'accuracy': accuracy,
        'macro_f1': report['macro avg']['f1-score']
    }
    
    # Add per-class F1 scores
    for emotion_id, emotion_name in EMOTIONS.items():
        if emotion_name in report:
            metrics[f'f1_{emotion_name}'] = report[emotion_name]['f1-score']
    
    return metrics

def main():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Load and prepare data
    print("Loading data...")
    train_df, test_df, text_column, label_classes = load_data()
    
    # Initialize tokenizer and model
    print("Initializing DistilBERT...")
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(EMOTIONS)
    )
    
    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, test_dataset = prepare_datasets(train_df, test_df, text_column, tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./models/checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./models/logs",
        logging_steps=100,
        save_steps=500,
        push_to_hub=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Training model...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print("\nEvaluation Results:")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Save the model and tokenizer
    print("\nSaving model and tokenizer...")
    save_path = './models/distilbert_emotion'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save emotion mapping
    with open(f'{save_path}/emotion_labels.txt', 'w') as f:
        for emotion_id, emotion_name in EMOTIONS.items():
            f.write(f"{emotion_id}\t{emotion_name}\n")
    
    print(f"\nTraining complete! Model saved in {save_path}/")

if __name__ == "__main__":
    main()