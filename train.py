"""
train.py - Model Training Script
Loads Olivetti faces dataset, trains DecisionTreeClassifier, saves model
"""

import joblib
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os

def load_data():
    """
    Load Olivetti faces dataset from sklearn
    
    Returns:
        X: Feature matrix (400 samples, 1850 features each)
        y: Target labels (10 classes, 40 samples per class)
    """
    print("Loading Olivetti faces dataset...")
    olivetti = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = olivetti.data
    y = olivetti.target
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    return X, y

def split_data(X, y, test_size=0.3, random_state=42):
    """
    Split data into training and testing sets
    
    Args:
        X: Feature matrix
        y: Target labels
        test_size: Proportion of test set (default: 30%)
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    print(f"Splitting data: 70% train, 30% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Ensures balanced class distribution
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train DecisionTreeClassifier model
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        model: Trained DecisionTreeClassifier
    """
    print("Training DecisionTreeClassifier...")
    model = DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    print(f"Training accuracy: {train_accuracy:.4f}")
    return model

def save_model(model, filepath='savedmodel.pth'):
    """
    Save trained model using joblib
    
    Args:
        model: Trained model
        filepath: Path to save model
    """
    print(f"Saving model to {filepath}...")
    joblib.dump(model, filepath)
    print(f"Model saved successfully!")

def main():
    """Main training pipeline"""
    try:
        # Load data
        X, y = load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Save model
        save_model(model, 'savedmodel.pth')
        
        print("\n✓ Training pipeline completed successfully!")
        
    except Exception as e:
        print(f"✗ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
