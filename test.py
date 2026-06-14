"""
test.py - Model Testing Script
Loads trained model, evaluates on test set, displays accuracy
"""

import joblib
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_test_data(test_size=0.3, random_state=42):
    """
    Load and split Olivetti faces dataset for testing
    
    Returns:
        X_test, y_test: Test features and labels
    """
    print("Loading test dataset...")
    olivetti = fetch_olivetti_faces(shuffle=True, random_state=random_state)
    X = olivetti.data
    y = olivetti.target
    
    _, X_test, _, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    return X_test, y_test

def load_model(filepath='savedmodel.pth'):
    """
    Load trained model from disk
    
    Args:
        filepath: Path to saved model
    
    Returns:
        model: Loaded model
    """
    print(f"Loading model from {filepath}...")
    model = joblib.load(filepath)
    print("Model loaded successfully!")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set with multiple metrics
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print("\nEvaluating model on test set...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics, y_pred

def display_results(metrics):
    """
    Display evaluation results in formatted output
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Test Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1-Score:       {metrics['f1_score']:.4f}")
    print("="*50 + "\n")

def main():
    """Main testing pipeline"""
    try:
        # Load test data
        X_test, y_test = load_test_data()
        
        # Load model
        model = load_model('savedmodel.pth')
        
        # Evaluate model
        metrics, predictions = evaluate_model(model, X_test, y_test)
        
        # Display results
        display_results(metrics)
        
        print("✓ Testing pipeline completed successfully!")
        
    except FileNotFoundError:
        print("✗ Error: savedmodel.pth not found. Please run train.py first.")
    except Exception as e:
        print(f"✗ Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
