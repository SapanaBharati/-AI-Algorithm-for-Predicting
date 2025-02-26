# AI-for-RTL-Timing-Predictor

Project Name: AI Algorithm for Predicting Timing Violations in RTL Design
## Code:
```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def load_data(file_path):
    """Load dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train multiple ML models and evaluate their performance."""
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
        "Neural Network": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name}: MAE = {mae:.4f}, R² Score = {r2:.4f}")
        
        # Scatter plot of actual vs predicted values
        plt.scatter(y_test, y_pred, label=name, alpha=0.6)
    
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
    plt.xlabel("Actual Logic Depth")
    plt.ylabel("Predicted Logic Depth")
    plt.title("Model Predictions vs Actual Values")
    plt.legend()
    plt.show()

def main():
    file_path = input("Enter the dataset CSV file path: ")
    df = load_data(file_path)
    
    # Extract features and target variable
    X = df[['Fan_In', 'Fan_Out', 'Path_Length', 'Gate_Count']]
    y = df['Logic_Depth']
    
    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
