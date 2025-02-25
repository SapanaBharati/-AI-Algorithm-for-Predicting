# AI-Algorithm-for-Predicting








import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def main():
    # Taking user input for training dataset
    n_train = int(input("Enter number of training samples: "))
    X_train, y_train = [], []
    
    print("Enter training features (Fan-In and Fan-Out) and corresponding depths:")
    for _ in range(n_train):
        fan_in, fan_out, depth = map(float, input().split())
        X_train.append([fan_in, fan_out])
        y_train.append(depth)
    
    # Convert to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Taking user input for test dataset
    n_test = int(input("Enter number of test samples: "))
    X_test, y_test = [], []
    
    print("Enter test features (Fan-In and Fan-Out) and corresponding actual depths:")
    for _ in range(n_test):
        fan_in, fan_out, depth = map(float, input().split())
        X_test.append([fan_in, fan_out])
        y_test.append(depth)
    
    # Convert to numpy arrays
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Display results
    print("Predicted Logic Depths:", y_pred)
    print("Mean Absolute Error (MAE):", mae)

if __name__ == "__main__":
    main()
