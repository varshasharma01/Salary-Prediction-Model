# Import libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
dataset = pd.read_csv("hiring.csv")

# Features (X) and Target (y)
X = dataset.iloc[:, :3]   # experience, test_score, interview_score
y = dataset.iloc[:, -1]   # salary

# Split dataset into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# ---- MODEL TESTING ----
y_pred = regressor.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("Predicted salaries:", y_pred.tolist())
print("Actual salaries:", y_test.tolist())




# ---- SAVE MODEL ----
pickle.dump(regressor, open("model.pkl", "wb"))

# ---- LOAD MODEL (for deployment/testing) ----
model = pickle.load(open("model.pkl", "rb"))
print("Example Prediction:", model.predict([[2, 9, 6]]))
