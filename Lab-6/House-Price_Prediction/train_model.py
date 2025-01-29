import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load Data
data = pd.read_csv("house-prices.csv")
df = pd.DataFrame(data)

# Features and Target
X = df[["SqFt"]]
y = df["Price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")