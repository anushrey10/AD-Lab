import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load dataset
data = pd.read_csv('house_data.csv')  

# Selecting relevant features
features = ['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'view',
            'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
            'zipcode', 'lat', 'long']
X = data[features]
y = data['price']

# Identify numerical and categorical features
num_features = ['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'sqft_above', 'sqft_basement', 
                'yr_built', 'yr_renovated', 'lat', 'long']
cat_features = ['waterfront', 'view', 'condition', 'grade', 'zipcode']

# Preprocessing: Scale numerical features & One-Hot Encode categorical features
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])

# Use a more advanced model: Random Forest Regressor
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Save model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
    print("Model saved to 'model.pkl'")  