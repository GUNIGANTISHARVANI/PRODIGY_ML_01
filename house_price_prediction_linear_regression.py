import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset (assuming you have downloaded 'train.csv' and 'test.csv')
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Display the first few rows of the dataset to understand its structure
print(train_data.head())

# Select relevant features for the model: Square footage, number of bedrooms, and bathrooms
# We will assume the following columns are present in the dataset:
# 'GrLivArea' - Above ground living area (square footage)
# 'TotRmsAbvGrd' - Total rooms above ground
# 'FullBath' - Number of full bathrooms
# 'HalfBath' - Number of half bathrooms

# For simplicity, let's use 'GrLivArea' (square footage), 'BedroomAbvGr' (number of bedrooms), 
# and 'FullBath' (number of full bathrooms).

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
X = train_data[features]
y = train_data['SalePrice']  # Target variable: Sale price of the house

# Handling missing values - for this example, we can drop rows with missing target values.
train_data = train_data.dropna(subset=['SalePrice'])

# Train-test split: 80% for training, 20% for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling: Scale the features for better performance in case of large differences in scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val_scaled)

# Evaluate the model performance
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

# Print the evaluation metrics
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')

# Plot Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred, alpha=0.5)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Price')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line
plt.show()

# Residuals plot
residuals = y_val - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Sale Price')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()

# Distribution of residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Distribution of Residuals')
plt.show()

# Predicting on the test set
X_test = test_data[features]
X_test_scaled = scaler.transform(X_test)
test_predictions = model.predict(X_test_scaled)

# Save predictions to a CSV file
submission = pd.DataFrame({
    'Id': test_data['Id'],  # Assuming the test data has an 'Id' column
    'SalePrice': test_predictions
})

submission.to_csv('house_price_predictions.csv', index=False)

print("Predictions saved to 'house_price_predictions.csv'")

