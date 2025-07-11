import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load the CSV file
df = pd.read_csv('life_expectancy.csv')

# Fix column name
df.rename(columns={'Life expectancy ': 'Life_expectancy'}, inplace=True)

# Drop missing values
df.dropna(inplace=True)

# Define features and target
X = df.drop(columns=['Life_expectancy', 'Country', 'Status', 'Year'])
y = df['Life_expectancy']

# Save the feature names used during training
feature_names = X.columns.tolist()
joblib.dump(feature_names, "model_columns.pkl")  # 🧠 Save to prevent mismatch

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model definition
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

# Cross-validation (optional)
mae_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print("Cross-validated MAE:", mae_scores.mean())

# Fit the model
model.fit(X_train, y_train)

# Evaluate
train_pred = model.predict(X_train)
train_mae = mean_absolute_error(y_train, train_pred)
print("Train MAE:", train_mae)

y_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
print("Test MAE:", test_mae)

print("Difference:", abs(train_mae - test_mae))

# Save model
joblib.dump(model, "random_forest_model.pkl")
print("✅ Model and columns saved successfully")
