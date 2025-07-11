import pandas as pd
import joblib

# Load model and training columns
model = joblib.load("random_forest_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Sample input (ensure all feature names match)
new_data = {
    'Adult_Mortality': 263,
    'infant_deaths': 62,
    'Alcohol': 0.01,
    'percentage_expenditure': 71.27962362,
    'Hepatitis_B': 65,
    'Measles': 1154,
    'BMI': 25.0,  # assuming overweight, adjust as needed
    'under_five_deaths': 83,
    'Polio': 6,
    'Total_expenditure': 8.16,
    'Diphtheria': 65,
    'HIV/AIDS': 0.1,
    'GDP': 584.25921,
    'Population': 33736494,
    'thinness__1_19_years': 17.2,
    'thinness_5_9_years': 17.3,
    'Income_composition_of_resources': 0.479,
    'Schooling': 10.1,
    'smoking': 1,               # smoker
    'fast_food_freq': 2,
    'exercise_per_week': 1,     # less exercise due to condition
    'chronic_disease': 1,       # chronic illness (fatty liver + Becker syndrome)
    'blood_pressure': 130,      # slightly elevated
    'blood_sugar': 120,         # prediabetic
    'bone_mass_index': 2.99
}

# Convert to DataFrame
df_example = pd.DataFrame([new_data])

# Ensure columns are in correct order
df_example = df_example[model_columns]

# Predict
prediction = model.predict(df_example)[0]
print("Predicted Life Expectancy:", prediction)
