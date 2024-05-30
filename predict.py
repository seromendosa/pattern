import pandas as pd
import joblib
import os

# Path to the trained model
model_path = '7amoksha_model.pkl'

# Load the trained model
pipeline = joblib.load(model_path)

# Load new testing set
new_test_data = pd.read_excel("C:\\Users\\FeelT\\Desktop\\FINAL PY PROFJECT\\pattern\\predict.xlsx")

# Ensure the new test data has the same preprocessing as the training data
# Drop 'Card Number' and 'Is Detected' if it exists in the new test data
new_test_features = new_test_data.drop(
    columns=['Card Number', 'Is Detected'], errors='ignore')

# Make predictions
new_predictions = pipeline.predict(new_test_features)
new_probabilities = pipeline.predict_proba(new_test_features)[:, 1]

# Add the predictions and probabilities to the new test data
new_test_data['Predictions'] = new_predictions
new_test_data['Probabilities'] = new_probabilities

# Get the directory of the model file
model_directory = os.path.dirname(model_path)

# Define the output file path
output_file = os.path.join(model_directory, 'predictions_output.xlsx')

# Save the results to a new sheet in the Excel file
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    new_test_data.to_excel(writer, sheet_name='Predictions', index=False)

print(f"Predictions and probabilities have been saved to {output_file}")