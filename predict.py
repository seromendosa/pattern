import pandas as pd
import joblib

# Load the trained model from lola_model.py
from training import model

# Load the new data
new_data = pd.read_excel("C:\\Users\\FeelT\Desktop\\styled_data.xlsx")

# Preprocess the new data (same preprocessing steps as training data)
# For example, drop unnecessary columns and handle missing values

# Assuming you have processed the new data into features
new_features = new_data.drop(columns=['Card Number'])

# Make predictions
new_data_prob = model.predict_proba(new_features)[:, 1]

# Create a DataFrame with Card Number and corresponding probabilities
predictions_df = pd.DataFrame({'Card Number': new_data['Card Number'], 'Probability': new_data_prob})

# Print or save the predictions
print(predictions_df)
predictions_df.to_excel('results.xlsx', index=False)
