import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib


# Assuming df is your DataFrame with the data
df = pd.read_excel("C:\\Users\\FeelT\\Desktop\\training set 2.xlsx")  # Load your data
df['Is Detected'] = df['Is Detected'].astype(int)  # Ensure the label is binary

# Define features and target
features = df.drop(columns=['Card Number', 'Is Detected'])
target = df['Is Detected']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

print(f'ROC AUC Score: {roc_auc}')
# # Save the trained model
# joblib.dump(model,'7amoksha_model.pkl')

# Print the feature names
print("Feature names:")
print(X_train.columns.tolist())