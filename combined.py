import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import io

# Function for the Pattern Detector
def check_pattern(row, pattern_range_end, threshold_value, outlier_value, consecutive_count_value):
    detected_indices = []
    consecutive_count = 0
    
    for i in range(1, pattern_range_end):
        if row.iloc[i] >= threshold_value:
            consecutive_count += 1
            if consecutive_count >= consecutive_count_value:
                detected_indices.extend(range(i - 1, i + 1))
                return detected_indices
        else:
            consecutive_count = 0
    
    for i in range(1, len(row)):
        if row.iloc[i] >= outlier_value:
            detected_indices.append(i)
    
    return detected_indices

def is_detected(row):
    for idx in detected_cells[row.name]:
        if isinstance(idx, int):
            return True
        elif len(idx) > 1:
            return True
    return False

def process_excel(file_path, pattern_range_end, threshold_value, outlier_value, consecutive_count_value, sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        global detected_cells
        detected_cells = df.apply(check_pattern, axis=1, pattern_range_end=pattern_range_end, threshold_value=threshold_value, 
                                  outlier_value=outlier_value, consecutive_count_value=consecutive_count_value)
        df['Is Detected'] = df.apply(is_detected, axis=1)
        return df
    except Exception as e:
        st.error(e)
        return None

# Function for the Random Forest Classifier Training
def train_random_forest(df):
    df['Is Detected'] = df['Is Detected'].astype(int)
    features = df.drop(columns=['Card Number', 'Is Detected'])
    target = df['Is Detected']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    imputer = SimpleImputer(strategy='mean')
    pipeline = Pipeline(steps=[
        ('imputer', imputer),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    st.write(f'Accuracy: {accuracy:.2f}')
    st.write(f'Precision: {precision:.2f}')
    st.write(f'Recall: {recall:.2f}')
    st.write(f'F1 Score: {f1:.2f}')
    st.write(f'ROC AUC Score: {roc_auc:.2f}')
    
    model_buffer = io.BytesIO()
    joblib.dump(pipeline, model_buffer)
    st.download_button(
        label="Download Model",
        data=model_buffer,
        file_name='random_forest_model.pkl',
        mime='application/octet-stream'
    )
    
    st.write("Feature names:")
    st.write(X_train.columns.tolist())

# Function for making predictions with the trained model
def make_predictions(file_path, model_file):
    try:
        pipeline = joblib.load(model_file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    new_test_data = pd.read_excel(file_path)
    new_test_features = new_test_data.drop(columns=['Is Detected'], errors='ignore')
    
    new_predictions = pipeline.predict(new_test_features)
    new_probabilities = pipeline.predict_proba(new_test_features)[:, 1]
    
    new_test_data['Predictions'] = new_predictions
    new_test_data['Probabilities'] = new_probabilities
    
    output_file = 'predictions_output.xlsx'
    new_test_data.to_excel(output_file, sheet_name='Predictions', index=False)
    
    st.write("Predictions and probabilities have been saved to:", output_file)
    st.write(new_test_data)

# Function to determine column data types for PostgreSQL
def get_column_data_type(series):
    if pd.api.types.is_integer_dtype(series):
        return "INT"
    elif pd.api.types.is_float_dtype(series):
        return "DECIMAL"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    else:
        max_length = series.astype(str).map(len).max()
        return f"VARCHAR({max_length})"

# Function to create a PostgreSQL table creation SQL command
def create_table_sql(df, table_name="uploaded_data"):
    create_table_statement = f"CREATE TABLE {table_name} (\n"
    for column in df.columns:
        column_type = get_column_data_type(df[column])
        create_table_statement += f"    {column.replace(' ', '_').lower()} {column_type},\n"
    create_table_statement = create_table_statement.rstrip(",\n") + "\n);"
    return create_table_statement

# Main function to create the Streamlit app
def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Pattern Detector", "Random Forest Classifier Training", "Model Predictions", "CSV/Excel to PostgreSQL Converter"])
    
    if app_mode == "Pattern Detector":
        st.title("Pattern Detector")
        
        pattern_range_end = st.number_input("Months to check:", min_value=1, value=10, step=1, help="Enter the Number of months to check")
        threshold_value = st.number_input("Value to check:", min_value=0, value=50, step=1, help="Enter the value to check in each cell")
        consecutive_count_value = st.number_input("Consecutive Count Value:", min_value=2, value=2, step=1, help="Check the threshold value for how many consecutive months?")
        outlier_value = st.number_input("Outlier Value:", min_value=0, value=100, step=1, help="If no pattern check for outlier of?")
        
        file = st.file_uploader("Upload Excel file", type=["xlsx"])
        
        if file is not None:
            sheet_names = pd.ExcelFile(file).sheet_names
            sheet_name = st.selectbox("Select Sheet", sheet_names)
        
            if st.button("Process"):
                df = process_excel(file, pattern_range_end, threshold_value, outlier_value, consecutive_count_value, sheet_name)
                if df is not None:
                    st.write(df)
                    
                    # Create a summary chart
                    st.subheader("Summary Chart: Detected vs Not Detected")
                    detected_count = df['Is Detected'].sum()
                    not_detected_count = len(df) - detected_count
                    counts = pd.DataFrame({'Counts': [detected_count, not_detected_count]}, index=['Detected', 'Not Detected'])
                    st.bar_chart(counts)
                    
                    # Detailed analysis
                    st.subheader("Detailed Analysis")
                    st.write(df.loc[df['Is Detected'], :])
                    
                    # # Visualization of Patterns
                    # st.subheader("Pattern Visualization")
                    # plt.figure(figsize=(10, 6))
                    # sns.heatmap(df.iloc[:, 1:pattern_range_end], annot=True, cmap="YlGnBu")
                    # st.pyplot(plt)
                    
                    # # Customizable heatmap
                    # st.subheader("Customizable Heatmap")
                    # cmap = st.selectbox("Select Heatmap Color Palette", sns.color_palette())
                    # annot = st.checkbox("Show Annotations", value=True)
                    # plt.figure(figsize=(10, 6))
                    # sns.heatmap(df.iloc[:, 1:pattern_range_end], annot=annot, cmap=cmap)
                    # st.pyplot(plt)
        
        st.write("---")
        st.write("Made by Business Excellence department")
        st.write("Manager: Bassant Shereba")
        st.write("Developer: Dr. Mohamed Magdy")
    
    elif app_mode == "Random Forest Classifier Training":
        st.title("Random Forest Classifier Training App")
        
        st.write("Upload your Excel file to train the model:")
        file = st.file_uploader("Upload Excel file", type=["xlsx"])
        
        if file is not None:
            df = pd.read_excel(file)
            st.write("Data preview:")
            st.write(df.head())
            
            if 'Is Detected' in df.columns:
                train_random_forest(df)
            else:
                st.error("The uploaded file does not contain the required 'Is Detected' column.")
        
        st.write("---")
        st.write("Made by Business Excellence department")
        st.write("Manager: Bassant Shereba")
        st.write("Developer: Dr. Mohamed Magdy")
    
    elif app_mode == "Model Predictions":
        st.title("Model Predictions")
        
        file = st.file_uploader("Upload Excel file for predictions", type=["xlsx"])
        model_file = st.file_uploader("Upload Trained Model", type=["pkl"])
        
        if file is not None and model_file is not None:
            if st.button("Make Predictions"):
                make_predictions(file, model_file)
        
        st.write("---")
        st.write("Made by Business Excellence department")
        st.write("Manager: Bassant Shereba")
        st.write("Developer: Dr. Mohamed Magdy")
    
    elif app_mode == "CSV/Excel to PostgreSQL Converter":
        st.title("Flexible CSV/Excel to PostgreSQL Converter")

        uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["csv", "xlsx"])

        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(df.head())
                table_name = st.text_input("Enter table name:", "uploaded_data")
                if st.button('Generate SQL'):
                    create_table_statement = create_table_sql(df, table_name)
                    st.write("Table Creation SQL:")
                    st.code(create_table_statement, language='sql')
                    st.download_button("Download SQL Command", create_table_statement, file_name='create_table.sql')
            else:
                sheet_names = pd.ExcelFile(uploaded_file).sheet_names
                sheet_name = st.selectbox("Select Sheet", sheet_names)
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                st.write("Data Preview:")
                st.dataframe(df.head())
                table_name = st.text_input("Enter table name:", "uploaded_data")
                if st.button('Generate SQL'):
                    create_table_statement = create_table_sql(df, table_name)
                    st.write("Table Creation SQL:")
                    st.code(create_table_statement, language='sql')
                    st.download_button("Download SQL Command", create_table_statement, file_name='create_table.sql')

if __name__ == "__main__":
    main()
