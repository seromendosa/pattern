import streamlit as st
import pandas as pd

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

def process_excel(file_path, pattern_range_end, threshold_value, outlier_value, consecutive_count_value):
    try:
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        global detected_cells
        detected_cells = df.apply(check_pattern, axis=1, pattern_range_end=pattern_range_end, threshold_value=threshold_value, 
                                  outlier_value=outlier_value, consecutive_count_value=consecutive_count_value)
        df['Is Detected'] = df.apply(is_detected, axis=1)
        return df
    except Exception as e:
        st.error(e)
        return None

def main():
    st.title("MedRight Pattern Detector")
    
    pattern_range_end = st.number_input("Pattern Range (end):", min_value=1, value=10, step=1)
    threshold_value = st.number_input("Threshold Value:", min_value=0, value=50, step=1)
    outlier_value = st.number_input("Outlier Value:", min_value=0, value=100, step=1)
    consecutive_count_value = st.number_input("Consecutive Count Value:", min_value=2, value=2, step=1)
    
    file = st.file_uploader("Upload Excel file", type=["xlsx"])
    
    if st.button("Process"):
        if file is not None:
            df = process_excel(file, pattern_range_end, threshold_value, outlier_value, consecutive_count_value)
            if df is not None:
                st.write(df)
                
                # Create a summary chart
                st.subheader("Summary Chart: Detected vs Not Detected")
                detected_count = df['Is Detected'].sum()
                not_detected_count = len(df) - detected_count
                counts = pd.DataFrame({'Counts': [detected_count, not_detected_count]}, index=['Detected', 'Not Detected'])
                st.bar_chart(counts)
    
    # Citation footer
    st.write("---")
    st.write("Made by Business Excellence department")
    st.write("Manager: Bassant Shereba")
    st.write("Developer: Dr. Mohamed Magdy")


if __name__ == "__main__":
    main()
