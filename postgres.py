import streamlit as st
import pandas as pd

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

def create_table_sql(df, table_name="uploaded_data"):
    create_table_statement = f"CREATE TABLE {table_name} (\n"
    for column in df.columns:
        column_type = get_column_data_type(df[column])
        create_table_statement += f"    {column.replace(' ', '_').lower()} {column_type},\n"
    create_table_statement = create_table_statement.rstrip(",\n") + "\n);"
    return create_table_statement

def main():
    st.title("Flexible CSV/Excel to PostgreSQL Converter")

    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("Data Preview:")
        st.dataframe(df.head())

        table_name = st.text_input("Enter table name:", "uploaded_data")

        if st.button('Generate SQL'):
            create_table_statement = create_table_sql(df, table_name)
            st.write("Table Creation SQL:")
            st.code(create_table_statement, language='sql')

            # Optional: Provide download link for the SQL command
            st.download_button("Download SQL Command", create_table_statement, file_name='create_table.sql')

if __name__ == '__main__':
    main()
