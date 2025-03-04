import pandas as pd

# Function to preprocess data
def prepare_data(data):
    try:
        # Removing duplicates
        df_nd = data.drop_duplicates()

        # Handling missing values
        df_nm = df_nd.dropna()

        # The first column is 'Store ID'. 
        # The name must be changed to be able to consult by name
        df_nm.rename(columns={df_nm.columns[0]: 'Store_ID'}, inplace=True)

        return df_nm
    except Exception as e:
        print("data_manipulation.prepare_data - Error: ", e)