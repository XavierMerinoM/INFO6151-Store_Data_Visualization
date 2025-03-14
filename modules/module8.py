import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def Chart_8():

    # Application title
    st.title("Correlation Analysis between Variables")

    # Load data
    df = pd.read_csv('data/Stores.csv')

    # Select columns of interest
    columns = ['Store_Area', 'Items_Available', 'Daily_Customer_Count', 'Store_Sales']
    df_selected = df[columns]

    # Calculate the correlation matrix
    correlation_matrix = df_selected.corr()

    # Visualize the correlation matrix using a heatmap
    st.subheader("Correlation Matrix between Variables")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    plt.title('Correlation Matrix between Variables')

    # Display the chart in Streamlit
    st.pyplot(fig)
