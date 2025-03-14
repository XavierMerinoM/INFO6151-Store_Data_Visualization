import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

def Chart_6():

    # Application title
    st.title("Daily Customers by Store Area Analysis")

    # Load data
    df = pd.read_csv('data/Stores.csv')

    # Define store area ranges
    n_bins = st.slider("Select the number of ranges:", min_value=3, max_value=10, value=5)
    binning = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

    # Apply discretization to create store area ranges
    df['Store_Area_Range'] = binning.fit_transform(df[['Store_Area']]).astype(int)

    # Calculate the average Daily_Customer_Count by store area range
    average_customers = df.groupby('Store_Area_Range')['Daily_Customer_Count'].mean()

    # Create labels for the ranges
    bin_edges = binning.bin_edges_[0]
    labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(n_bins)]

    # Visualize the average daily customers by store area range
    st.subheader("Average Daily Customers by Store Area Range")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, average_customers, color='skyblue', edgecolor='black')
    ax.set_title('Average Daily Customers by Store Area Range')
    ax.set_xlabel('Store Area Range')
    ax.set_ylabel('Average Daily Customers')
    ax.set_xticklabels(labels, rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)
