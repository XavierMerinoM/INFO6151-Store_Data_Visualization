import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def Chart_5():

    # Application title
    st.title("Store Sales Distribution Analysis")

    # Load data
    df = pd.read_csv('data/Stores.csv')

    # Analyze the distribution of Store_Sales
    store_sales = df['Store_Sales']

    # Calculate descriptive statistics
    mean_sales = store_sales.mean()
    median_sales = store_sales.median()
    q1 = store_sales.quantile(0.25)  # First quartile (25%)
    q3 = store_sales.quantile(0.75)  # Third quartile (75%)
    iqr = q3 - q1  # Interquartile range (IQR)

    # Identify outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = store_sales[(store_sales < lower_bound) | (store_sales > upper_bound)]

    # Display statistics in Streamlit
    st.subheader("Descriptive Statistics")
    st.write(f"🔹 **Mean Sales:** {mean_sales:.2f}")
    st.write(f"🔹 **Median Sales:** {median_sales:.2f}")
    st.write(f"🔹 **First Quartile (Q1):** {q1:.2f}")
    st.write(f"🔹 **Third Quartile (Q3):** {q3:.2f}")
    st.write(f"🔹 **Interquartile Range (IQR):** {iqr:.2f}")
    st.write(f"🔹 **Number of Outliers Detected:** {len(outliers)}")

    # Visualize the distribution with a boxplot
    st.subheader("Sales Distribution with Boxplot")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=store_sales, color='skyblue', ax=ax)
    ax.set_title('Store Sales Distribution')
    ax.set_xlabel('Store Sales')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    st.pyplot(fig)
