import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def Chart_7():

    # Application title
    st.title("Store Sales Trend Analysis")

    # Load data
    df = pd.read_csv('data/Stores.csv')

    # Create a fictitious time column (assuming the data is chronologically ordered)
    df['Time'] = np.arange(1, len(df) + 1)

    # Select columns of interest
    X = df[['Time']]
    y = df['Store_Sales']

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the Store_Sales values
    y_pred = model.predict(X)

    # Calculate evaluation metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Display metrics in Streamlit
    st.subheader("Evaluation Metrics")
    st.write(f" **Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² (Coefficient of Determination):** {r2:.2f}")

    # Display the trend equation
    st.subheader("Trend Line Equation")
    st.write(f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

    # Visualize the sales trend over time
    st.subheader("Sales Trend Chart")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Time'], df['Store_Sales'], marker='o', color='blue', label='Actual Sales')
    ax.plot(df['Time'], y_pred, color='red', label='Linear Trend')
    ax.set_title('Store Sales Trend Over Time')
    ax.set_xlabel('Time (Months)')
    ax.set_ylabel('Store Sales')
    ax.legend()
    ax.grid(True)

    # Display the chart in Streamlit
    st.pyplot(fig)
