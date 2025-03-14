import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def Chart_4():

    # Application title
    st.title("Linear Regression: Customers vs. Sales")

    # Load data
    df = pd.read_csv('data/Stores.csv')

    # Select columns of interest
    X = df[['Daily_Customer_Count']]  # Independent variable
    y = df['Store_Sales']  # Dependent variable

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict values for the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display metrics in Streamlit
    st.subheader("Model Metrics:")
    st.write(f"🔹 **Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"🔹 **Coefficient of Determination (R²):** {r2:.2f}")

    # Display the regression line equation
    st.subheader("Regression Line Equation:")
    st.latex(f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

    # Create scatter plot with regression line
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_test, y_test, color='blue', label='Actual Data')
    ax.plot(X_test, y_pred, color='red', label='Regression Line')
    ax.set_title('Relationship Between Daily Customer Count and Store Sales')
    ax.set_xlabel('Daily Customer Count')
    ax.set_ylabel('Store Sales')
    ax.legend()
    ax.grid(True)

    # Display the chart in Streamlit
    st.pyplot(fig)
