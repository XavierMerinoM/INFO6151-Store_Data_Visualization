import streamlit as st
import utils.plot as pl
import utils.data_manipulation as dm
import utils.csv as csv

def init():
    #working with csv data
    st.title("Store Data Visualizer with Python")
            
    # Define the options to plot
    st.write("**Charts**")
    option = st.selectbox(
        'Select your plot:',
        ('','1. Distribution of Store Area', 
            '2. Relationship between Store Area and Daily Customer Count',
            '3. Distribution of Items Available',
            '4. Relationship between Daily Customer Count and Store Sales',
            '5. Distribution of Store Sales',
            '6. Average Daily Customer Count by Store Area',
            '7. Store Sales Trend Over Time',
            '8. Correlation Matrix Heatmap')
    )

    try:
        # Reading the csv file
        data = csv.read_data('utils\Stores.csv')
        # Preprocessing the data
        data_pr = dm.prepare_data(data)
        # Testing the plot data
        pl.scatter_plot((8,6), data_pr, 'Store_ID', 'Store_Sales', ['red'],
                        'Store vs Sales', 'Store ID', 'Store Sales', 4)
    except:
        st.error("view.py - Error to get data.")

    #if option == '1. Distribution of Store Area':