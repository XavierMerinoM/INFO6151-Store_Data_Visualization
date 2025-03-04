import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import utils.data_manipulation as dm
from datetime import datetime

# Function to read the csv data
def read_data(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()