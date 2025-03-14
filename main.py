import streamlit as st

from modules.module1 import Chart_1
from modules.module2 import Chart_2
from modules.module3 import Chart_3
from modules.module4 import Chart_4
from modules.module5 import Chart_5
from modules.module6 import Chart_6
from modules.module7 import Chart_7
from modules.module8 import Chart_8


st.sidebar.header("Select a Visualization")
option = st.sidebar.selectbox(
    "Select your chart:",
    ("", 
    "1. Store Area Distribution", 
    "2. Relationship Between Area and Daily Customers",
    "3. Distribution of Available Products",
    "4. Relationship Between Customers and Sales",
    "5. Sales Distribution",
    "6. Average Customers by Area",
    "7. Sales Trend Over Time",
    "8. Correlation Heatmap")
)

# Show charts based on the selected option
if option == "1. Store Area Distribution":
    Chart_1()

elif option == "2. Relationship Between Area and Daily Customers":
    Chart_2()
    
elif option == "3. Distribution of Available Products":
    Chart_3()

elif option == "4. Relationship Between Customers and Sales":
    Chart_4()

elif option == "5. Sales Distribution":
    Chart_5()
elif option == "6. Average Customers by Area":
    Chart_6()

elif option == "7. Sales Trend Over Time":
    Chart_7()

elif option == "8. Correlation Heatmap":
    Chart_8()

else:
    st.write("Select an option from the menu to visualize the chart.")