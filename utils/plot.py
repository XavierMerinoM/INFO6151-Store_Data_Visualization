import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st

# Function to plot a box plot
def box_plot(fig_size, data_x, data_y, title, x_label, y_label, linewidth,
            style, palette, marker, markersize):
    plt.figure(figsize=fig_size)
    sns.set_style(style)
    flierprops = dict(marker=marker, markersize=markersize)
    sns.boxplot(x=data_x, y=data_y, hue=data_x, linewidth=linewidth,
               palette=palette, flierprops=flierprops)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    st.pyplot(plt)

# Function to plot a bar plot
def bar_plot(fig_size, data_x, data_y, colors, title, x_label, y_label, edgecolor):
    plt.figure(figsize=fig_size)
    plt.barh(data_x, data_y, color=colors, edgecolor = edgecolor, linewidth = 2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    st.pyplot(plt)

# Function to plot a scatter plot
def scatter_plot(fig_size, data, x, y, colors, title, x_label, y_label, data_filter = '', grid = True, data_list = []):
    plt.figure(figsize=fig_size)
    if data_list:
        # Plotting by each specie
        for i in range(len(data_list)):
            data_x = data[x][data[data_filter] == data_list[i]]
            data_y = data[y][data[data_filter] == data_list[i]]
            # Calculating the ratio of 'sepal length' to 'sepal width'
            ratio = list((np.array(data_x) / np.array(data_y)))
            # Plotting
            plt.scatter(data_x, 
                        data_y, 
                        color=colors[i],
                        label=data_list[i],
                        s = ratio,
                        marker='x')
    else:
        # Plotting
        plt.scatter(data[x], data[y], color=colors[0])
        
    plt.title(title, fontweight ='bold', fontsize = 15)
    plt.xlabel(x_label, fontweight ='bold', fontsize = 15)
    plt.ylabel(y_label, fontweight ='bold', fontsize = 15)
    plt.legend()
    plt.grid(grid)
    st.pyplot(plt)