import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import os
import time

from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from pyts.datasets import load_gunpoint
from PIL import Image
from random import *

def plot_csv_with_limited_x_axis(csv_path, column_index, s_idx):
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Select the second column (index 1)
    data = df.iloc[:, column_index]
    
    # Extract data up to the last s_idx value
    last_index = s_idx[-1]
    data_subset = data.iloc[:last_index + 1]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(data_subset, label="Data Values", color='blue')
    
    # Draw rectangles based on s_idx
    for start, end in zip(s_idx, s_idx[1:]):
        plt.axvspan(start, end - 1, edgecolor='red', facecolor='none', linewidth=1)
    
    # Add rectangle for the last segment from the last s_idx to the last index
    plt.axvspan(s_idx[-1], last_index, edgecolor='red', facecolor='none', linewidth=1)
    
    # Limit the x-axis range to the last s_idx value
    plt.xlim(0, last_index)
    
    # Customize labels and title
    plt.title("Cycle based Data Segmentation", fontsize=32, pad=20)
    plt.xlabel("Timestamp", fontsize=28)
    plt.ylabel("Value", fontsize=28)
    plt.legend(fontsize=24)
    plt.savefig('makegraph_limited_x_axis.png')

normal = pd.read_csv("test_10.csv")

path = ''


column_values = normal.iloc[:, 18]

# 12657~13085
selected_values = column_values[12657:13085]

plt.figure(figsize=(16, 10))
plt.plot(selected_values)
plt.xlabel("Timestamp", fontsize = 32)
plt.ylabel("Value", fontsize = 32)
plt.tick_params(axis='x', labelsize=26)
plt.tick_params(axis='y', labelsize=26)
plt.title('Part of LIT301', fontsize=40, pad=20)
plt.savefig("LIT301_portion.png")
