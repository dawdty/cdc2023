import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load your wildfire data from an Excel file or other sources
# Replace 'your_data.csv' with the actual data file path
data = pd.read_csv('edmund_data2.csv')

# Select the features for clustering (date, longitude, latitude)
features = data[['DISCOVERY_DATE', 'LONGITUDE', 'LATITUDE', 'FIRE_YEAR']].values

# Extract LONGITUDE, LATITUDE, and DISCOVERY_DATE columns
longitude = data['LONGITUDE']
latitude = data['LATITUDE']
discovery_date = data['DISCOVERY_DATE']
fire_year = data['FIRE_YEAR']

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot data points
ax.scatter(longitude, latitude, fire_year, marker='o', c=discovery_date, cmap='viridis')

# Set axis labels
ax.set_xlabel('LONGITUDE')
ax.set_ylabel('LATITUDE')
ax.set_zlabel('FIRE_YEAR')

# Set plot title
ax.set_title('Wildfire Data in 3D')

# Show the plot
plt.show()