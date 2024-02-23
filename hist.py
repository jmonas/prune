import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the model state dictionary from .pt file
model_path = '/Users/owner/Desktop/cos598d_pruning/Results/data/singleshot/0/model.pt'  # Replace with the path to your .pt file
model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Prepare the figure and 3D axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define the number of bins for the histogram
num_bins = 25

# Iterate over each layer and plot the histogram
for idx, (layer_name, weights) in enumerate(model_state_dict.items()):
    if 'weight' in layer_name:  # Filter out other parameters like biases
        # Flatten the weights to 1D and compute the histogram
        hist, bins = np.histogram(weights.numpy().flatten(), bins=num_bins)
        
        # The x position corresponds to the bin centers
        x_position = (bins[:-1] + bins[1:]) / 2
        
        # The y position corresponds to the layer number
        y_position = np.ones_like(x_position) * idx
        
        # The z position (height) corresponds to the histogram counts
        z_position = np.zeros_like(x_position)
        
        # Define the width, depth, and color of the bars
        dx = np.ones_like(hist) * (bins[1] - bins[0])
        dy = np.ones_like(hist) * 0.8  # A small gap between layers for clarity
        dz = hist
        
        ax.bar3d(x_position, y_position, z_position, dx, dy, dz, zsort='average')

# Set labels and title
ax.set_xlabel('Weight Values')
ax.set_ylabel('Layer Number')
ax.set_zlabel('Counts')

# Set the viewing angle for better visibility
# ax.view_init(elev=30, azim=45)

# Show the plot
plt.show()
