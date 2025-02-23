import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Define data path
control_files = sorted(glob.glob("Data/Control/*.txt"))
stuttering_files = sorted(glob.glob("Data/Stuttering/*.txt"))

# Function to load data
def load_data(files):
    datasets = {}
    for file in files:
        data = np.loadtxt(file)
        df = pd.DataFrame(data, columns=["Current (pA)", "Voltage (mV)"])
        datasets[os.path.basename(file)] = df
    return datasets

# Load datasets
control_data = load_data(control_files)
stuttering_data = load_data(stuttering_files)

import seaborn as sns

# Prepare data for heatmap: each row represents a neuron, columns are time points
control_matrix = np.array([df["Voltage (mV)"].values for df in control_data.values()])
stuttering_matrix = np.array([df["Voltage (mV)"].values for df in stuttering_data.values()])

# Plot Control Voltage Heatmap 
plt.figure(figsize=(12, 6))
sns.heatmap(control_matrix, cmap="coolwarm", cbar=True, 
            xticklabels=np.linspace(0, 6000, num=7, dtype=int)/10, 
            yticklabels=[f"N{i+1}" for i in range(control_matrix.shape[0])])

plt.xticks(ticks=np.linspace(0, 6000, num=7), labels=np.linspace(0, 600, num=7, dtype=int))
plt.xlabel("Time (ms)")
plt.ylabel("Neurons")
plt.title("Voltage Heatmap - Control Group")
plt.show()

# Plot Stuttering Voltage Heatmap 
plt.figure(figsize=(12, 6))
sns.heatmap(stuttering_matrix, cmap="coolwarm", cbar=True, 
            xticklabels=np.linspace(0, 6000, num=7, dtype=int)/10, 
            yticklabels=[f"N{i+1}" for i in range(stuttering_matrix.shape[0])])

plt.xticks(ticks=np.linspace(0, 6000, num=7), labels=np.linspace(0, 600, num=7, dtype=int))
plt.xlabel("Time (ms)")
plt.ylabel("Neurons")
plt.title("Voltage Heatmap - Stuttering Group")
plt.show()

# Select one control and one stuttering dataset for direct comparison
control_sample = list(control_data.values())[0]  # First control neuron
stuttering_sample = list(stuttering_data.values())[0]  # First stuttering neuron

# Combine the selected neurons into one matrix for heatmap
comparison_matrix = np.vstack([control_sample["Voltage (mV)"].values, 
                               stuttering_sample["Voltage (mV)"].values])

# Plot a one to one comparison voltage heatmap

plt.figure(figsize=(12, 6))
sns.heatmap(comparison_matrix, cmap="coolwarm", cbar=True, 
            xticklabels=np.linspace(0, 6000, num=7, dtype=int)/10, 
            yticklabels=["Control N1", "Stuttering N1"])

plt.xticks(ticks=np.linspace(0, 6000, num=7), labels=np.linspace(0, 600, num=7, dtype=int))

plt.xlabel("Time (ms)")
plt.ylabel("Neuron Type")
plt.title("Voltage Heatmap - Control vs Stuttering (Single Neuron Comparison)")
plt.show()
