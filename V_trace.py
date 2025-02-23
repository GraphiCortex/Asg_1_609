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

# Plot Voltage Traces

# Convert sample indices to time in milliseconds
time_axis = np.arange(6000) / 10  # Since each sample is 0.1 ms

# Plot Control samples 
plt.figure(figsize=(12, 6))
for i, (filename, df) in enumerate(control_data.items()):
    if i >= 2:  # Limit to first 2 for visualization
        break
    plt.plot(time_axis, df["Voltage (mV)"], label=f"Control {filename}")

plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Voltage Traces: Control Group (Time Corrected)")
plt.legend()
plt.show()

# Plot Stuttering samples 
plt.figure(figsize=(12, 6))
for i, (filename, df) in enumerate(stuttering_data.items()):
    if i >= 2:  # Limit to first 2 for visualization
        break
    plt.plot(time_axis, df["Voltage (mV)"], linestyle="dashed", label=f"Stuttering {filename}")

plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Voltage Traces: Stuttering Group (Time Corrected)")
plt.legend()
plt.show()

# Plot Combined Control and Stuttering
plt.figure(figsize=(12, 6))
for i, (filename, df) in enumerate(control_data.items()):
    if i >= 1:
        break
    plt.plot(time_axis, df["Voltage (mV)"], label=f"Control {filename}")

for i, (filename, df) in enumerate(stuttering_data.items()):
    if i >= 1:
        break
    plt.plot(time_axis, df["Voltage (mV)"], linestyle="dashed", label=f"Stuttering {filename}")

plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Voltage Traces: Control vs Stuttering (Time Corrected)")
plt.legend()
plt.show()

