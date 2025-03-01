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
plt.title("Voltage Traces: Control Group")
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
plt.title("Voltage Traces: Stuttering Group")
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
plt.title("Voltage Traces: Control vs Stuttering")
plt.legend()
plt.show()


# # Define the range of interest (200 ms - 400 ms)

# start_index = int(200 * 10)  # Convert ms to sample index (10 kHz sampling rate)
# end_index = int(400 * 10)

# # Implementing a spike train averaging method from the paper "A Simple Algorithm for Averaging Spike Trains" by Julienne & Houghton

# def exponential_filter(spike_train, tau=5):
#     """
#     Applies an exponential filter to a spike train.

#     Parameters:
#         spike_train (np.array): Binary array indicating spike occurrences.
#         tau (float): Time constant of the exponential filter.

#     Returns:
#         np.array: Filtered spike train function.
#     """
#     filtered_train = np.zeros_like(spike_train, dtype=np.float64)
#     for t in range(1, len(spike_train)):
#         filtered_train[t] = filtered_train[t - 1] * np.exp(-1 / tau) + spike_train[t]
#     return filtered_train

# def map_spike_trains_to_functions(data_dict, start_index, end_index, tau=5):
#     """
#     Converts spike trains into functions using an exponential filter.

#     Parameters:
#         data_dict (dict): Dictionary containing neuron voltage traces.
#         start_index (int): Start index for averaging (200 ms in samples).
#         end_index (int): End index for averaging (400 ms in samples).
#         tau (float): Time constant of the exponential filter.

#     Returns:
#         np.array: Matrix of filtered spike train functions.
#     """
#     spike_functions = []
#     for df in data_dict.values():
#         voltage_trace = df["Voltage (mV)"].values[start_index:end_index]
#         spike_train = (voltage_trace > -20).astype(int)  # Convert to binary spike train
#         filtered_train = exponential_filter(spike_train, tau=tau)
#         spike_functions.append(filtered_train)
#     return np.array(spike_functions)

# def reconstruct_spike_train(avg_function, threshold=0.5):
#     """
#     Uses a greedy algorithm to reconstruct a spike train from the averaged function.

#     Parameters:
#         avg_function (np.array): The averaged filtered spike train function.
#         threshold (float): Value above which a spike is considered.

#     Returns:
#         np.array: Reconstructed spike train.
#     """
#     reconstructed = np.zeros_like(avg_function, dtype=int)
#     for i in range(1, len(avg_function) - 1):
#         if avg_function[i] > threshold and avg_function[i] > avg_function[i - 1]:
#             reconstructed[i] = 1
#     return reconstructed

# # Convert control and stuttering spike trains to functions
# control_functions = map_spike_trains_to_functions(control_data, start_index, end_index)
# stuttering_functions = map_spike_trains_to_functions(stuttering_data, start_index, end_index)

# # Compute the average function across neurons
# avg_function_control = np.mean(control_functions, axis=0)
# avg_function_stuttering = np.mean(stuttering_functions, axis=0)

# # Reconstruct the central spike train using the greedy algorithm
# central_spike_train_control = reconstruct_spike_train(avg_function_control)
# central_spike_train_stuttering = reconstruct_spike_train(avg_function_stuttering)

# # Define time window for visualization
# time_window = np.linspace(200, 400, len(avg_function_control))  # Time in ms

# # Plot the central spike train reconstruction
# plt.figure(figsize=(12, 6))
# plt.plot(time_window, avg_function_control, label="Averaged Function - Control", color='blue', alpha=0.6)
# plt.plot(time_window, avg_function_stuttering, label="Averaged Function - Stuttering", color='red', alpha=0.6)
# plt.scatter(time_window[central_spike_train_control == 1], np.ones_like(central_spike_train_control[central_spike_train_control == 1]) * 0.8, color='blue', marker='|', label="Central Spike Train - Control")
# plt.scatter(time_window[central_spike_train_stuttering == 1], np.ones_like(central_spike_train_stuttering[central_spike_train_stuttering == 1]) * 0.6, color='red', marker='|', label="Central Spike Train - Stuttering")
# plt.xlabel("Time (ms)")
# plt.ylabel("Spike Probability / Filtered Response")
# plt.title("Central Spike Train Reconstruction (Control vs. Stuttering)")
# plt.legend()
# plt.show()


