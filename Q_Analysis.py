from scipy.signal import find_peaks
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

# Constants
SPIKE_THRESHOLD = -20  # Voltage threshold for spike detection (mV)
SAMPLING_RATE = 10  # 10 kHz sampling rate (1 sample = 0.1 ms)

# Function to detect spikes
def detect_spikes(voltage_trace, threshold=SPIKE_THRESHOLD):
    """
    Detects spikes in a voltage trace using a threshold-crossing method.

    Returns:
        spike_indices (list): Indices where spikes occur.
        spike_peaks (list): Corresponding peak voltages.
    """
    peaks, properties = find_peaks(voltage_trace, height=threshold)
    return peaks, properties["peak_heights"]

def compute_spike_threshold(voltage_trace, spike_indices, sampling_rate=10, dvdt_threshold=10):
    """
    Computes the spike threshold as the first point where dV/dt ≥ dvdt_threshold before a spike.

    Returns:
        threshold_values (list): List of refined threshold voltages.
    """
    threshold_values = []
    dvdt = np.zeros_like(voltage_trace)
    
    # Compute dV/dt using central difference method
    dvdt[1:-1] = (voltage_trace[2:] - voltage_trace[:-2]) / (2 * (1 / sampling_rate))

    for idx in spike_indices:
        search_range = min(30, idx)  # Limit search window
        pre_spike_idx = idx - search_range

        while pre_spike_idx > 0 and dvdt[pre_spike_idx] < dvdt_threshold:
            pre_spike_idx += 1  # Move forward to find first valid dV/dt

        # Ensure we did not exit too early
        if pre_spike_idx >= idx - 5:
            pre_spike_idx = idx - 10  # Default backup if threshold search fails

        threshold_values.append(voltage_trace[pre_spike_idx])

    return threshold_values


# Function to compute spike frequency
def compute_spike_frequency(spike_indices, total_time_ms=200):
    """
    Computes spike frequency (Hz) based on detected spikes.
    """
    return len(spike_indices) / (total_time_ms / 1000) if spike_indices.size > 0 else 0


# Function to plot voltage trace with detected thresholds
def plot_voltage_trace(voltage_trace, spike_indices, threshold_values):
    """
    Plots the voltage trace with detected spikes and thresholds.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(voltage_trace, label="Voltage Trace", color="black", alpha=0.6)

    # Mark detected thresholds
    threshold_indices = [spike_indices[i] for i in range(len(spike_indices)) if threshold_values[i] is not None]
    threshold_voltages = [threshold_values[i] for i in range(len(spike_indices)) if threshold_values[i] is not None]

    plt.scatter(threshold_indices, threshold_voltages, color="red", marker="o", label="Thresholds")
    plt.scatter(spike_indices, voltage_trace[spike_indices], color="blue", marker="x", label="Spikes")

    plt.xlabel("Time (Samples)")
    plt.ylabel("Voltage (mV)")
    plt.title("Voltage Trace with Detected Thresholds")
    plt.legend()
    
    # Ensure the plot displays properly on your system
    plt.show(block=True)


# Function to compute spike amplitude
def compute_spike_amplitude(spike_peaks, threshold_values):
    """
    Computes spike amplitude for each detected spike.

    Parameters:
        spike_peaks (list): Peak voltages of detected spikes.
        threshold_values (list): Threshold voltages of detected spikes.

    Returns:
        amplitudes (list): Spike amplitude for each spike.
    """
    amplitudes = []
    for peak, threshold in zip(spike_peaks, threshold_values):
        if threshold is not None:  # Ensure valid threshold values
            amplitudes.append(peak - threshold)
        else:
            amplitudes.append(None)  # If threshold was not detected properly

    return amplitudes


# Function to compute Time-to-Peak Action Potential (TTPAP)
def compute_ttpap(voltage_trace, spike_indices, threshold_values, sampling_rate=SAMPLING_RATE):
    """
    Computes Time-to-Peak Action Potential (TTPAP) for each detected spike.

    Parameters:
        voltage_trace (np.array): Voltage values.
        spike_indices (list): Indices of detected spikes.
        threshold_values (list): Threshold voltages for each spike.
        sampling_rate (int): Sampling rate in kHz (default: 10 kHz).

    Returns:
        ttpap_values (list): Time (ms) from threshold crossing to peak.
    """
    ttpap_values = []

    for spike_idx, threshold in zip(spike_indices, threshold_values):
        if threshold is not None:
            # Find the first threshold crossing *before* the spike peak
            threshold_idx = None
            for i in range(spike_idx, 0, -1):  # Scan backwards from spike peak
                if voltage_trace[i] <= threshold:
                    threshold_idx = i
                    break

            if threshold_idx is not None:
                # Compute TTPAP as the time from threshold to peak
                ttpap = (spike_idx - threshold_idx) / sampling_rate
                ttpap_values.append(ttpap)
            else:
                ttpap_values.append(None)
        else:
            ttpap_values.append(None)

    return ttpap_values

###############################################
# Function to compute Time-to-Peak Action Potential (TTPAP) with debug output
# def compute_ttpap_debug(spike_indices, threshold_values, voltage_trace, sampling_rate=SAMPLING_RATE):
#     """
#     Computes time-to-peak action potential (TTPAP) for each detected spike with debugging.

#     Returns:
#         ttpap_values (list): Time (ms) from threshold to peak for each spike.
#     """
#     ttpap_values = []
    
#     for i, (spike_idx, threshold) in enumerate(zip(spike_indices, threshold_values)):
#         if threshold is not None:
#             # Find threshold crossing index
#             threshold_idx = np.where(voltage_trace[:spike_idx] <= threshold)[0]
#             if len(threshold_idx) > 0:
#                 threshold_idx = threshold_idx[-1]  # Last occurrence before peak

#                 # Compute time difference in milliseconds
#                 ttpap = (spike_idx - threshold_idx) / sampling_rate
#                 ttpap_values.append(ttpap)

#                 # Print debug values for the first few spikes
#                 if i < 5:
#                     print(f"Spike {i+1}:")
#                     print(f"  - Threshold Index: {threshold_idx}")
#                     print(f"  - Spike Peak Index: {spike_idx}")
#                     print(f"  - Computed TTPAP: {ttpap} ms")
#                     print("-" * 40)
#             else:
#                 ttpap_values.append(None)
#         else:
#             ttpap_values.append(None)

#     return ttpap_values

# # Compute TTPAP with debugging
# spike_features["ttpap_values"] = compute_ttpap_debug(
#     spike_features["spike_indices"],
#     spike_features["threshold_values"],
#     voltage_trace
# )
###############################################

# Function to compute Delay to First Spike (DTFS)
def compute_delay_to_first_spike(spike_indices, stimulus_start_idx=2000, sampling_rate=SAMPLING_RATE):
    """
    Computes the delay from stimulus onset (200 ms) to the first detected spike.

    Parameters:
        spike_indices (list): Indices of detected spikes.
        stimulus_start_idx (int): Index corresponding to 200 ms (default: 2000 for 10 kHz sampling rate).
        sampling_rate (int): Sampling rate in kHz (default: 10 kHz).

    Returns:
        float: Time (ms) to first spike or None if no spike detected.
    """
    if len(spike_indices) == 0:
        return None  # No spikes detected

    first_spike_idx = spike_indices[0]  # Get first spike occurrence

    if first_spike_idx >= stimulus_start_idx:
        return (first_spike_idx - stimulus_start_idx) / sampling_rate  # Convert to ms
    else:
        return None  # Spike occurred before stimulus start (unexpected case)

# Function to compute Spike Width at Half-Max Amplitude
def compute_spike_width(voltage_trace, spike_indices, threshold_values, sampling_rate=SAMPLING_RATE):
    """
    Computes spike width at half-max amplitude for each detected spike.

    Parameters:
        voltage_trace (np.array): Voltage values.
        spike_indices (list): Indices of detected spikes.
        threshold_values (list): Threshold voltages of detected spikes.
        sampling_rate (int): Sampling rate in kHz (default: 10 kHz).

    Returns:
        spike_widths (list): Width (ms) for each spike.
    """
    spike_widths = []

    for spike_idx, threshold in zip(spike_indices, threshold_values):
        if threshold is not None:
            peak_voltage = voltage_trace[spike_idx]  # Get the peak voltage
            half_max = (peak_voltage + threshold) / 2  # Compute half-max amplitude

            # Find the first crossing (upstroke) before the peak
            left_idx = None
            for i in range(spike_idx, 0, -1):
                if voltage_trace[i] <= half_max:
                    left_idx = i
                    break

            # Find the second crossing (downstroke) after the peak
            right_idx = None
            for i in range(spike_idx, len(voltage_trace)):
                if voltage_trace[i] <= half_max:
                    right_idx = i
                    break

            # Compute spike width if valid indices are found
            if left_idx is not None and right_idx is not None:
                spike_width = (right_idx - left_idx) / sampling_rate  # Convert to ms
                spike_widths.append(spike_width)
            else:
                spike_widths.append(None)  # If invalid, store None
        else:
            spike_widths.append(None)

    return spike_widths

# Function to compute Afterhyperpolarization (AHP) without the last spike
def compute_ahp_adaptive(voltage_trace, spike_indices):
    """
    Computes Afterhyperpolarization (AHP) values using interspike intervals, 
    but does NOT compute AHP for the last spike.

    Parameters:
        voltage_trace (np.array): Voltage values.
        spike_indices (list): Indices of detected spikes.

    Returns:
        ahp_values (list): Minimum voltage between consecutive spikes (AHP). 
                           The last spike's AHP is set to None.
    """
    ahp_values = []

    for i in range(len(spike_indices) - 1):  # Exclude last spike
        search_start = spike_indices[i]
        search_end = spike_indices[i + 1]

        # Find the minimum voltage in this range
        ahp_values.append(np.min(voltage_trace[search_start:search_end]))

    return ahp_values

# Function to compute Time-to-Peak AHP (TTPAHP) correctly from threshold crossing
def compute_ttpath(voltage_trace, spike_indices, threshold_values, ahp_values, sampling_rate=10):
    """
    Computes Time-to-Peak Afterhyperpolarization (TTPAHP) for all spikes that have an AHP value.

    Parameters:
        voltage_trace (np.array): Voltage values.
        spike_indices (list): Indices of detected spikes.
        threshold_values (list): Threshold voltages for each spike.
        ahp_values (list): AHP voltage values.
        sampling_rate (int): Sampling rate in kHz (default: 10 kHz).

    Returns:
        ttpath_values (list): Time (ms) from threshold to AHP minimum, matching AHP count.
    """
    ttpath_values = []

    for i in range(len(ahp_values)):  # Ensure we iterate over all AHP values
        if threshold_values[i] is not None and ahp_values[i] is not None:
            threshold_idx = None

            # Find threshold crossing before the spike peak
            for t in range(spike_indices[i], 0, -1):
                if voltage_trace[t] <= threshold_values[i]:
                    threshold_idx = t
                    break

            if threshold_idx is not None:
                # Find the AHP minimum after threshold crossing (within valid range)
                search_start = threshold_idx
                search_end = spike_indices[i+1] if i + 1 < len(spike_indices) else len(voltage_trace) - 1
                ahp_idx = search_start + np.argmin(voltage_trace[search_start:search_end])

                # Compute TTPAHP as time from threshold to AHP min
                ttpath_values.append((ahp_idx - threshold_idx) / sampling_rate)
            else:
                ttpath_values.append(None)
        else:
            ttpath_values.append(None)

    return ttpath_values

# Re-run feature extraction with the corrected TTPAHP function
def extract_spike_features(voltage_trace, sampling_rate=10, dvdt_threshold=10):
    spike_indices, spike_peaks = detect_spikes(voltage_trace)
    spike_frequency = compute_spike_frequency(spike_indices)
    threshold_values_refined = compute_spike_threshold(voltage_trace, spike_indices, sampling_rate, dvdt_threshold)
    spike_amplitudes = compute_spike_amplitude(spike_peaks, threshold_values_refined)
    ttpap_values = compute_ttpap(voltage_trace, spike_indices, threshold_values_refined, sampling_rate)
    delay_to_first_spike = compute_delay_to_first_spike(spike_indices)
    spike_widths = compute_spike_width(voltage_trace, spike_indices, threshold_values_refined, sampling_rate)
    ahp_values = compute_ahp_adaptive(voltage_trace, spike_indices)
    ttpath_values = compute_ttpath(voltage_trace, spike_indices, threshold_values_refined, ahp_values, sampling_rate)

    return {
        "spike_indices": spike_indices,
        "spike_peaks": spike_peaks,
        "spike_frequency": spike_frequency,
        "threshold_values": threshold_values_refined,
        "spike_amplitudes": spike_amplitudes,
        "ttpap_values": ttpap_values,
        "delay_to_first_spike": delay_to_first_spike,
        "spike_widths": spike_widths,
        "ahp_values": ahp_values,
        "ttpahp_values": ttpath_values  
    }

# Select a sample neuron and process it
test_neuron = list(control_data.values())[3]  # Pick a sample control neuron
voltage_trace = test_neuron["Voltage (mV)"].values

# Extract spike features
spike_features = extract_spike_features(voltage_trace)

# Display results
print(f"Spike Frequency: {spike_features['spike_frequency']} Hz")
print(f"Threshold Values: {spike_features['threshold_values'][:]}")
print(f"Amplitude Values: {spike_features['spike_amplitudes'][:]}")
print(f"TTPAP Values: {spike_features['ttpap_values'][:]}")
print(f"Delay to First Spike: {spike_features['delay_to_first_spike']} ms")
print(f"Spike Width Values: {spike_features['spike_widths'][:]}")
print(f"After Hyperpolarization Values: {spike_features['ahp_values'][:]}")
print(f"TTPAHP values: {spike_features['ttpahp_values'][:]}")


# Plot the results
plot_voltage_trace(voltage_trace, spike_features["spike_indices"], spike_features["threshold_values"])


# Function to export spike features for all neurons
def export_spike_features(data_dict, filename="spike_features.csv"):
    all_features = []
    for neuron_name, df in data_dict.items():
        voltage_trace = df["Voltage (mV)"].values
        spike_features = extract_spike_features(voltage_trace)
        num_spikes = len(spike_features["spike_indices"])

        for i in range(num_spikes):
            all_features.append({
                "Neuron": neuron_name,
                "Spike Index": spike_features["spike_indices"][i],
                "Spike Peak (mV)": spike_features["spike_peaks"][i],
                "Threshold (mV)": spike_features["threshold_values"][i],
                "Spike Amplitude (mV)": spike_features["spike_amplitudes"][i],
                "Spike Frequency (Hz)": spike_features["spike_frequency"],  
                "TTPAP (ms)": spike_features["ttpap_values"][i],
                "Delay to First Spike (ms)": spike_features["delay_to_first_spike"] if i == 0 else None,
                "Spike Width (ms)": spike_features["spike_widths"][i],
                "AHP (mV)": spike_features["ahp_values"][i] if i < len(spike_features["ahp_values"]) else None,
                "TTPAHP (ms)": spike_features["ttpahp_values"][i] if i < len(spike_features["ttpahp_values"]) else None
            })

    df_spike_features = pd.DataFrame(all_features)
    df_spike_features.to_csv(filename, index=False)

# Export features for all neurons
export_spike_features({**control_data, **stuttering_data})  

