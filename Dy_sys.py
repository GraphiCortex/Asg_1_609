import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Re-load the spike features dataset
file_path = "spike_features.csv"
spike_features = pd.read_csv(file_path)

# Re-load raw voltage traces for phase plots
control_files = ["Data/Control/N1_Ctrl.txt", "Data/Control/N2_Ctrl.txt", "Data/control/N3_Ctrl.txt", "Data/Control/N4_Ctrl.txt", "Data/Control/N5_Ctrl.txt"]
stuttering_files = ["Data/Stuttering/N1_ST.txt", "Data/Stuttering/N2_ST.txt", "Data/Stuttering/N3_ST.txt", "Data/Stuttering/N4_ST.txt", "Data/Stuttering/N5_ST.txt"]

# Function to load voltage data
def load_voltage_data(file_path):
    data = np.loadtxt(file_path)
    time = np.arange(len(data)) * 0.1  # Assuming a 10 kHz sampling rate (0.1 ms per step)
    voltage = data[:, 1]  # Second column contains voltage
    return time, voltage

# Compute dV/dt for phase plot
def compute_dv_dt(time, voltage):
    dv_dt = np.gradient(voltage, time)
    return dv_dt

# Create phase plots
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# Plot control neurons
for file in control_files:
    time, voltage = load_voltage_data(file)
    dv_dt = compute_dv_dt(time, voltage)
    axes[0].plot(voltage, dv_dt, label=file.split('/')[-1])

axes[0].set_title("Phase Plot - Control Neurons")
axes[0].set_xlabel("Membrane Voltage (mV)")
axes[0].set_ylabel("dV/dt (mV/ms)")
axes[0].legend()

# Plot stuttering neurons
for file in stuttering_files:
    time, voltage = load_voltage_data(file)
    dv_dt = compute_dv_dt(time, voltage)
    axes[1].plot(voltage, dv_dt, label=file.split('/')[-1])

axes[1].set_title("Phase Plot - Stuttering Neurons")
axes[1].set_xlabel("Membrane Voltage (mV)")
axes[1].legend()

# Improve plot layout
plt.tight_layout()
plt.show()

# Function to compute peak dV/dt and variability
def analyze_phase_trajectory(files):
    peak_dv_dt = []
    loop_widths = []
    
    for file in files:
        time, voltage = load_voltage_data(file)
        dv_dt = compute_dv_dt(time, voltage)
        
        # Compute peak dV/dt (max absolute value)
        peak_dv_dt.append(np.max(np.abs(dv_dt)))
        
        # Compute loop width (difference between max and min voltage in the loop)
        loop_widths.append(np.max(voltage) - np.min(voltage))
    
    return np.array(peak_dv_dt), np.array(loop_widths)

# Analyze control and stuttering neurons
control_peak_dv_dt, control_loop_widths = analyze_phase_trajectory(control_files)
stuttering_peak_dv_dt, stuttering_loop_widths = analyze_phase_trajectory(stuttering_files)

# Compute statistics
control_stats = {
    "Peak dV/dt (mean ± std)": (np.mean(control_peak_dv_dt), np.std(control_peak_dv_dt)),
    "Loop Width (mean ± std)": (np.mean(control_loop_widths), np.std(control_loop_widths)),
}

stuttering_stats = {
    "Peak dV/dt (mean ± std)": (np.mean(stuttering_peak_dv_dt), np.std(stuttering_peak_dv_dt)),
    "Loop Width (mean ± std)": (np.mean(stuttering_loop_widths), np.std(stuttering_loop_widths)),
}

# Display results
import pandas as pd
stats_df = pd.DataFrame([control_stats, stuttering_stats], index=["Control", "Stuttering"])

print(stats_df)

from scipy.stats import ttest_ind, mannwhitneyu

# Perform statistical tests
stats_results = {
    "Peak dV/dt (t-test p-value)": ttest_ind(control_peak_dv_dt, stuttering_peak_dv_dt, equal_var=False).pvalue,
    "Peak dV/dt (Mann-Whitney p-value)": mannwhitneyu(control_peak_dv_dt, stuttering_peak_dv_dt, alternative="two-sided").pvalue,
    "Loop Width (t-test p-value)": ttest_ind(control_loop_widths, stuttering_loop_widths, equal_var=False).pvalue,
    "Loop Width (Mann-Whitney p-value)": mannwhitneyu(control_loop_widths, stuttering_loop_widths, alternative="two-sided").pvalue,
}

# Convert results to DataFrame
stats_df = pd.DataFrame(stats_results, index=["Statistical Test Results"])

print(stats_df)


# Function to compute the average trajectory and variability for phase plots
def compute_avg_trajectory(files):
    voltage_list = []
    dv_dt_list = []
    
    for file in files:
        time, voltage = load_voltage_data(file)
        dv_dt = compute_dv_dt(time, voltage)
        voltage_list.append(voltage)
        dv_dt_list.append(dv_dt)
    
    # Interpolate to ensure all trajectories have the same length for averaging
    min_length = min(len(v) for v in voltage_list)
    voltage_matrix = np.array([v[:min_length] for v in voltage_list])
    dv_dt_matrix = np.array([d[:min_length] for d in dv_dt_list])
    
    # Compute mean and standard deviation
    avg_voltage = np.mean(voltage_matrix, axis=0)
    avg_dv_dt = np.mean(dv_dt_matrix, axis=0)
    std_dv_dt = np.std(dv_dt_matrix, axis=0)
    
    return avg_voltage, avg_dv_dt, std_dv_dt

# Compute average trajectories
control_avg_v, control_avg_dv_dt, control_std_dv_dt = compute_avg_trajectory(control_files)
stuttering_avg_v, stuttering_avg_dv_dt, stuttering_std_dv_dt = compute_avg_trajectory(stuttering_files)

# Plot with shaded variability
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# Control group
ax[0].plot(control_avg_v, control_avg_dv_dt, color='blue', label="Control Avg Trajectory")
ax[0].fill_between(control_avg_v, control_avg_dv_dt - control_std_dv_dt, control_avg_dv_dt + control_std_dv_dt, color='blue', alpha=0.2)
ax[0].scatter(control_avg_v[np.argmax(control_avg_dv_dt)], np.max(control_avg_dv_dt), color='black', marker='x', label="Peak dV/dt")
ax[0].set_title("Phase Plot - Control Neurons")
ax[0].set_xlabel("Membrane Voltage (mV)")
ax[0].set_ylabel("dV/dt (mV/ms)")
ax[0].legend()

# Stuttering group
ax[1].plot(stuttering_avg_v, stuttering_avg_dv_dt, color='red', label="Stuttering Avg Trajectory")
ax[1].fill_between(stuttering_avg_v, stuttering_avg_dv_dt - stuttering_std_dv_dt, stuttering_avg_dv_dt + stuttering_std_dv_dt, color='red', alpha=0.2)
ax[1].scatter(stuttering_avg_v[np.argmax(stuttering_avg_dv_dt)], np.max(stuttering_avg_dv_dt), color='black', marker='x', label="Peak dV/dt")
ax[1].set_title("Phase Plot - Stuttering Neurons")
ax[1].set_xlabel("Membrane Voltage (mV)")
ax[1].legend()

# Improve layout
plt.tight_layout()
plt.show()

# Compute d²V/dt² for acceleration phase plot
def compute_d2v_dt2(time, voltage):
    dv_dt = compute_dv_dt(time, voltage)  # First derivative
    d2v_dt2 = np.gradient(dv_dt, time)  # Second derivative
    return d2v_dt2

# Function to compute the average trajectory and variability for acceleration phase plots
def compute_avg_acceleration_trajectory(files):
    voltage_list = []
    d2v_dt2_list = []
    
    for file in files:
        time, voltage = load_voltage_data(file)
        d2v_dt2 = compute_d2v_dt2(time, voltage)
        voltage_list.append(voltage)
        d2v_dt2_list.append(d2v_dt2)
    
    # Interpolate to ensure all trajectories have the same length for averaging
    min_length = min(len(v) for v in voltage_list)
    voltage_matrix = np.array([v[:min_length] for v in voltage_list])
    d2v_dt2_matrix = np.array([d[:min_length] for d in d2v_dt2_list])
    
    # Compute mean and standard deviation
    avg_voltage = np.mean(voltage_matrix, axis=0)
    avg_d2v_dt2 = np.mean(d2v_dt2_matrix, axis=0)
    std_d2v_dt2 = np.std(d2v_dt2_matrix, axis=0)
    
    return avg_voltage, avg_d2v_dt2, std_d2v_dt2

# Compute average acceleration trajectories
control_avg_v, control_avg_d2v_dt2, control_std_d2v_dt2 = compute_avg_acceleration_trajectory(control_files)
stuttering_avg_v, stuttering_avg_d2v_dt2, stuttering_std_d2v_dt2 = compute_avg_acceleration_trajectory(stuttering_files)

# Plot acceleration phase plots with shaded variability
fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# Control group
ax[0].plot(control_avg_v, control_avg_d2v_dt2, color='blue', label="Control Avg Acceleration")
ax[0].fill_between(control_avg_v, 
                   control_avg_d2v_dt2 - control_std_d2v_dt2, 
                   control_avg_d2v_dt2 + control_std_d2v_dt2, 
                   color='blue', alpha=0.2)
ax[0].scatter(control_avg_v[np.argmax(control_avg_d2v_dt2)], np.max(control_avg_d2v_dt2), color='black', marker='x', label="Peak d²V/dt²")
ax[0].scatter(control_avg_v[np.argmin(control_avg_d2v_dt2)], np.min(control_avg_d2v_dt2), color='green', marker='o', label="min d²V/dt²")
ax[0].set_title("Acceleration Phase Plot - Control Neurons")
ax[0].set_xlabel("Membrane Voltage (mV)")
ax[0].set_ylabel("d²V/dt² (mV/ms²)")
ax[0].legend()

# Stuttering group
ax[1].plot(stuttering_avg_v, stuttering_avg_d2v_dt2, color='red', label="Stuttering Avg Acceleration")
ax[1].fill_between(stuttering_avg_v, 
                   stuttering_avg_d2v_dt2 - stuttering_std_d2v_dt2, 
                   stuttering_avg_d2v_dt2 + stuttering_std_d2v_dt2, 
                   color='red', alpha=0.2)
ax[1].scatter(stuttering_avg_v[np.argmax(stuttering_avg_d2v_dt2)], np.max(stuttering_avg_d2v_dt2), color='black', marker='x', label="Peak d²V/dt²")
ax[1].scatter(stuttering_avg_v[np.argmin(stuttering_avg_d2v_dt2)], np.min(stuttering_avg_d2v_dt2), color='green', marker='o', label="min d²V/dt²")
ax[1].set_title("Acceleration Phase Plot - Stuttering Neurons")
ax[1].set_xlabel("Membrane Voltage (mV)")
ax[1].legend()

# Improve layout
plt.tight_layout()
plt.show()
