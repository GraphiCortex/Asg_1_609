import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
spike_features = pd.read_csv("spike_features.csv")

# Create a "Group" column based on file names
spike_features["Group"] = spike_features["Neuron"].apply(lambda x: "Control" if "Ctrl" in x else "Stuttering")

# Define the feature to analyze
feature = "Spike Peak (mV)"

# Extract data for both groups
control_data = spike_features[spike_features["Group"] == "Control"][feature].dropna()
stuttering_data = spike_features[spike_features["Group"] == "Stuttering"][feature].dropna()

# Perform statistical tests
t_stat, t_pval = ttest_ind(control_data, stuttering_data, equal_var=False)  # Welch's t-test
u_stat, u_pval = mannwhitneyu(control_data, stuttering_data, alternative="two-sided")

# Store statistical results in a DataFrame
spike_peak_stats = pd.DataFrame({
    "T-Test Statistic": [t_stat],
    "T-Test p-value": [t_pval],
    "Mann-Whitney U Statistic": [u_stat],
    "Mann-Whitney p-value": [u_pval]
})

# Print the results
print("Spike Peak (mV)")
print(spike_peak_stats)

# Define the feature to visualize
feature = "Spike Peak (mV)"

# Generate boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=spike_features, x="Group", y=feature, palette=["blue", "red"])
plt.title(f"Comparison of {feature} in Control vs Stuttering Neurons")
plt.xlabel("Group")
plt.ylabel(feature)
plt.show()

# Generate violin plot
plt.figure(figsize=(8, 5))
sns.violinplot(data=spike_features, x="Group", y=feature, palette=["blue", "red"], inner="quartile")
plt.title(f"Distribution of {feature} in Control vs Stuttering Neurons")
plt.xlabel("Group")
plt.ylabel(feature)
plt.show()


# Define the feature we are analyzing
feature = "Threshold (mV)"

# Extract data for both groups
control_data = spike_features[spike_features["Group"] == "Control"][feature].dropna()
stuttering_data = spike_features[spike_features["Group"] == "Stuttering"][feature].dropna()

# Perform statistical tests
t_stat, t_pval = ttest_ind(control_data, stuttering_data, equal_var=False)  # Welch's t-test
u_stat, u_pval = mannwhitneyu(control_data, stuttering_data, alternative="two-sided")

# Store statistical results
threshold_stats = pd.DataFrame({
    "T-Test Statistic": [t_stat],
    "T-Test p-value": [t_pval],
    "Mann-Whitney U Statistic": [u_stat],
    "Mann-Whitney p-value": [u_pval]
})

print("Threshold (mV)")
print(threshold_stats)

# Visualize with boxplot and violin plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=spike_features, x="Group", y=feature, palette=["blue", "red"])
plt.title(f"Comparison of {feature} in Control vs Stuttering Neurons")
plt.xlabel("Group")
plt.ylabel(feature)
plt.show()

plt.figure(figsize=(8, 5))
sns.violinplot(data=spike_features, x="Group", y=feature, palette=["blue", "red"], inner="quartile")
plt.title(f"Distribution of {feature} in Control vs Stuttering Neurons")
plt.xlabel("Group")
plt.ylabel(feature)
plt.show()


# Define the feature we are analyzing
feature = "Spike Amplitude (mV)"

# Extract data for both groups
control_data = spike_features[spike_features["Group"] == "Control"][feature].dropna()
stuttering_data = spike_features[spike_features["Group"] == "Stuttering"][feature].dropna()

# Perform statistical tests
t_stat, t_pval = ttest_ind(control_data, stuttering_data, equal_var=False)  # Welch's t-test
u_stat, u_pval = mannwhitneyu(control_data, stuttering_data, alternative="two-sided")

# Store statistical results
spike_amplitude_stats = pd.DataFrame({
    "T-Test Statistic": [t_stat],
    "T-Test p-value": [t_pval],
    "Mann-Whitney U Statistic": [u_stat],
    "Mann-Whitney p-value": [u_pval]
})

print("Spike Amplitude (mV)")
print(spike_amplitude_stats)

# Visualize with boxplot and violin plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=spike_features, x="Group", y=feature, palette=["blue", "red"])
plt.title(f"Comparison of {feature} in Control vs Stuttering Neurons")
plt.xlabel("Group")
plt.ylabel(feature)
plt.show()

plt.figure(figsize=(8, 5))
sns.violinplot(data=spike_features, x="Group", y=feature, palette=["blue", "red"], inner="quartile")
plt.title(f"Distribution of {feature} in Control vs Stuttering Neurons")
plt.xlabel("Group")
plt.ylabel(feature)
plt.show()


# Define the feature we are analyzing
feature = "TTPAP (ms)"

# Extract data for both groups
control_data = spike_features[spike_features["Group"] == "Control"][feature].dropna()
stuttering_data = spike_features[spike_features["Group"] == "Stuttering"][feature].dropna()

# Perform statistical tests
t_stat, t_pval = ttest_ind(control_data, stuttering_data, equal_var=False)  # Welch's t-test
u_stat, u_pval = mannwhitneyu(control_data, stuttering_data, alternative="two-sided")

# Store statistical results
ttpap_stats = pd.DataFrame({
    "T-Test Statistic": [t_stat],
    "T-Test p-value": [t_pval],
    "Mann-Whitney U Statistic": [u_stat],
    "Mann-Whitney p-value": [u_pval]
})

print("TTPAP (ms)")
print(ttpap_stats)

# Visualize with boxplot and violin plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=spike_features, x="Group", y=feature, palette=["blue", "red"])
plt.title(f"Comparison of {feature} in Control vs Stuttering Neurons")
plt.xlabel("Group")
plt.ylabel(feature)
plt.show()

plt.figure(figsize=(8, 5))
sns.violinplot(data=spike_features, x="Group", y=feature, palette=["blue", "red"], inner="quartile")
plt.title(f"Distribution of {feature} in Control vs Stuttering Neurons")
plt.xlabel("Group")
plt.ylabel(feature)
plt.show()


# Define the feature we are analyzing
feature = "Delay to First Spike (ms)"

# Extract data for both groups
control_data = spike_features[spike_features["Group"] == "Control"][feature].dropna()
stuttering_data = spike_features[spike_features["Group"] == "Stuttering"][feature].dropna()

# Perform statistical tests
t_stat, t_pval = ttest_ind(control_data, stuttering_data, equal_var=False)  # Welch's t-test
u_stat, u_pval = mannwhitneyu(control_data, stuttering_data, alternative="two-sided")

# Store statistical results
delay_first_spike_stats = pd.DataFrame({
    "T-Test Statistic": [t_stat],
    "T-Test p-value": [t_pval],
    "Mann-Whitney U Statistic": [u_stat],
    "Mann-Whitney p-value": [u_pval]
})

print("Delay to first Spike (ms)")
print(delay_first_spike_stats)

# Visualize with boxplot and violin plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=spike_features, x="Group", y=feature, palette=["blue", "red"])
plt.title(f"Comparison of {feature} in Control vs Stuttering Neurons")
plt.xlabel("Group")
plt.ylabel(feature)
plt.show()

plt.figure(figsize=(8, 5))
sns.violinplot(data=spike_features, x="Group", y=feature, palette=["blue", "red"], inner="quartile")
plt.title(f"Distribution of {feature} in Control vs Stuttering Neurons")
plt.xlabel("Group")
plt.ylabel(feature)
plt.show()


# Define the feature we are analyzing
feature = "Spike Width (ms)"

# Extract data for both groups
control_data = spike_features[spike_features["Group"] == "Control"][feature].dropna()
stuttering_data = spike_features[spike_features["Group"] == "Stuttering"][feature].dropna()

# Perform statistical tests
t_stat, t_pval = ttest_ind(control_data, stuttering_data, equal_var=False)  # Welch's t-test
u_stat, u_pval = mannwhitneyu(control_data, stuttering_data, alternative="two-sided")

# Store statistical results
spike_width_stats = pd.DataFrame({
    "T-Test Statistic": [t_stat],
    "T-Test p-value": [t_pval],
    "Mann-Whitney U Statistic": [u_stat],
    "Mann-Whitney p-value": [u_pval]
})

print("Spike Width (ms)")
print(delay_first_spike_stats)

# Visualize with boxplot and violin plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=spike_features, x="Group", y=feature, palette=["blue", "red"])
plt.title(f"Comparison of {feature} in Control vs Stuttering Neurons")
plt.xlabel("Group")
plt.ylabel(feature)
plt.show()

plt.figure(figsize=(8, 5))
sns.violinplot(data=spike_features, x="Group", y=feature, palette=["blue", "red"], inner="quartile")
plt.title(f"Distribution of {feature} in Control vs Stuttering Neurons")
plt.xlabel("Group")
plt.ylabel(feature)
plt.show()

# Define the feature we are analyzing
feature = "AHP (mV)"

# Extract data for both groups
control_data = spike_features[spike_features["Group"] == "Control"][feature].dropna()
stuttering_data = spike_features[spike_features["Group"] == "Stuttering"][feature].dropna()

# Perform statistical tests
t_stat, t_pval = ttest_ind(control_data, stuttering_data, equal_var=False)  # Welch's t-test
u_stat, u_pval = mannwhitneyu(control_data, stuttering_data, alternative="two-sided")

# Store statistical results
ahp_stats = pd.DataFrame({
    "T-Test Statistic": [t_stat],
    "T-Test p-value": [t_pval],
    "Mann-Whitney U Statistic": [u_stat],
    "Mann-Whitney p-value": [u_pval]
})

print("AHP (mV)")
print(ahp_stats)

# Visualize with boxplot and violin plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=spike_features, x="Group", y=feature, palette=["blue", "red"])
plt.title(f"Comparison of {feature} in Control vs Stuttering Neurons")
plt.xlabel("Group")
plt.ylabel(feature)
plt.show()

plt.figure(figsize=(8, 5))
sns.violinplot(data=spike_features, x="Group", y=feature, palette=["blue", "red"], inner="quartile")
plt.title(f"Distribution of {feature} in Control vs Stuttering Neurons")
plt.xlabel("Group")
plt.ylabel(feature)
plt.show()



# Define the feature we are analyzing
feature = "TTPAHP (ms)"

# Extract data for both groups
control_data = spike_features[spike_features["Group"] == "Control"][feature].dropna()
stuttering_data = spike_features[spike_features["Group"] == "Stuttering"][feature].dropna()

# Perform statistical tests
t_stat, t_pval = ttest_ind(control_data, stuttering_data, equal_var=False)  # Welch's t-test
u_stat, u_pval = mannwhitneyu(control_data, stuttering_data, alternative="two-sided")

# Store statistical results
ttpahp_stats = pd.DataFrame({
    "T-Test Statistic": [t_stat],
    "T-Test p-value": [t_pval],
    "Mann-Whitney U Statistic": [u_stat],
    "Mann-Whitney p-value": [u_pval]
})

print("TTPAHP (ms)")
print(ttpahp_stats)

# Visualize with boxplot and violin plot
plt.figure(figsize=(8, 5))
sns.boxplot(data=spike_features, x="Group", y=feature, palette=["blue", "red"])
plt.title(f"Comparison of {feature} in Control vs Stuttering Neurons")
plt.xlabel("Group")
plt.ylabel(feature)
plt.show()

plt.figure(figsize=(8, 5))
sns.violinplot(data=spike_features, x="Group", y=feature, palette=["blue", "red"], inner="quartile")
plt.title(f"Distribution of {feature} in Control vs Stuttering Neurons")
plt.xlabel("Group")
plt.ylabel(feature)
plt.show()
