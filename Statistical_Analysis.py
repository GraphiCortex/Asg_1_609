import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
spike_features = pd.read_csv("spike_features.csv")

# Create a "Group" column based on file names
spike_features["Group"] = spike_features["Neuron"].apply(lambda x: "Control" if "Ctrl" in x else "Stuttering")

# Define function for statistical analysis
def analyze_feature(feature):
    control_data = spike_features[spike_features["Group"] == "Control"][feature].dropna()
    stuttering_data = spike_features[spike_features["Group"] == "Stuttering"][feature].dropna()

    # Perform statistical tests
    t_stat, t_pval = ttest_ind(control_data, stuttering_data, equal_var=False)  # Welch's t-test
    u_stat, u_pval = mannwhitneyu(control_data, stuttering_data, alternative="two-sided")

    # Store statistical results
    stats_df = pd.DataFrame({
        "Feature": [feature],
        "T-Test Statistic": [t_stat],
        "T-Test p-value": [t_pval],
        "Mann-Whitney U Statistic": [u_stat],
        "Mann-Whitney p-value": [u_pval]
    })

    print(f"\n### {feature} Statistical Analysis ###")
    print(stats_df)

    return stats_df

# Define function for plotting
def plot_feature(feature):
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

feature_name = "Spike Peak (mV)"
spike_freq_stats = analyze_feature(feature_name)
plot_feature(feature_name)

feature_name = "Threshold (mV)"
spike_freq_stats = analyze_feature(feature_name)
plot_feature(feature_name)

feature_name = "Spike Amplitude (mV)"
spike_freq_stats = analyze_feature(feature_name)
plot_feature(feature_name)

feature_name = "Spike Frequency (Hz)"
spike_freq_stats = analyze_feature(feature_name)
plot_feature(feature_name)

feature_name = "TTPAP (ms)"
spike_freq_stats = analyze_feature(feature_name)
plot_feature(feature_name)

feature_name = "Delay to First Spike (ms)"
spike_freq_stats = analyze_feature(feature_name)
plot_feature(feature_name)

feature_name = "Spike Width (ms)"
spike_freq_stats = analyze_feature(feature_name)
plot_feature(feature_name)

feature_name = "AHP (mV)"
spike_freq_stats = analyze_feature(feature_name)
plot_feature(feature_name)

feature_name = "TTPAHP (ms)"
spike_freq_stats = analyze_feature(feature_name)
plot_feature(feature_name)
