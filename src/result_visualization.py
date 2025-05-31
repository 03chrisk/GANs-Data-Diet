import pandas as pd
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_mean_sem_fid(results_df):
    mean_fid = results_df.groupby('percentage')['fid_score'].mean()
    sem_fid = results_df.groupby('percentage')['fid_score'].apply(sem)
    return mean_fid, sem_fid

results_random = pd.read_csv('results/cifar10_random_20250523_232825/results.csv')
results_easiest = pd.read_csv('results/cifar10_easiest_20250524_132543/results.csv')
results_hardest = pd.read_csv('results/cifar10_hardest_20250524_231250/results.csv')
results_easiest_balanced = pd.read_csv('results/cifar10_easiest_balanced_20250527_000457/results.csv')

mean_random, sem_random = calculate_mean_sem_fid(results_random)
mean_easiest, sem_easiest = calculate_mean_sem_fid(results_easiest)
mean_hardest, sem_hardest = calculate_mean_sem_fid(results_hardest)
mean_easiest_balanced, sem_easiest_balanced = calculate_mean_sem_fid(results_easiest_balanced)

# Plot all results together
import pandas as pd
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_mean_sem_fid(results_df):
    mean_fid = results_df.groupby('percentage')['fid_score'].mean()
    sem_fid = results_df.groupby('percentage')['fid_score'].apply(sem)
    return mean_fid, sem_fid

#CIFAR10 results
# results_random = pd.read_csv('results/cifar10_random_20250523_232825/results.csv')
# results_easiest = pd.read_csv('results/cifar10_easiest_20250524_132543/results.csv')
# results_hardest = pd.read_csv('results/cifar10_hardest_20250524_231250/results.csv')
# results_easiest_balanced = pd.read_csv('results/cifar10_easiest_balanced_20250527_000457/results.csv')

#DIGITS results
# results_random = pd.read_csv('results/digits_random_20250525_141958/results.csv')
# results_easiest = pd.read_csv('results/digits_easiest_20250525_165555/results.csv')
# results_hardest = pd.read_csv('results/digits_hardest_20250525_182403/results.csv')
# results_easiest_balanced = pd.read_csv('results/digits_easiest_balanced_20250527_165927/results.csv')

#FASHION results
results_random = pd.read_csv('results/fashion_random_20250527_235632/results.csv')
results_easiest = pd.read_csv('results/fashion_easiest_20250528_104044/results.csv')
results_hardest = pd.read_csv('results/fashion_hardest_20250529_000648/results.csv')
results_easiest_balanced = pd.read_csv('results/fashion_easiest_balanced_20250527_223420/results.csv')

# Calculate means and SEMs
mean_random, sem_random = calculate_mean_sem_fid(results_random)
mean_easiest, sem_easiest = calculate_mean_sem_fid(results_easiest)
mean_hardest, sem_hardest = calculate_mean_sem_fid(results_hardest)
mean_easiest_balanced, sem_easiest_balanced = calculate_mean_sem_fid(results_easiest_balanced)

print("Mean FID Scores:")
print("Random:", mean_random)
print("Easiest:", mean_easiest)
print("Hardest:", mean_hardest)
print("Easiest Balanced:", mean_easiest_balanced)

# Define custom colors for each experiment type
experiment_colors = {
    'hardest': 'red',
    'random': 'black',
    'easiest': 'green',
    'easiest_balanced': 'darkblue'
}

# Create the figure
plt.figure(figsize=(12, 8))

# Prepare datasets with their configurations
datasets = [
    (mean_hardest, sem_hardest, 'hardest', 'Hardest'),
    (mean_random, sem_random, 'random', 'Random'),
    (mean_easiest, sem_easiest, 'easiest', 'Easiest'),
    (mean_easiest_balanced, sem_easiest_balanced, 'easiest_balanced', 'Easiest Balanced')
]

# Plot each experiment
for mean_data, sem_data, exp_key, exp_label in datasets:
    color = experiment_colors[exp_key]
    
    # Filter data if needed (example: exclude 100% for random like in your code)
    if exp_key == 'random':
        # Filter to include only subsets 50 to 90 (adjust as needed)
        percentages = [p for p in mean_data.index if 50 <= p <= 90]
    else:
        percentages = list(mean_data.index)
    
    # Get corresponding values
    fid_values = [mean_data.loc[p] for p in percentages]
    sem_values = [sem_data.loc[p] for p in percentages]
    
    # Plot with error bars
    plt.errorbar(percentages, fid_values, yerr=sem_values, 
                marker='o', label=f'Experiment: {exp_label}', 
                color=color, linewidth=2, markersize=6, capsize=4)

# Add horizontal reference line for 100% subset with SEM shaded region
reference_fid = mean_random.loc[100]  # Assuming 100% random as reference
reference_sem = sem_random.loc[100]  # Get SEM for 100% subset

plt.axhline(y=reference_fid, color='red', linestyle='--', 
           label='Subset 100 Reference', alpha=0.8)

# Add shaded region for SEM around the baseline
# Add shaded region for SEM around the baseline (full x-axis range)
x_min = min([min(mean_data.index) for mean_data, _, _, _ in datasets])
x_max = max([max(mean_data.index) for mean_data, _, _, _ in datasets])
plt.fill_between([x_min, x_max], 
                reference_fid - reference_sem, 
                reference_fid + reference_sem, 
                color='red', alpha=0.2, zorder=1)

# Customize the plot
plt.xlabel('Subset Percentage (%)')
plt.ylabel('FID Score (lower is better)')
plt.title('FID Scores for Different Subsets Across Experiments')
plt.legend()
plt.grid(True, alpha=0.3)

# Set x-ticks to show all unique percentages
all_percentages = set()
for mean_data, _, _, _ in datasets:
    all_percentages.update(mean_data.index)
plt.xticks(sorted(all_percentages))

# Save the plot (adjust path as needed)
# plt.savefig('fid_comparison_across_experiments.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()