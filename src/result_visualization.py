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


def calculate_mean_sem_fid(results_df):
    mean_fid = results_df.groupby('percentage')['fid_score'].mean()
    sem_fid = results_df.groupby('percentage')['fid_score'].apply(sem)
    return mean_fid, sem_fid

def calculate_mean_sem_pr(results_df):
    mean_p = results_df.groupby('percentage')['precision'].mean()
    sem_p = results_df.groupby('percentage')['precision'].apply(sem)
    mean_r = results_df.groupby('percentage')['recall'].mean()
    sem_r = results_df.groupby('percentage')['recall'].apply(sem)
    return mean_p, sem_p, mean_r, sem_r

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

# results_random = pd.read_csv('results/digits_random_20250601_125836/results.csv')
# results_easiest = pd.read_csv('results/digits_easiest_20250601_195212/results.csv')
# results_hardest = pd.read_csv('results/digits_hardest_20250602_092046/results.csv')
# results_easiest_balanced = pd.read_csv('results/digits_easiest_balanced_20250601_220851/results.csv')

#FASHION results
# results_random = pd.read_csv('results/fashion_random_20250527_235632/results.csv')
# results_easiest = pd.read_csv('results/fashion_easiest_20250528_104044/results.csv')
# results_hardest = pd.read_csv('results/fashion_hardest_20250529_000648/results.csv')
# results_easiest_balanced = pd.read_csv('results/fashion_easiest_balanced_20250527_223420/results.csv')

results_random = pd.read_csv('results/fashion_random_20250604_155308/results.csv')
results_easiest = pd.read_csv('results/fashion_easiest_20250604_195532/results.csv')
results_hardest = pd.read_csv('results/fashion_hardest_20250604_213744/results.csv')
results_easiest_balanced = pd.read_csv('results/fashion_easiest_balanced_20250604_181711/results.csv')

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

# Calculate Precision and Recall means and SEMs
mean_random_p, sem_random_p, mean_random_r, sem_random_r = calculate_mean_sem_pr(results_random)
mean_easiest_p, sem_easiest_p, mean_easiest_r, sem_easiest_r = calculate_mean_sem_pr(results_easiest)
mean_hardest_p, sem_hardest_p, mean_hardest_r, sem_hardest_r = calculate_mean_sem_pr(results_hardest)
mean_easiest_balanced_p, sem_easiest_balanced_p, mean_easiest_balanced_r, sem_easiest_balanced_r = calculate_mean_sem_pr(results_easiest_balanced)

experiment_colors = {
    'hardest': 'red',
    'random': 'black',
    'easiest': 'green',
    'easiest_balanced': 'darkblue'
}

plt.figure(figsize=(12, 8))

datasets = [
    (mean_hardest, sem_hardest, 'hardest', 'Hardest'),
    (mean_random, sem_random, 'random', 'Random'),
    (mean_easiest, sem_easiest, 'easiest', 'Easiest'),
    (mean_easiest_balanced, sem_easiest_balanced, 'easiest_balanced', 'Easiest Balanced')
]

for mean_data, sem_data, exp_key, exp_label in datasets:
    color = experiment_colors[exp_key]
    
    if exp_key == 'random':
        percentages = [p for p in mean_data.index if 50 <= p <= 90]
    else:
        percentages = list(mean_data.index)
    
    fid_values = [mean_data.loc[p] for p in percentages]
    sem_values = [sem_data.loc[p] for p in percentages]
    
    plt.errorbar(percentages, fid_values, yerr=sem_values, 
                marker='o', label=f'Experiment: {exp_label}', 
                color=color, linewidth=2, markersize=6, capsize=4)

# Add horizontal reference line for 100% subset with SEM shaded region
reference_fid = mean_random.loc[100]
reference_sem = sem_random.loc[100]

plt.axhline(y=reference_fid, color='red', linestyle='--', 
           label='Subset 100 Reference', alpha=0.8)

x_min = min([min(mean_data.index) for mean_data, _, _, _ in datasets])
x_max = max([max(mean_data.index) for mean_data, _, _, _ in datasets])
plt.fill_between([x_min, x_max], 
                reference_fid - reference_sem, 
                reference_fid + reference_sem, 
                color='red', alpha=0.2, zorder=1)

plt.xlabel('Subset Percentage (%)')
plt.ylabel('FID Score (lower is better)')
plt.title('FID Scores for Different Subsets Across Experiments')
plt.legend()
plt.grid(True, alpha=0.3)

all_percentages = set()
for mean_data, _, _, _ in datasets:
    all_percentages.update(mean_data.index)
plt.xticks(sorted(all_percentages))

# plt.savefig('fid_comparison_across_experiments.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))

datasets = [
    (mean_hardest_p, sem_hardest_p, mean_hardest_r, sem_hardest_r, 'hardest', 'Hardest'),
    (mean_random_p, sem_random_p, mean_random_r, sem_random_r, 'random', 'Random'),
    (mean_easiest_p, sem_easiest_p, mean_easiest_r, sem_easiest_r, 'easiest', 'Easiest'),
    (mean_easiest_balanced_p, sem_easiest_balanced_p, mean_easiest_balanced_r, sem_easiest_balanced_r, 'easiest_balanced', 'Easiest Balanced')
]

for mean_p, sem_p, mean_r, sem_r, exp_key, exp_label in datasets:
    color = experiment_colors[exp_key]
    
    if exp_key == 'random':
        percentages = [p for p in mean_p.index if 50 <= p <= 90]
    else:
        percentages = list(mean_p.index)
    
    precision_values = [mean_p.loc[p] for p in percentages]
    sem_precision_values = [sem_p.loc[p] for p in percentages]
    recall_values = [mean_r.loc[p] for p in percentages]
    sem_recall_values = [sem_r.loc[p] for p in percentages]
    
    plt.errorbar(percentages, precision_values, yerr=sem_precision_values, 
                marker='o', label=f'Precision: {exp_label}', 
                color=color, linewidth=2, markersize=6, capsize=4)
    
    plt.errorbar(percentages, recall_values, yerr=sem_recall_values, 
                marker='x', linestyle='--', label=f'Recall: {exp_label}', 
                color=color, linewidth=2, markersize=6, capsize=4)

reference_precision = mean_random_p.loc[100]
reference_recall = mean_random_r.loc[100]
reference_sem_precision = sem_random_p.loc[100]
reference_sem_recall = sem_random_r.loc[100]

plt.axhline(y=reference_precision, color='red', linestyle='--', 
           label='Subset 100 Reference Precision', alpha=0.8)
plt.axhline(y=reference_recall, color='blue', linestyle='--', 
           label='Subset 100 Reference Recall', alpha=0.8)

x_min = min([min(mean_p.index) for mean_p, _, _, _,_,_ in datasets])
x_max = max([max(mean_p.index) for mean_p, _, _, _,_,_ in datasets])
plt.fill_between([x_min, x_max], 
                reference_precision - reference_sem_precision, 
                reference_precision + reference_sem_precision, 
                color='red', alpha=0.2, zorder=1)

x_min = min([min(mean_r.index) for _, _, mean_r, _,_,_ in datasets])    
x_max = max([max(mean_r.index) for _, _, mean_r, _,_,_ in datasets])
plt.fill_between([x_min, x_max], 
                reference_recall - reference_sem_recall, 
                reference_recall + reference_sem_recall, 
                color='blue', alpha=0.2, zorder=1)

plt.xlabel('Subset Percentage (%)')
plt.ylabel('Precision and Recall')
plt.title('Precision and Recall for Different Subsets Across Experiments')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()