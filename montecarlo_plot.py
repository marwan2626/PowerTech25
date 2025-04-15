import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# === Step 1: Load deterministic OPF result and find high-load timesteps ===
with open("drcc_results_drcc_0_0.15.pkl", "rb") as f:
    base_opf = pickle.load(f)

high_load_timesteps = []
for t, loading_dict in base_opf['transformer_loading'].items():
    if any(val > 60 for val in loading_dict.values()):
        high_load_timesteps.append(t)

print(f"High-load timesteps (>70%%): {high_load_timesteps}")

# === Step 2: Define MC result files ===
mc_files = {
    "Deterministic": "mc_results_drcc_0_e_15.pkl",
    r'DRCC, $\varepsilon=0.15$': "mc_results_drcc_1_e_15.pkl",
    r'DRCC, $\varepsilon=0.10$': "mc_results_drcc_1_e_10.pkl",
    r'DRCC, $\varepsilon=0.05$': "mc_results_drcc_1_e_05.pkl"
}

# === Step 3: Load and filter MC results ===
def load_filtered_mc_trafo_results(file, selected_timesteps):
    all_results = pickle.load(open(file, "rb"))
    
    all_rows = []
    for sample_idx, res in enumerate(all_results):
        df = res[3]  # res_trafo
        df = df[df['time_step'].isin(selected_timesteps)].copy()
        df['sample_idx'] = sample_idx
        all_rows.append(df)

    combined_df = pd.concat(all_rows, ignore_index=True)
    
    # Optional check
    print(combined_df.head())
    print(f"Filtered {len(combined_df)} rows from {len(all_results)} samples")
    
    return combined_df


filtered_data = []
labels = []

for label, file in mc_files.items():
    filtered_df = load_filtered_mc_trafo_results(file, high_load_timesteps)
    filtered_data.append(filtered_df['loading_percent'].values)
    labels.append(label)

# === Step 4: Boxplot ===
rcParams['font.family'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'

plt.figure(figsize=(10, 6))
boxplot = plt.boxplot(filtered_data, labels=labels, patch_artist=True, showfliers=True, vert=False)

for box in boxplot['boxes']:
    box.set_facecolor('gray')
for median in boxplot['medians']:
    median.set_color('black')

plt.xlabel('Transformer Loading Percentage (%)', fontsize=22)
plt.yticks(fontsize=20)
plt.xticks(ticks=range(0, 101, 20), fontsize=18)
plt.grid(axis='x')
plt.tight_layout()
plt.show()

# === Step 5: Calculate CVaR95 and Violation Rate for each case ===
cvar90_values = []
violation_rates = []

for values in filtered_data:
    values = np.array(values)
    # CVaR95
    p90 = np.percentile(values, 90)
    tail = values[values > p90]
    cvar90 = tail.mean() if len(tail) > 0 else 0
    cvar90_values.append(cvar90)

    # Violation rate above 80%
    violations = values[values > 80]
    rate = len(violations) / len(values) * 100
    violation_rates.append(rate)

# === Step 6: Plot CVaR and Violation Rate ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

# Bar plot for CVaR90
bars1 = axes[0].bar(labels, cvar90_values, color='gray', edgecolor='black')
axes[0].set_title("CVaR$_{90}$", fontsize=16)
axes[0].set_ylabel("Transformer Loading [%]", fontsize=14)
axes[0].set_ylim(80, max(cvar90_values) + 1)  # Y-axis from 80%
axes[0].grid(axis='y')
for bar, val in zip(bars1, cvar90_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}%", ha='center', fontsize=12)

# Bar plot for violation rate > 80%
bars2 = axes[1].bar(labels, violation_rates, color='darkgray', edgecolor='black')
axes[1].set_title("Constraint Violation Rate > 80%", fontsize=16)
axes[1].set_ylabel("Constraint Violation [%]", fontsize=14)
axes[1].set_ylim(0, 40)  # Y-axis from 0 to 20
axes[1].grid(axis='y')
for bar, val in zip(bars2, violation_rates):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}%", ha='center', fontsize=12)

# Shared formatting
for ax in axes:
    ax.set_xticklabels(labels, rotation=15, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

plt.suptitle("Transformer Risk Metrics Across Optimization Cases", fontsize=18)
plt.tight_layout()
plt.show()