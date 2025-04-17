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

# Set font sizes
rcParams['axes.titlesize'] = 24       # title
rcParams['axes.labelsize'] = 24       # x/y labels
rcParams['xtick.labelsize'] = 20      # x tick labels
rcParams['ytick.labelsize'] = 16      # y tick labels
rcParams['legend.fontsize'] = 18      # legend text
rcParams['figure.titlesize'] = 18     # suptitle if used


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
fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=False)

# Bar plot for CVaR90
bars1 = axes[0].bar(labels, cvar90_values, color='gray', edgecolor='black')
#axes[0].set_title("CVaR$_{90}$")

axes[0].set_ylabel("Transformer Loading [%]")
axes[0].set_ylim(80, 95)  # Y-axis from 80%
axes[0].grid(axis='y')
for bar, val in zip(bars1, cvar90_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}%", ha='center', fontsize=18)

# Bar plot for violation rate > 80%
bars2 = axes[1].bar(labels, violation_rates, color='darkgray', edgecolor='black')
#axes[1].set_title("Constraint Violation Rate > 80%")
axes[1].set_ylabel("Chance of Constraint Violation [%]")
axes[1].set_ylim(0, 40)  # Y-axis from 0 to 20
axes[1].grid(axis='y')
for bar, val in zip(bars2, violation_rates):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}%", ha='center', fontsize=18)

# Shared formatting
for ax in axes:
    ax.set_xticklabels(labels, rotation=30)
    ax.tick_params(axis='y')
axes[1].text(0.5, -0.20, "(b) Constraint Violation Rate", ha='center', va='top',
             fontsize=26, transform=axes[1].transAxes)
axes[0].text(0.5, -0.20, r"(a) $\mathrm{CVaR}_{90}$ – Constraint Violation Severity",
             ha='center', va='top', fontsize=26, transform=axes[0].transAxes)
#plt.suptitle("Transformer Risk Metrics Across Optimization Cases", fontsize=18)
plt.tight_layout()
plt.show()


# === Step X: Helper to load filtered results for line and bus ===
def load_filtered_component_results(file, selected_timesteps, result_index):
    all_results = pickle.load(open(file, "rb"))
    all_rows = []
    for sample_idx, res in enumerate(all_results):
        df = res[result_index]  # index 2: line, 1: bus
        df = df[df['time_step'].isin(selected_timesteps)].copy()
        df['sample_idx'] = sample_idx
        all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)

# === Step X: Load line and bus results ===
line_filtered_data = []
bus_filtered_data = []
for label, file in mc_files.items():
    line_df = load_filtered_component_results(file, high_load_timesteps, result_index=2)
    bus_df = load_filtered_component_results(file, high_load_timesteps, result_index=1)

    line_filtered_data.append(line_df['loading_percent'].values)
    bus_filtered_data.append(bus_df['vm_pu'].values)

# === Step X: Compute CVaR and violation rate ===
def compute_cvar_violation(values_list, threshold, percentile=90):
    cvar_vals = []
    violations = []
    for values in values_list:
        values = np.array(values)
        p_cutoff = np.percentile(values, percentile)
        tail = values[values > p_cutoff]
        cvar = tail.mean() if len(tail) > 0 else 0
        cvar_vals.append(cvar)
        rate = (values > threshold).sum() / len(values) * 100
        violations.append(rate)
    return cvar_vals, violations

line_cvar90, line_violation_rate = compute_cvar_violation(line_filtered_data, threshold=80)
voltage_cvar90, voltage_violation_rate = compute_cvar_violation(bus_filtered_data, threshold=1.05)

# === Step X: Plot Line Loading CVaR & Violation Rate ===
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
bars1 = axes[0].bar(labels, line_cvar90, color='gray', edgecolor='black')
axes[0].set_ylabel("Line Loading [%]")
#axes[0].set_ylim(80, 100)
axes[0].grid(axis='y')
for bar, val in zip(bars1, line_cvar90):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}%", ha='center', fontsize=18)

bars2 = axes[1].bar(labels, line_violation_rate, color='darkgray', edgecolor='black')
axes[1].set_ylabel("Constraint Violation [%]")
#axes[1].set_ylim(0, 40)
axes[1].grid(axis='y')
for bar, val in zip(bars2, line_violation_rate):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}%", ha='center', fontsize=18)

for ax in axes:
    ax.set_xticklabels(labels, rotation=30)
    ax.tick_params(axis='y')
axes[0].text(0.5, -0.20, r"(c) $\mathrm{CVaR}_{90}$ – Line Loading Severity", ha='center', va='top', fontsize=26, transform=axes[0].transAxes)
axes[1].text(0.5, -0.20, "(d) Line Constraint Violation Rate", ha='center', va='top', fontsize=26, transform=axes[1].transAxes)
plt.tight_layout()
plt.show()

# === Step X: Plot Voltage CVaR & Violation Rate ===
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
bars1 = axes[0].bar(labels, voltage_cvar90, color='gray', edgecolor='black')
axes[0].set_ylabel("Voltage Magnitude [p.u.]")
#axes[0].set_ylim(1.04, 1.07)
axes[0].grid(axis='y')
for bar, val in zip(bars1, voltage_cvar90):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.001, f"{val:.3f}", ha='center', fontsize=18)

bars2 = axes[1].bar(labels, voltage_violation_rate, color='darkgray', edgecolor='black')
axes[1].set_ylabel("Voltage Constraint Violations [%]")
#axes[1].set_ylim(0, 20)
axes[1].grid(axis='y')
for bar, val in zip(bars2, voltage_violation_rate):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.1f}%", ha='center', fontsize=18)

for ax in axes:
    ax.set_xticklabels(labels, rotation=30)
    ax.tick_params(axis='y')
axes[0].text(0.5, -0.20, r"(e) $\mathrm{CVaR}_{90}$ – Voltage Severity", ha='center', va='top', fontsize=26, transform=axes[0].transAxes)
axes[1].text(0.5, -0.20, "(f) Voltage Constraint Violation Rate", ha='center', va='top', fontsize=26, transform=axes[1].transAxes)
plt.tight_layout()
plt.show()
