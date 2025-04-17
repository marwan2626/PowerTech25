import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import rcParams
from statsmodels.distributions.empirical_distribution import ECDF

# Define your result files and labels for each case
result_files = [
    "drcc_results_drcc_0_0.15.pkl",
    "drcc_results_drcc_1_0.15.pkl",
    "drcc_results_drcc_1_0.1.pkl",
    "drcc_results_drcc_1_0.05.pkl"
]
labels = ["Deterministic", r'DRCC, $\varepsilon=0.15$', r'DRCC, $\varepsilon=0.10$', r'DRCC, $\varepsilon=0.05$']

# Initialize storage
all_trafo_loading = []
all_line_loading = []  # still extracting if you need it later

rcParams['font.family'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'

# Set font sizes
rcParams['axes.titlesize'] = 24       # title
rcParams['axes.labelsize'] = 20       # x/y labels
rcParams['xtick.labelsize'] = 20      # x tick labels
rcParams['ytick.labelsize'] = 20      # y tick labels
rcParams['legend.fontsize'] = 18      # legend text
rcParams['figure.titlesize'] = 18     # suptitle if used

# Load and extract transformer & line loading values
for file in result_files:
    with open(file, "rb") as f:
        results = pickle.load(f)

    # Trafo: flatten all loading values across timesteps
    trafo_loading_flat = []
    for t in results['transformer_loading']:
        trafo_loading_flat.extend(results['transformer_loading'][t].values())
    all_trafo_loading.append(trafo_loading_flat)

    # Line: flatten all loading values across timesteps and lines
    line_loading_flat = []
    for t in results['line_loading']:
        line_loading_flat.extend(results['line_loading'][t].values())
    all_line_loading.append(line_loading_flat)


medians = np.array([np.median(x) for x in all_trafo_loading])
maxima = np.array([np.max(x) for x in all_trafo_loading])


# Bar Plot of Median and Max Transformer Loading with Uncertainty Margin
x = np.arange(len(labels))  # x locations for the groups
width = 0.35  # width of the bars

plt.figure(figsize=(10, 6))

# # Base bar: median
# plt.bar(x - width/2, medians, width, label='Median', color='gray', edgecolor='black')

# # Base bar: max (main bar from 0 to max)
# plt.bar(x + width/2, maxima, width, label='Maximum', color='darkgray', edgecolor='black')

import matplotlib.patches as mpatches

# Bar Plot of Median and Max Transformer Loading with Uncertainty Margin
x = np.arange(len(labels))  # x locations for the groups
width = 0.35  # width of the bars

plt.figure(figsize=(10, 6))

# Base bar: median
plt.bar(x - width/2, medians, width, label='Median', color='gray', edgecolor='black')

# Base bar: max (main bar from 0 to max)
plt.bar(x + width/2, maxima, width, label='Maximum', color='darkgray', edgecolor='black')

# Hatched bar: from max to 80% (only if max < 80)
for i in range(len(labels)):
    if maxima[i] < 80:
        plt.bar(x[i] + width/2, 80 - maxima[i], width,
                bottom=maxima[i],
                color='none',
                edgecolor='black',
                hatch='///')

# Add reference line
plt.axhline(y=80, color='red', linestyle='-', linewidth=1.5, label="Reference Limit (80%)")

# --- Custom legend entry for hatched area ---
hatch_patch = mpatches.Patch(facecolor='none', edgecolor='black', hatch='///', label='Uncertainty Margin')

# Final formatting
plt.xticks(x, labels)
plt.ylabel("Transformer Loading [%]")
plt.ylim(50, 100)
plt.grid(True, axis='y')

# Add custom legend
plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + [hatch_patch])

plt.tight_layout()
plt.show()


