###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### SCRIPTS ####
import data as dt
import griddata as gd 
import montecarlo_validation as mc
import results as rs
import plot as pl
import opf as opf
import drcc as drcc
import reliability_parallel as rl

#### PACKAGES ####
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams



all_results_drcc_4 = rs.load_optim_results("mc_results_drcc_1_e_05.pkl")
all_results_drcc_3 = rs.load_optim_results("mc_results_drcc_1_e_10.pkl")
all_results_drcc_2 = rs.load_optim_results("mc_results_drcc_1_e_15.pkl")
all_results_drcc_1 = rs.load_optim_results("mc_results_drcc_0_e_0.pkl")

trafo_loading_percentages_drcc_1 = [
    result[3]['loading_percent'] for result in all_results_drcc_1
]

trafo_loading_percentages_drcc_2 = [
    result[3]['loading_percent'] for result in all_results_drcc_2
]

trafo_loading_percentages_drcc_3 = [
    result[3]['loading_percent'] for result in all_results_drcc_3
]

trafo_loading_percentages_drcc_4 = [
    result[3]['loading_percent'] for result in all_results_drcc_4
]

# Combine all loading percentages into a single DataFrame
combined_trafo_loading_drcc_4 = pd.concat(trafo_loading_percentages_drcc_1, ignore_index=True)
combined_trafo_loading_drcc_3 = pd.concat(trafo_loading_percentages_drcc_2, ignore_index=True)
combined_trafo_loading_drcc_2 = pd.concat(trafo_loading_percentages_drcc_3, ignore_index=True)
combined_trafo_loading_drcc_1 = pd.concat(trafo_loading_percentages_drcc_4, ignore_index=True)

# Prepare data for the box plot
data = [combined_trafo_loading_drcc_1, combined_trafo_loading_drcc_2, combined_trafo_loading_drcc_3, combined_trafo_loading_drcc_4]

labels = ['Deterministic', r'DRCC, $\varepsilon=0.15$', r'DRCC, $\varepsilon=0.10$', r'DRCC, $\varepsilon=0.5$']

labels = [r'DRCC, $\mathrm{\varepsilon}=0.05$',
          r'DRCC, $\mathrm{\varepsilon}=0.10$', 
          r'DRCC, $\mathrm{\varepsilon}=0.15$',
          'Deterministic' 
          ]

# Set global font to Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'

# Create the box plot
plt.figure(figsize=(10, 6))
boxplot = plt.boxplot(data, labels=labels, patch_artist=True, showfliers=False, vert=False)
# Set the fill color to gray for the boxes
for box in boxplot['boxes']:
    box.set_facecolor('gray')

# Change the color of the median lines to black
for median in boxplot['medians']:
    median.set_color('black')
    #median.set_linewidth(1.5)  # Optional: Make the line thicker

#plt.title('Transformer Loading Percentages')
#plt.xlabel('Simulation Groups')
plt.margins(x=0.01)

plt.xlabel('Transformer Loading Percentage (%)', fontsize=22)
# Customize x-axis label font size and rotation
plt.yticks(fontsize=20)
plt.xticks(ticks=range(0, 101, 20), fontsize=18)
plt.grid(axis='x')
plt.xlim(0, 100)
plt.tight_layout()
plt.show()

