"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
Optimization by Marwan Mostafa 

Main File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### SCRIPTS ####
import data as dt
import griddata as gd 
#import results as rs
import plot as pl
import opf as opf
#import drcc as drcc
#import montecarlo as mc

net, const_load_household, const_load_heatpump, time_steps, df_household, df_season_heatpump_prognosis, heatpump_scaling_factors_df, T_amb = gd.setup_grid_powertech25(season='winter')
Bbus = dt.calculate_bbus_matrix(net)
results = opf.solve_opf(net, time_steps, const_load_heatpump, const_load_household, heatpump_scaling_factors_df, T_amb, Bbus)

pl.plot_opf_results_plotly(results)