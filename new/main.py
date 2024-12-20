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
import results as rs
import plot as pl
import opf as opf
import drcc as drcc
import montecarlo as mc

#violations_df_opf = rs.load_optim_results("mc_results_opf.pkl")
#violations_df_drcc = rs.load_optim_results("mc_results_drcc.pkl")
opf_results = rs.load_optim_results("opf_results.pkl")
#drcc_results = rs.load_optim_results("drcc_opf_results.pkl") 

pl.plot_opf_results_plotly(opf_results)
#pl.plot_opf_results_plotly(drcc_results)

#pl.compare_heatmap(violations_df_opf, violations_df_drcc, threshold=0.05)