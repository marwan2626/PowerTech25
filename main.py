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
import montecarlo_validation as mc
#import results as rs
import plot as pl
import opf as opf
import drcc as drcc
import drcc2 as drcc2
#import montecarlo as mc

net, const_load_household, const_load_heatpump, time_steps, df_household, df_season_heatpump_prognosis, heatpump_scaling_factors_df, T_amb = gd.setup_grid_powertech25(season='winter')
Bbus = dt.calculate_bbus_matrix(net)
#results = opf.solve_opf(net, time_steps, const_load_heatpump, const_load_household, heatpump_scaling_factors_df, T_amb, Bbus)
#pl.plot_opf_results_plotly(results)
#results_drcc= drcc.drcc_opf(net, time_steps, const_load_heatpump, const_load_household, Bbus, df_season_heatpump_prognosis, heatpump_scaling_factors_df,T_amb, max_iter_drcc=100, alpha=0.05, eta=5e-4)
results_drcc = drcc2.solve_drcc_opf(net, time_steps, const_load_heatpump, const_load_household, df_season_heatpump_prognosis, heatpump_scaling_factors_df, T_amb, Bbus)
pl.plot_opf_results_plotly(results_drcc)

mc_samples = mc.generate_samples(df_season_heatpump_prognosis)
all_results_drcc, violation_probability_drcc, violations_df_drcc, trafo_violations_df, overall_line_violations_drcc, drcc_mc_line_results_df = mc.montecarlo_analysis_with_violations(
    net,
    time_steps,
    results_drcc,
    const_load_household,
    const_load_heatpump,
    heatpump_scaling_factors_df,
    mc_samples,
    n_jobs=-1,
    log_file="violation_log_drcc.txt",
)
