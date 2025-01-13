"""
Code for the publication Reliability-Constrained Thermal Storage Sizing in
Multi-Energy Systems under Uncertainty
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
import results as rs
import plot as pl
import opf as opf
import drcc as drcc
import reliability_parallel as rl

net, const_load_household, const_load_heatpump, time_steps, df_household, df_season_heatpump_prognosis, heatpump_scaling_factors_df, T_amb = gd.setup_grid_powertech25(season='winter')
Bbus = dt.calculate_bbus_matrix(net)

results_drcc = drcc.solve_drcc_opf(net, time_steps, const_load_heatpump, const_load_household, df_season_heatpump_prognosis, heatpump_scaling_factors_df, T_amb, Bbus)
#pl.plot_opf_results_plotly(results_drcc)

### RELIABILITY ANALYSIS ###
#results_rel = rl.reliability_analysis(net, time_steps, const_load_heatpump, const_load_household, T_amb, Bbus, n_jobs=-1)



### DRCC MONTE CARLO ANALYSIS ###
#mc_samples = mc.generate_samples(df_season_heatpump_prognosis)
# all_results_drcc, violation_probability_drcc, violations_df_drcc, trafo_violations_df, overall_line_violations_drcc, drcc_mc_line_results_df = mc.montecarlo_analysis_with_violations(
#     net,
#     time_steps,
#     results_drcc,
#     const_load_household,
#     const_load_heatpump,
#     heatpump_scaling_factors_df,
#     mc_samples,
#     n_jobs=-1,
#     log_file="violation_log_drcc.txt",
# )

# # Save the results to a file
# if trafo_violations_df is not None:
#     rs.save_optim_results(trafo_violations_df, "mc_results_opf.pkl")
