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

#### PACKAGES ####
#import pandapower.plotting as pp 
#import matplotlib.pyplot as plt

season = 'winter'
#net, const_load_household, const_load_heatpump, time_steps, df_season_heatpump_prognosis, df_household, df_heatpump, heatpump_scaling_factors_df = gd.setup_grid_irep(season)
net, const_load_household, const_load_heatpump, time_steps, df_season_heatpump_prognosis, df_household, heatpump_scaling_factors_df = gd.setup_grid_irep_forecast(season)
Bbus = dt.calculate_bbus_matrix(net)
mc_samples = mc.generate_samples(df_season_heatpump_prognosis)


#opf_results = opf.solve_opf6(net, time_steps, const_load_heatpump, const_load_household, heatpump_scaling_factors_df, Bbus)
opf_results = rs.load_optim_results("opf_results.pkl")
# pl.plot_opf_results_plotly(opf_results)

# all_results_opf, violation_probability_opf, violations_df_opf, overall_line_violations_opf, opf_mc_line_results_df = mc.montecarlo_analysis_with_violations(
#     net,
#     time_steps,
#     opf_results,
#     const_load_household,
#     const_load_heatpump,
#     heatpump_scaling_factors_df,
#     mc_samples,
#     n_jobs=-1,
#     log_file="violation_log_opf.txt",
# )

# # Save the results to a file
# if violations_df_opf is not None:
#     rs.save_optim_results(violations_df_opf, "mc_results_opf.pkl")

# # Save the results to a file
# if opf_mc_line_results_df is not None:
#     rs.save_optim_results(opf_mc_line_results_df, "opf_mc_line_results.pkl")

violations_df_opf = rs.load_optim_results("mc_results_opf.pkl")
opf_mc_line_results_df = rs.load_optim_results("opf_mc_line_results.pkl")



net_id_before = id(net)
net, const_load_household, const_load_heatpump, time_steps, df_season_heatpump_prognosis, df_household, heatpump_scaling_factors_df = gd.setup_grid_irep_forecast(season)
print("New net ID:", id(net), "Old net ID:", net_id_before)
#Bbus = dt.calculate_bbus_matrix(net)

drcc_results = drcc.drcc_opf(net, time_steps, const_load_heatpump, const_load_household, Bbus, df_season_heatpump_prognosis, heatpump_scaling_factors_df, max_iter_drcc=100, alpha=0.05, eta=5e-4)
#drcc_results = rs.load_optim_results("drcc_opf_results.pkl") 
pl.plot_opf_results_plotly(drcc_results)
all_results_drcc, violation_probability_drcc, violations_df_drcc, overall_line_violations_drcc, drcc_mc_line_results_df = mc.montecarlo_analysis_with_violations(
    net,
    time_steps,
    drcc_results,
    const_load_household,
    const_load_heatpump,
    heatpump_scaling_factors_df,
    mc_samples,
    n_jobs=-1,
    log_file="violation_log_drcc.txt",
)

# Save the results to a file
if violations_df_drcc is not None:
    rs.save_optim_results(violations_df_drcc, "mc_results_drcc.pkl")

# Save the results to a file
if drcc_mc_line_results_df is not None:
    rs.save_optim_results(drcc_mc_line_results_df, "drcc_mc_line_results.pkl")


pl.compare_heatmap(violations_df_opf, violations_df_drcc, threshold=0.05)


pl.box_line_loading_two_subplots(opf_mc_line_results_df, drcc_mc_line_results_df)

