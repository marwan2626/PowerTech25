import ldf_lc as ldf
import griddata as gd
import plot as pl
import drcc_ldf as drcc
import montecarlo_validation as mc
import results as rs
import parameters as par

import pandas as pd
import pickle





net, time_steps, const_pv, const_load_household_P, const_load_household_Q, const_load_heatpump, const_load_heatpump_Q ,df_household_prognosis, df_season_heatpump_prognosis, df_season_pv_prognosis, heatpump_scaling_factors_df, T_amb, electricity_price, original_sgen_p_mw = gd.setup_grid_IAS(season='winter')


#results = ldf.manual_lindistflow_timeseries(time_steps, net)
drcc_results = drcc.solve_drcc_opf(net, time_steps, electricity_price, const_pv, const_load_household_P, const_load_household_Q, const_load_heatpump, T_amb)

# filename = f"drcc_results_drcc_{par.DRCC_FLG}_{par.epsilon}.pkl"
# drcc_results = rs.load_optim_results(filename)

### DRCC MONTE CARLO ANALYSIS ###
# mc_samples = mc.generate_samples_ldf(df_season_heatpump_prognosis,df_season_pv_prognosis)

# with open("mc_samples.pkl", "wb") as f:
#     pickle.dump(mc_samples, f)

# with open("mc_samples.pkl", "rb") as f:
#     mc_samples = pickle.load(f)

# mc_samples = mc_samples

# all_results_drcc, violation_probability_drcc, violations_df_drcc, trafo_violations_df, overall_line_violations_drcc, drcc_mc_line_results_df = mc.ldf_montecarlo_analysis_with_violations(
#     net,
#     original_sgen_p_mw,
#     time_steps,
#     drcc_results,
#     heatpump_scaling_factors_df,
#     const_load_household_P, 
#     const_load_household_Q,
#     mc_samples,
#     n_jobs=-1,
#     log_file="violation_log_drcc_1_e_10.txt",
# )

# # Save the results to a file
# if all_results_drcc is not None:
#     rs.save_optim_results(all_results_drcc, "mc_results_drcc_1_e_10.pkl")
#pl.plot_ldf_results_plotly(results_variance)
#pl.plot_ldf_results_plotly(results)
#pl.plot_ldf_drcc_results_plotly(drcc_results)