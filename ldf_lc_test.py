import ldf_lc as ldf
import griddata as gd
import plot as pl
import drcc_ldf as drcc

import pandas as pd




net, time_steps, const_pv, const_load_household_P, const_load_household_Q, const_load_heatpump, const_load_heatpump_Q ,df_household_prognosis, df_season_heatpump_prognosis, heatpump_scaling_factors_df, T_amb, electricity_price = gd.setup_grid_IAS(season='winter')
print("\n--- Raw forecast data for controllable loads at t=0 ---")
df = const_load_heatpump.data_source.df

# Print all profiles used
for col in df.columns:
    print(f"{col}: {df.loc[0, col]:.5f} MW")

#variance_net = gd.setup_grid_IAS_variance(season='winter')

for t in time_steps:
    electricity_price[t] = 100

#results_variance = drcc.calculate_variance_propagation(time_steps, variance_net)
#V_var = results_variance.loc[20, ("V_variance", 19)]
#print(f"Variance of voltage at bus 19: {V_var}")



#results = ldf.manual_lindistflow_timeseries(time_steps, net)
drcc_results = drcc.solve_drcc_opf(net, time_steps, electricity_price, const_pv, const_load_household_P, const_load_household_Q, const_load_heatpump, T_amb)

#pl.plot_ldf_results_plotly(results_variance)
#pl.plot_ldf_results_plotly(results)
pl.plot_ldf_drcc_results_plotly(drcc_results)