import ldf_lc as ldf
import griddata as gd
import plot as pl
import drcc_ldf as drcc

import pandas as pd

net, time_steps, const_pv, const_load_household_P, const_load_household_Q, const_load_heatpump, const_load_heatpump_Q ,df_household_prognosis, df_season_heatpump_prognosis, heatpump_scaling_factors_df, T_amb, electricity_price = gd.setup_grid_IAS(season='winter')

for t in time_steps:
    electricity_price[t] = 100

#results = ldf.manual_lindistflow_timeseries(time_steps, net)
drcc_results = drcc.solve_drcc_opf(net, time_steps, electricity_price, const_pv, const_load_household_P, const_load_household_Q, const_load_heatpump, T_amb)

#pl.plot_branch_current(results)
#pl.plot_ldf_results_plotly(results)
pl.plot_ldf_drcc_results_plotly(drcc_results)