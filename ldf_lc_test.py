import ldf_lc as ldf
import griddata as gd
import plot as pl

import pandas as pd

net, const_load_household_P, const_load_household_Q, const_load_heatpump, time_steps, df_household_prognosis, df_season_heatpump_prognosis, heatpump_scaling_factors_df, T_amb = gd.setup_grid_IAS(season='winter')

Gbus = ldf.calculate_gbus_matrix(net)
Bbus = ldf.calculate_bbus_matrix(net)

# Debugging Prints
#print("\n===== DEBUGGING Gbus and Bbus =====")
#print(f"Gbus Shape: {Gbus.shape}, Expected: ({len(net.bus)}, {len(net.bus)})")
#print(f"Bbus Shape: {Bbus.shape}, Expected: ({len(net.bus)}, {len(net.bus)})\n")

# Print entire Gbus and Bbus matrices
#print("\n===== Gbus Matrix =====")
#print(pd.DataFrame(Gbus, index=net.bus.index, columns=net.bus.index))

#print("\n===== Bbus Matrix =====")
#print(pd.DataFrame(Bbus, index=net.bus.index, columns=net.bus.index))

results = ldf.manual_lindistflow_timeseries(time_steps, net, const_load_household_P, const_load_household_Q, const_load_heatpump, Gbus, Bbus)

#pl.plot_voltage_magnitude(results)
#pl.plot_line_power_flow(results)
#pl.plot_load_power(results)
#pl.plot_branch_current(results)
#pl.plot_ldf_lc_results(results)
pl.plot_ldf_results_plotly(results)