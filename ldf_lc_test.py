import ldf_lc as ldf
import griddata as gd
import plot as pl
import drcc_ldf as drcc

import pandas as pd

net, time_steps, const_pv, const_load_household_P, const_load_household_Q, const_load_heatpump, const_load_heatpump_Q ,df_household_prognosis, df_season_heatpump_prognosis, heatpump_scaling_factors_df, T_amb, electricity_price = gd.setup_grid_IAS(season='summer')
# Debugging Prints
#print("\n===== DEBUGGING Gbus and Bbus =====")
#print(f"Gbus Shape: {Gbus.shape}, Expected: ({len(net.bus)}, {len(net.bus)})")
#print(f"Bbus Shape: {Bbus.shape}, Expected: ({len(net.bus)}, {len(net.bus)})\n")

# Print entire Gbus and Bbus matrices
#print("\n===== Gbus Matrix =====")
#print(pd.DataFrame(Gbus, index=net.bus.index, columns=net.bus.index))

#print("\n===== Bbus Matrix =====")
#print(pd.DataFrame(Bbus, index=net.bus.index, columns=net.bus.index))

#results = ldf.manual_lindistflow_timeseries(time_steps, net)
results = drcc.solve_drcc_opf(net, time_steps, electricity_price, const_pv, const_load_household_P, const_load_household_Q, const_load_heatpump, T_amb)

#pl.plot_voltage_magnitude(results)
#pl.plot_line_power_flow(results)
#pl.plot_load_power(results)
#pl.plot_branch_current(results)
#pl.plot_ldf_lc_results(results)
pl.plot_ldf_results_plotly(results)