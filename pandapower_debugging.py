"""
Code for the publication Reliability-Constrained Thermal Storage Sizing in
Multi-Energy Systems under Uncertainty
Optimization by Marwan Mostafa 

Pandapower debugging File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### SCRIPTS ####
import griddata as gd 
import opf as opf
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
import pandas as pd
import os
import pandapower as pp

# Set up the grid and time series data
net, time_steps, const_pv, const_load_household_P, const_load_household_Q, const_load_heatpump, const_load_heatpump_Q ,df_household_prognosis, df_season_heatpump_prognosis, heatpump_scaling_factors_df, T_amb, electricity_price = gd.setup_grid_IAS(season='winter')

from pandapower.timeseries import OutputWriter

def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xlsx", log_variables=list())

    # Bus Variables
    ow.log_variable('res_bus', 'vm_pu')  # Voltage Magnitude
    ow.log_variable('res_bus', 'p_mw')   # Active Power (P)
    ow.log_variable('res_bus', 'q_mvar') # Reactive Power (Q)
    ow.log_variable('res_bus', 'va_degree')  # Voltage Angle

    # Load Variables
    ow.log_variable('res_load', 'p_mw')  # Load Power
    ow.log_variable('res_load', 'q_mvar')  # Load Reactive Power

    # Line Variables
    ow.log_variable('res_line', 'p_from_mw')  # Line Active Power
    ow.log_variable('res_line', 'q_from_mvar')  # Line Reactive Power
    ow.log_variable('res_line', 'i_ka')  # Line Current

    # Transformer Variables
    ow.log_variable('res_trafo', 'loading_percent')  # Transformer Loading

    return ow


output_dir = "output"

# Define the function to run DC power flow
def run_dc_power_flow(net, **kwargs):
    #print(f"Loads at each bus: {net.load.p_mw.values.tolist()}")
    #print(f"Generation at each bus: {net.sgen.p_mw.values.tolist()}")
    
    # Run the DC power flow
    pp.runpp(net)

# Create output writer and run the time series
ow = create_output_writer(net, time_steps, output_dir=output_dir)
run_timeseries(net, time_steps=time_steps, run=run_dc_power_flow)

# Plot and print results
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt

output_dir = "output"

def plot_results():
    # Helper function to read and plot data
    def plot_file(file_path, ylabel, title, label=None):
        if os.path.exists(file_path):
            df = pd.read_excel(file_path, index_col=0)
            if not df.empty:
                df.plot()
                plt.xlabel("Time Step")
                plt.ylabel(ylabel)
                plt.title(title)
                plt.grid()
                plt.show()
            else:
                print(f"Info: {title} data is empty. Skipping plot.")
        else:
            print(f"Warning: {file_path} not found. Skipping {title}.")

    # 1️⃣ Voltage Magnitude
    plot_file(os.path.join(output_dir, "res_bus", "vm_pu.xlsx"), 
              "Voltage Magnitude [p.u.]", "Voltage Magnitude")

    # 2️⃣ P Node (Bus Active Power)
    plot_file(os.path.join(output_dir, "res_bus", "p_mw.xlsx"), 
              "Bus Power [MW]", "Bus Active Power (P)")

    # 3️⃣ Q Node (Bus Reactive Power)
    plot_file(os.path.join(output_dir, "res_bus", "q_mvar.xlsx"), 
              "Bus Reactive Power [MVar]", "Bus Reactive Power (Q)")

    # 4️⃣ Load Power
    plot_file(os.path.join(output_dir, "res_load", "p_mw.xlsx"), 
              "Load Power [MW]", "Load Active Power")
    
    # 4️⃣ Load Power
    plot_file(os.path.join(output_dir, "res_load", "q_mvar.xlsx"), 
              "Load Power [MVar]", "Load Reactive Power")

    # 5️⃣ Line Power (Active)
    plot_file(os.path.join(output_dir, "res_line", "p_from_mw.xlsx"), 
              "Line Power [MW]", "Line Active Power (P)")
    
    # 6️⃣ Line Reactive Power
    plot_file(os.path.join(output_dir, "res_line", "q_from_mvar.xlsx"), 
              "Line Reactive Power [MVar]", "Line Reactive Power (Q)")

    # 7️⃣ Transformer Loading
    plot_file(os.path.join(output_dir, "res_trafo", "loading_percent.xlsx"), 
              "Transformer Loading [%]", "Transformer Loading")

    # 8️⃣ Line Current
    plot_file(os.path.join(output_dir, "res_line", "i_ka.xlsx"), 
              "Current Magnitude [kA]", "Line Current")

    # Debugging Bbus matrix
    print("Bbus Matrix:")
    print(net._ppc['internal']['Bbus'].A)

# Call the function
plot_results()
