"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
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
season = 'winter'
net, const_load_household, const_load_heatpump, time_steps, df_household, df_heatpump = gd.setup_grid_irep(season)

#net, df_pv, df, pv_generators, const_load, const_pv = gd.setup_grid()
#time_steps = df_pv.index

# Create the output writer
def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xlsx", log_variables=list())
    # Log these variables during the time series loop
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'va_degree')  # Log voltage angle in degrees
    ow.log_variable('res_bus', 'p_mw')  # Log bus active power
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_ka')
    return ow

output_dir = "output"

# Define the function to run DC power flow
def run_dc_power_flow(net, **kwargs):
    # Print loads and generation for debugging
    print(f"Time step {kwargs.get('time_step', 'unknown')}:")
    print(f"Loads at each bus: {net.load.p_mw.values.tolist()}")
    print(f"Generation at each bus: {net.sgen.p_mw.values.tolist()}")
    
    # Run the DC power flow
    pp.rundcpp(net)

# Create output writer and run the time series
ow = create_output_writer(net, time_steps, output_dir=output_dir)
run_timeseries(net, time_steps=time_steps, run=run_dc_power_flow)

# Plot and print results
import matplotlib.pyplot as plt

# Voltage results
vm_pu_file = os.path.join(output_dir, "res_bus", "vm_pu.xlsx")
vm_pu = pd.read_excel(vm_pu_file, index_col=0)
vm_pu.plot(label="vm_pu")
plt.xlabel("time step")
plt.ylabel("voltage mag. [p.u.]")
plt.title("Voltage Magnitude")
plt.grid()
plt.show()

# Voltage results
bus_pmw_file = os.path.join(output_dir, "res_bus", "p_mw.xlsx")
bus_p_mw = pd.read_excel(bus_pmw_file, index_col=0)
bus_p_mw.plot(label="p_mw")
plt.xlabel("time step")
plt.ylabel("power [MW]")
plt.title("Power")
plt.grid()
plt.show()

# Voltage angle results
va_degree_file = os.path.join(output_dir, "res_bus", "va_degree.xlsx")
va_degree = pd.read_excel(va_degree_file, index_col=0)
va_degree.plot(label="va_degree")
plt.xlabel("time step")
plt.ylabel("voltage degree")
plt.title("Voltage Angle")
plt.grid()
plt.show()
print(va_degree)

# Current results
i_ka_file = os.path.join(output_dir, "res_line", "i_ka.xlsx")
i_ka = pd.read_excel(i_ka_file, index_col=0)
i_ka.plot(label="i_ka")
plt.xlabel("time step")
plt.ylabel("current magnitude [kA]")
plt.title("Current Magnitude")
plt.grid()
plt.show()

# load results
load_file = os.path.join(output_dir, "res_load", "p_mw.xlsx")
load_p_mw = pd.read_excel(load_file, index_col=0)
load_p_mw.plot(label="p_mw")
plt.xlabel("time step")
plt.ylabel("p_mw")
plt.title("load power")
plt.grid()
plt.show()
print(load_p_mw)

# Print Bbus matrix for debugging
print(net._ppc['internal']['Bbus'].A)