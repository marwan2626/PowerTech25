"""
Code for the publication Reliability-Constrained Thermal Storage Sizing in
Multi-Energy Systems under Uncertainty
Optimization by Marwan Mostafa 

Data File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### PACKAGES ####
import pandapower as pp
import pandas as pd
import numpy as np
#### SCRIPTS ####
import griddata as gd

def run_dc_load_flow(Bbus, net, P_mw):
    # Convert P from MW to per unit
    P_pu = P_mw / net.sn_mva
    print(f"Converted P to per unit: {P_pu}")

    # Identify the slack bus (usually the bus connected to ext_grid)
    slack_bus_index = net.ext_grid.bus.iloc[0]
    print(f"Slack Bus Index: {slack_bus_index}")
    
    # Reduce P_pu by removing the slack bus power
    P_pu_reduced = np.delete(P_pu, slack_bus_index)
    
    # Remove slack bus row and column from Bbus to form the reduced Bbus
    Bbus_reduced = np.delete(Bbus, slack_bus_index, axis=0)
    Bbus_reduced = np.delete(Bbus_reduced, slack_bus_index, axis=1)
    
    # Solve for voltage angles (theta) in radians for the reduced system
    theta_reduced = np.linalg.solve(Bbus_reduced, P_pu_reduced)
    
    # Initialize theta_full to include all buses with slack bus set to 0 radians
    theta_full = np.zeros(len(net.bus), dtype=np.float64)

    
    # Insert the calculated angles for active buses into the original bus index space
    idx = 0
    for i in range(len(theta_full)):
        if i != slack_bus_index:
            theta_full[i] = theta_reduced[idx]
            idx += 1
    
    theta_full_degrees = (theta_full*180)/np.pi
    print(f"Theta Full (degrees): {theta_full_degrees}")
    
    # Initialize lists to store results
    line_pl_mw = []
    line_loading_percent = []
    line_current_mag = []
    transformer_pl_mw = []
    
    # Power flow calculations for each line
    for line in net.line.itertuples():
        from_bus = line.from_bus
        to_bus = line.to_bus
        base_voltage = net.bus.at[from_bus, 'vn_kv'] * 1e3  # Convert kV to V
        Z_base = base_voltage ** 2 / net.sn_mva  # Calculate base impedance
        Y_base = 1 / Z_base  # Calculate base admittance
        x_pu = line.x_ohm_per_km * line.length_km / Z_base
        r_pu = line.r_ohm_per_km * line.length_km / Z_base
        
        Y_series = 1 / (r_pu + 1j * x_pu)  # Series admittance
        print(f"Y_series: {Y_series}")

        # Calculate power flow in per unit
        power_flow_pu = (theta_full[from_bus] - theta_full[to_bus]) / x_pu
        # Convert to MW
        power_flow_mw = power_flow_pu * net.sn_mva /1e6
        line_pl_mw.append(power_flow_mw)
        
        # Calculate line current magnitude in kA
        current_mag_ka = abs(power_flow_mw) / (np.sqrt(3)*base_voltage / 1e3)
        line_current_mag.append(current_mag_ka)
        
        # Calculate line loading percent
        if hasattr(line, 'max_i_ka'):
            loading_percent = 100 * (current_mag_ka / line.max_i_ka)
        else:
            loading_percent = np.nan
            print(f"Warning: 'max_i_ka' not found in net.line. 'line_loading_percent' set to NaN.")
        
        line_loading_percent.append(loading_percent)

    # Power flow calculations for transformers
    for trafo in net.trafo.itertuples():
        hv_bus = trafo.hv_bus
        lv_bus = trafo.lv_bus

        # Transformer reactance in per unit (from its data)
        x_pu = (trafo.vk_percent / 100) / trafo.sn_mva

        # Power flow calculation for transformer (similar to a line)
        power_flow_pu = (theta_full[hv_bus] - theta_full[lv_bus]) / x_pu
        power_flow_mw = power_flow_pu * net.sn_mva / 1e6  #1e6 nicht nötig
        
        # Account for transformer losses (copper and iron losses)
        copper_losses_mw = (trafo.vkr_percent / 100) * net.sn_mva / 1e6  # Copper losses #1e6 nicht nötig
        iron_losses_mw = trafo.pfe_kw / 1000  # Iron losses (already in kW)

        # Correct external grid power (subtract losses)
        external_grid_power_mw = power_flow_mw - (copper_losses_mw + iron_losses_mw) #absolute needed because in some cases the transformer generated power

        transformer_pl_mw.append(external_grid_power_mw)  # Store the corrected external grid power

    
    # Convert lists to Pandas Series and store them in a dictionary
    results = {
        'theta_degrees': pd.Series(np.degrees(theta_full), index=net.bus.index),
        'line_pl_mw': pd.Series(line_pl_mw, index=net.line.index),
        'line_loading_percent': pd.Series(line_loading_percent, index=net.line.index),
        'line_current_mag': pd.Series(line_current_mag, index=net.line.index),
        'transformer_pl_mw': pd.Series(transformer_pl_mw, index=net.trafo.index)
    }
    
    return results

def manual_dc_timeseries(time_steps, net, const_pv, const_load, Ybus):
    results = {
        "time_step": [],
        "theta_degrees": [],
        "line_loading_percent": [],
        "line_current_mag": [],
        "load_p_mw": [],
        "sgen_p_mw": [],
        "line_pl_mw": []  # Ensure this key is included
    }

    line_indices = None

    for t in time_steps:
        # Update controls using const_pv and const_load
        const_pv.time_step(net, time=t)
        const_load.time_step(net, time=t)

        # Recalculate the power injection vector P immediately after the update
        P = np.zeros(len(net.bus), dtype=np.float64)
        if not net.load.empty:
            P[net.load.bus.values.astype(int)] -= net.load.p_mw.values.astype(np.float64)
        if not net.sgen.empty:
            P[net.sgen.bus.values.astype(int)] += net.sgen.p_mw.values.astype(np.float64)

        # Print load and generation values at each bus and time step
        #print(f"Time step {t}:")
        #print(f"Loads at each bus: {net.load.p_mw.values.tolist()}")
        #print(f"Generation at each bus: {net.sgen.p_mw.values.tolist()}")
        #print(f"Power Injection Vector (P): {P}")

        # Run the DC load flow calculation
        flow_results = run_dc_load_flow(Ybus, net, P)

        if line_indices is None:
            line_indices = flow_results['line_pl_mw'].index

        results["time_step"].append(t)
        results["theta_degrees"].append(flow_results['theta_degrees'])
        results["line_loading_percent"].append(flow_results['line_loading_percent'].tolist())
        results["line_current_mag"].append(flow_results['line_current_mag'].tolist())
        results["load_p_mw"].append(net.load.p_mw.values.tolist())
        results["sgen_p_mw"].append(net.sgen.p_mw.values.tolist())
        results["line_pl_mw"].append(flow_results['line_pl_mw'].tolist())

    theta_degrees_df = pd.DataFrame(results["theta_degrees"], index=results["time_step"], columns=net.bus.index)
    line_loading_percent_df = pd.DataFrame(results["line_loading_percent"], index=results["time_step"], columns=line_indices)
    line_current_mag_df = pd.DataFrame(results["line_current_mag"], index=results["time_step"], columns=line_indices)
    load_p_mw_df = pd.DataFrame(results["load_p_mw"], index=results["time_step"], columns=net.load.index)
    sgen_p_mw_df = pd.DataFrame(results["sgen_p_mw"], index=results["time_step"], columns=net.sgen.index)
    line_pl_mw_df = pd.DataFrame(results["line_pl_mw"], index=results["time_step"], columns=line_indices)

    results_df = pd.concat({
        "theta_degrees": theta_degrees_df,
        "line_loading_percent": line_loading_percent_df,
        "line_current_mag": line_current_mag_df,
        "load_p_mw": load_p_mw_df,
        "sgen_p_mw": sgen_p_mw_df,
        "line_pl_mw": line_pl_mw_df
    }, axis=1)

    results_df.to_excel("output_results.xlsx")
    
    return results_df


def calculate_bbus_matrix(net):
    # Identify the slack bus (usually the bus connected to ext_grid)
    slack_bus_index = net.ext_grid.bus.iloc[0]
    
    num_buses = len(net.bus)  # Include all buses, including the slack bus
    Bbus = np.zeros((num_buses, num_buses))  # Initialize Bbus matrix
    
    # Get system base values
    base_MVA = net.sn_mva  # System base MVA (typically 100 MVA)
    
    # Add line reactances
    for line in net.line.itertuples():
        from_bus = line.from_bus
        to_bus = line.to_bus
        
        # Convert reactance to per-unit (X_pu = X_ohm / (Base Voltage^2 / Base MVA))
        base_voltage = net.bus.at[from_bus, 'vn_kv']  # Base voltage for the from bus
        x_pu = line.x_ohm_per_km * line.length_km / ((base_voltage ** 2) / base_MVA)
        
        # Bbus off-diagonal elements
        Bbus[from_bus, to_bus] -= 1 / x_pu
        Bbus[to_bus, from_bus] -= 1 / x_pu
        
        # Bbus diagonal elements
        Bbus[from_bus, from_bus] += 1 / x_pu
        Bbus[to_bus, to_bus] += 1 / x_pu
    
    # Add transformer reactances
    for trafo in net.trafo.itertuples():
        hv_bus = trafo.hv_bus
        lv_bus = trafo.lv_bus
        
        # Transformer reactance in per-unit (already in per-unit based on the transformer's base values)
        x_pu = (trafo.vk_percent / 100) / trafo.sn_mva
    
        # Bbus off-diagonal elements
        Bbus[hv_bus, lv_bus] -= 1 / x_pu
        Bbus[lv_bus, hv_bus] -= 1 / x_pu
    
        # Bbus diagonal elements
        Bbus[hv_bus, hv_bus] += 1 / x_pu
        Bbus[lv_bus, lv_bus] += 1 / x_pu
      
    return Bbus



def manual_dc_timeseries2(time_steps, net, const_pv, const_load, Ybus):
    results = {
        "time_step": [],
        "theta_degrees": [],
        "line_loading_percent": [],
        "line_current_mag": [],
        "load_p_mw": [],
        "sgen_p_mw": [],
        "line_pl_mw": []  # Ensure this key is included
    }

    line_indices = None

    for t in time_steps:
        # Update controls using const_pv and const_load
        const_pv.time_step(net, time=t)
        const_load.time_step(net, time=t)

        # Recalculate the power injection vector P immediately after the update
        P = np.zeros(len(net.bus), dtype=np.float64)
        if not net.load.empty:
            P[net.load.bus.values.astype(int)] -= net.load.p_mw.values.astype(np.float64)
        if not net.sgen.empty:
            P[net.sgen.bus.values.astype(int)] += net.sgen.p_mw.values.astype(np.float64)

        # Print load and generation values at each bus and time step
        #print(f"Time step {t}:")
        #print(f"Loads at each bus: {net.load.p_mw.values.tolist()}")
        #print(f"Generation at each bus: {net.sgen.p_mw.values.tolist()}")
        #print(f"Power Injection Vector (P): {P}")

        # Run the DC load flow calculation
        flow_results = run_dc_load_flow(Ybus, net, P)

        if line_indices is None:
            line_indices = flow_results['line_pl_mw'].index

        results["time_step"].append(t)
        results["theta_degrees"].append(flow_results['theta_degrees'])
        results["line_loading_percent"].append(flow_results['line_loading_percent'].tolist())
        results["line_current_mag"].append(flow_results['line_current_mag'].tolist())
        results["load_p_mw"].append(net.load.p_mw.values.tolist())
        results["sgen_p_mw"].append(net.sgen.p_mw.values.tolist())
        results["line_pl_mw"].append(flow_results['line_pl_mw'].tolist())

    theta_degrees_df = pd.DataFrame(results["theta_degrees"], index=results["time_step"], columns=net.bus.index)
    line_loading_percent_df = pd.DataFrame(results["line_loading_percent"], index=results["time_step"], columns=line_indices)
    line_current_mag_df = pd.DataFrame(results["line_current_mag"], index=results["time_step"], columns=line_indices)
    load_p_mw_df = pd.DataFrame(results["load_p_mw"], index=results["time_step"], columns=net.load.index)
    sgen_p_mw_df = pd.DataFrame(results["sgen_p_mw"], index=results["time_step"], columns=net.sgen.index)
    line_pl_mw_df = pd.DataFrame(results["line_pl_mw"], index=results["time_step"], columns=line_indices)

    results_df = pd.concat({
        "theta_degrees": theta_degrees_df,
        "line_loading_percent": line_loading_percent_df,
        "line_current_mag": line_current_mag_df,
        "load_p_mw": load_p_mw_df,
        "sgen_p_mw": sgen_p_mw_df,
        "line_pl_mw": line_pl_mw_df
    }, axis=1)

    results_df.to_excel("output_results.xlsx")
    
    return results_df


def get_electricity_price(filepath):
    # Read the electricity price data
    electricity_price = pd.read_csv(filepath)
    #print(f"Electricity Price Data: {electricity_price}")
    return electricity_price
    