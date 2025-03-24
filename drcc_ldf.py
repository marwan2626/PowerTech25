###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### PACKAGES ####
import gurobipy as gp
from gurobipy import GRB
import pickle as pkl
import numpy as np
import copy
import pandas as pd

#### SCRIPTS ####
import parameters as par
import data as dt 
import griddata as gd
import results as rs
import ldf_lc as ldf

###############################################################################
## Calculate Variance Propagation ##
###############################################################################

def accumulate_downstream_power(A, P_mw, Q_mw, net, downstream_map):
    """Accumulates downstream power flows using the correct downstream node mapping."""
    num_buses = A.shape[0]

    # Initialize accumulated power with the nodal values
    P_accumulated = P_mw.copy()
    Q_accumulated = Q_mw.copy()

    # Identify slack bus
    slack_bus_index = net.ext_grid.bus.iloc[0]

    # Traverse each bus and accumulate power from its downstream buses
    for bus in range(num_buses):
        if bus == slack_bus_index:
            continue  # Skip slack bus

        for child_bus in downstream_map[bus]:  # All downstream buses
            P_accumulated[bus] += P_mw[child_bus]
            Q_accumulated[bus] += Q_mw[child_bus]

    return P_accumulated, Q_accumulated


def run_lindistflow(net, downstream_map, P_mw, Q_mw, Ybus, A, Z):
    """Runs the LinDistFlow calculation with correct power accumulation, including transformers."""

    import numpy as np
    import pandas as pd

    # Convert power injections to per unit
    S_pu = (P_mw + 1j * Q_mw) / net.sn_mva  

    # Identify the slack bus
    slack_bus_index = net.ext_grid.bus.iloc[0]

    # Compute voltage magnitudes using existing working approach
    Ybus_reduced = np.delete(np.delete(Ybus, slack_bus_index, axis=0), slack_bus_index, axis=1)
    S_pu_reduced = np.delete(S_pu, slack_bus_index)
    
    Zbus_reduced = np.linalg.inv(Ybus_reduced)
    R = np.real(Zbus_reduced)
    X = np.imag(Zbus_reduced)

    V_reduced = np.ones(len(S_pu_reduced)) + R @ np.real(S_pu_reduced) + X @ np.imag(S_pu_reduced)

    # Reconstruct full voltage vector
    V_nodes = np.ones(len(net.bus))
    idx = 0
    for i in range(len(V_nodes)):
        if i != slack_bus_index:
            V_nodes[i] = V_reduced[idx]
            idx += 1

    # Compute accumulated power flows
    P_accumulated, Q_accumulated = accumulate_downstream_power(A, P_mw, Q_mw, net, downstream_map)

    # === Compute P_branch and Q_branch for Lines and Transformers ===
    num_lines = len(net.line)
    num_trafo = len(net.trafo)

    P_branch = np.zeros(num_lines)
    Q_branch = np.zeros(num_lines)


    for line_idx in range(num_lines):
        from_bus = net.line.from_bus[line_idx]
        to_bus = net.line.to_bus[line_idx]

        # Accumulated Power at Receiving End
        P_recv = P_accumulated[to_bus]
        Q_recv = Q_accumulated[to_bus]

        # Extract Line Impedance from Z
        line_z = Z[line_idx]  # Line impedance
        R_line = np.real(line_z)
        X_line = np.imag(line_z)

        # Estimated To-Bus Voltage (Avoid divide-by-zero)
        V_to = max(V_nodes[to_bus], 1e-4)

        # Compute Line Losses
        P_loss_line = R_line * (P_recv**2 + Q_recv**2) / V_to**2
        Q_loss_line = X_line * (P_recv**2 + Q_recv**2) / V_to**2

        # Estimate From-Bus Power (Sending End)
        P_send = P_recv - P_loss_line
        Q_send = Q_recv - Q_loss_line

        # Store results
        P_branch[line_idx] = P_send
        Q_branch[line_idx] = Q_send

    # Transformer Power
    P_trafo = np.zeros(num_trafo)
    Q_trafo = np.zeros(num_trafo)
    S_trafo = np.zeros(num_trafo)
    Trafo_loading_percent = np.zeros(num_trafo)



    for trafo_idx in range(num_trafo):
        lv_bus = net.trafo.lv_bus[trafo_idx]
        hv_bus = net.trafo.hv_bus[trafo_idx]

        # Accumulated Power at LV Side
        P_LV = P_accumulated[lv_bus]
        Q_LV = Q_accumulated[lv_bus]

        # Extract Transformer Impedance from Z
        trafo_z = Z[len(net.line) + trafo_idx]  # Transformer impedance is after lines in Z
        R_trafo = np.real(trafo_z)
        X_trafo = np.imag(trafo_z)

        # Estimated LV Voltage (Avoid divide-by-zero)
        V_LV = max(V_nodes[lv_bus], 1e-4) 

        # Compute Transformer Losses
        P_loss_trafo = R_trafo * (P_LV**2 + Q_LV**2) / V_LV**2
        Q_loss_trafo = X_trafo * (P_LV**2 + Q_LV**2) / V_LV**2

        # Estimate HV-Side Power
        P_HV = P_LV - P_loss_trafo
        Q_HV = Q_LV - Q_loss_trafo

        # Store results
        P_trafo[trafo_idx] = P_HV  
        Q_trafo[trafo_idx] = Q_HV  
        S_trafo[trafo_idx] = np.sqrt(P_HV**2 + Q_HV**2)

        # Compute loading percentage
        S_rated = net.trafo.sn_mva.iloc[trafo_idx]  # Transformer rated power
        Trafo_loading_percent[trafo_idx] = (S_trafo[trafo_idx] / S_rated) * 100

    # Compute complex power flow for lines
    S_branch = P_branch + 1j * Q_branch  # MW + jMVar

    # Compute current only for lines
    I_branch_pu = np.zeros(num_lines, dtype=complex)
    for line_idx in range(num_lines):
        from_bus = net.line.from_bus[line_idx]
        I_branch_pu[line_idx] = np.conj(S_branch[line_idx]) / V_nodes[from_bus]  # I = S*/V_from

    # Convert current to kA (using base voltage)
    V_base_kv = net.bus.vn_kv.iloc[1]  # Just take the first value
    I_branch_ka = I_branch_pu * (net.sn_mva / (np.sqrt(3) * V_base_kv))  # Convert to kA

    # Create final results dictionary
    results = {
        'V_magnitude': pd.Series(np.abs(V_nodes), index=net.bus.index),  # p.u.
        'P_node': pd.Series(P_mw, index=net.bus.index),  # MW
        'Q_node': pd.Series(Q_mw, index=net.bus.index),  # MVar
        'P_line_flow': pd.Series(P_branch, index=net.line.index),  # MW
        'Q_line_flow': pd.Series(Q_branch, index=net.line.index),  # MVar
        'P_trafo_flow': pd.Series(P_trafo, index=net.trafo.index) if num_trafo > 0 else pd.Series([], dtype=np.float64),  # MW
        'Q_trafo_flow': pd.Series(Q_trafo, index=net.trafo.index) if num_trafo > 0 else pd.Series([], dtype=np.float64),  # MVar
        'S_trafo': pd.Series(S_trafo, index=net.trafo.index) if num_trafo > 0 else pd.Series([], dtype=np.float64),  # MVA
        'Trafo_loading_percent': pd.Series(Trafo_loading_percent, index=net.trafo.index) if num_trafo > 0 else pd.Series([], dtype=np.float64),  # %
        'I_branch': pd.Series(np.abs(I_branch_ka), index=net.line.index)  # kA (ONLY FOR LINES)
    }

    return results


def calculate_variance_propagation(time_steps, net):
    results = {
        "time_step": [],
        "V_magnitude": [],
        "P_node": [],
        "Q_node": [],
        "line_pl_mw": [],
        "line_ql_mvar": [],
        "trafo_pl_mw": [],
        "trafo_ql_mvar": [],
        "S_trafo": [],  # Apparent power of transformer
        "Trafo_loading_percent": [],  # Transformer loading percentage
        "I_branch": [],
        "load_p_mw": [],
        "sgen_p_mw": []
    }


    Z = ldf.calculate_z_matrix(net)
    Gbus = ldf.calculate_gbus_matrix(net)
    Bbus = ldf.calculate_bbus_matrix(net)
    Ybus = ldf.compute_Ybus(Gbus, Bbus)
    A = ldf.compute_incidence_matrix(net)
    # Get correct downstream mappings
    downstream_map = ldf.compute_downstream_nodes(A, net)
    # Retrieve all controllers from net
    controllers = net.controller
    print("extracted controllers")


    for t in time_steps:

        # Apply all active controllers
        if not controllers.empty:
            for _, controller in controllers.iterrows():
                controller.object.time_step(net, time=t)  # Dynamically apply controllers

        # Compute power injections
        P = np.zeros(len(net.bus), dtype=np.float64)
        Q = np.zeros(len(net.bus), dtype=np.float64)
        
        if not net.load.empty:
            for i, bus in enumerate(net.load.bus.values.astype(int)):
                P[bus] -= net.load.p_mw.iloc[i]
                Q[bus] -= net.load.q_mvar.iloc[i]

        if not net.sgen.empty:
            for i, bus in enumerate(net.sgen.bus.values.astype(int)):
                P[bus] += net.sgen.p_mw.iloc[i]
                Q[bus] += net.sgen.q_mvar.iloc[i]

        # Run LinDistFlow calculation with corrected Z matrix
        flow_results = run_lindistflow(net, downstream_map, P, Q, Ybus, A, Z)

        # Extract relevant results
        I_branch = flow_results["I_branch"]

        num_lines = len(net.line)
        results["I_branch"].append(I_branch[:num_lines])

        results["time_step"].append(t)
        results["V_magnitude"].append(flow_results["V_magnitude"])
        results["P_node"].append([-p for p in flow_results["P_node"]])  # Reverse direction
        results["Q_node"].append([-q for q in flow_results["Q_node"]])  # Reverse direction
        results["line_pl_mw"].append([-p for p in flow_results["P_line_flow"]])  # Reverse direction
        results["line_ql_mvar"].append([-q for q in flow_results["Q_line_flow"]])  # Reverse direction
        results["trafo_pl_mw"].append([-p for p in flow_results["P_trafo_flow"]])  # Reverse direction
        results["trafo_ql_mvar"].append([-q for q in flow_results["Q_trafo_flow"]])  # Reverse direction
        results["S_trafo"].append(flow_results["S_trafo"])
        results["Trafo_loading_percent"].append(flow_results["Trafo_loading_percent"])
        results["load_p_mw"].append(net.load.p_mw.values.tolist())
        results["sgen_p_mw"].append(net.sgen.p_mw.values.tolist())

    # Convert results into DataFrames
    V_magnitude_df = pd.DataFrame(results["V_magnitude"], index=results["time_step"], columns=net.bus.index)

    P_node_df = pd.DataFrame(results["P_node"], index=results["time_step"], columns=net.bus.index)
    Q_node_df = pd.DataFrame(results["Q_node"], index=results["time_step"], columns=net.bus.index)

    load_p_mw_df = pd.DataFrame(results["load_p_mw"], index=results["time_step"], columns=net.load.index)
    sgen_p_mw_df = pd.DataFrame(results["sgen_p_mw"], index=results["time_step"], columns=net.sgen.index)

    line_indices = net.line.index
    trafo_indices = net.trafo.index

    line_pl_mw_df = pd.DataFrame(results["line_pl_mw"], index=results["time_step"], columns=line_indices)
    line_ql_mvar_df = pd.DataFrame(results["line_ql_mvar"], index=results["time_step"], columns=line_indices)
    trafo_pl_mw_df = pd.DataFrame(results["trafo_pl_mw"], index=results["time_step"], columns=trafo_indices)
    trafo_ql_mvar_df = pd.DataFrame(results["trafo_ql_mvar"], index=results["time_step"], columns=trafo_indices)
    
    S_trafo_df = pd.DataFrame(results["S_trafo"], index=results["time_step"], columns=trafo_indices)
    Trafo_loading_df = pd.DataFrame(results["Trafo_loading_percent"], index=results["time_step"], columns=trafo_indices)

    I_branch_df = pd.DataFrame(results["I_branch"], index=results["time_step"], columns=line_indices)

    # Combine all DataFrames
    results_df = pd.concat({
        "V_magnitude": V_magnitude_df,
        "P_node": P_node_df,
        "Q_node": Q_node_df,
        "load_p_mw": load_p_mw_df,
        "sgen_p_mw": sgen_p_mw_df,
        "line_pl_mw": line_pl_mw_df,
        "line_ql_mvar": line_ql_mvar_df,
        "trafo_pl_mw": trafo_pl_mw_df,
        "trafo_ql_mvar": trafo_ql_mvar_df,
        "S_trafo": S_trafo_df,
        "Trafo_loading_percent": Trafo_loading_df,
        "I_branch": I_branch_df
    }, axis=1)

    return results_df

###############################################################################
## Main Function ##
###############################################################################

def solve_drcc_opf(net, time_steps, electricity_price, const_pv, const_load_household_P, const_load_household_Q, const_load_heatpump, T_amb):
    #variance_net= gd.setup_grid_IAS_variance(season)
    #var_results = calculate_variance_propagation(time_steps, variance_net)
    #var_p_mw_t0 = var_results.loc[0, ("line_pl_mw")]
    #print(f"Variance Propagation Results at time 0: {var_p_mw_t0}")
    # Filter all line_pl_mw values across all time steps
    #var_P_line_dict = {}
    #var_P_trafo_dict = {}

    # for time_step in var_results.index:
    #     # Access the line_pl_mw and transformer_pl_mw for the current time step
    #     var_P_line = var_results.loc[time_step, ("line_pl_mw")]
    #     var_P_trafo = var_results.loc[time_step, ("transformer_pl_mw")]

    #     # Apply the filter: set values less than 1e-6 to 0
    #     var_P_line = var_P_line.where(var_P_line.abs() >= 1e-6, 0)
    #     #var_P_trafo = var_P_trafo.where(var_P_trafo.abs() >= 1e-6, 0)

    #     # Save the filtered results in a nested dictionary
    #     var_P_line_dict[time_step] = var_P_line.to_dict()
    #     var_P_trafo_dict[time_step] = var_P_trafo.to_dict()
        
    # Access the results as var_P_line_dict[t][line]
    #print(f"Var Line Value, {var_P_line_dict}")
    #print(f"Var Trafo Value, {var_P_trafo_dict}")

    pd.set_option('display.precision', 10)
    model = gp.Model("opf_with_ldf_lc")

    # Define the costs
    curtailment_cost = par.curtailment_cost  
    storage_cost = par.c_cost
    r = 0.05 #interest rate
    n= 20 #lifetime of the storage

    storage_cost_levelized = storage_cost * ((r*(1+r)**n) / (((1+r)**n) - 1))
    #print(f"Levelized Storage Cost: {storage_cost_levelized}")

    HNS_price = par.HNS_cost

    ### Define the variables ###
    epsilon = 100e-9  # Small positive value to ensure some external grid usage

    # Extract transformer capacity in MW (assuming sn_mva is in MVA)
    transformer_capacity_mw = net.trafo['sn_mva'].values[0]
    #print(f"Transformer Capacity: {transformer_capacity_mw}")


    # Initialize decision variables
    pv_gen_vars = {}  # Store PV generation decision variables
    ext_grid_import_P_vars = {}  # Store external grid import power decision variables
    ext_grid_import_Q_vars = {}  # Store external grid import power decision variables
    ext_grid_export_P_vars = {}  # Store external grid export power decision variables
    ext_grid_export_Q_vars = {}  # Store external grid export power decision variables
    V_vars = {}  # Store voltage angle decision variables (radians)
    curtailment_vars = {} # Store decision variables for curtailment
    flexible_load_P_vars = {}  # New flexible load variables
    flexible_load_Q_vars = {}  # New flexible load variables
    P_branch_vars = {}  # Store line power flow decision variables
    Q_branch_vars = {}  # Store line power flow decision variables
    Line_loading_vars = {}  # Store line loading decision variables
    P_trafo_vars = {}  # Store transformer loading decision variables
    Q_trafo_vars = {}  # Store transformer loading decision variables
    S_trafo_vars = {}  # Store transformer loading decision variables
    transformer_loading_vars = {}  # Store transformer loading percentage decision variables 
    transformer_loading_perc_vars = {}  # Store transformer loading percentage decision variables  
    ts_capacity_vars = {}  # Store thermal storage capacity decision variables
    eta_pl_vars = {}  # Store partload electrical efficiency of heat pumps
    HNS_vars = {}  # Store HNS variables

    # Add thermal storage variables
    ts_in_vars = {t: {} for t in time_steps}
    ts_out_vars = {t: {} for t in time_steps}
    ts_sof_vars = {t: {} for t in time_steps}
    energy_stored_vars = {t: {} for t in time_steps}    

    slack_bus_index = net.ext_grid.bus.iloc[0]

    Z = ldf.calculate_z_matrix(net)
    Gbus = ldf.calculate_gbus_matrix(net)
    Bbus = ldf.calculate_bbus_matrix(net)
    Ybus = ldf.compute_Ybus(Gbus, Bbus)
    A = ldf.compute_incidence_matrix(net)
    downstream_map = ldf.compute_downstream_nodes(A, net)
    # Get correct downstream mappings
    downstream_map = ldf.compute_downstream_nodes(A, net)
    # Retrieve all controllers from net

    Ybus_reduced = np.delete(np.delete(Ybus, slack_bus_index, axis=0), slack_bus_index, axis=1)


    # Dictionaries to store results
    pv_gen_results = {}
    flexible_load_P_results = {}
    flexible_load_Q_results = {}
    non_flexible_load_P_results = {}
    non_flexible_load_Q_results = {}
    HNS_results = {}
    load_Q_results = {}
    ext_grid_import_P_results = {}
    ext_grid_import_Q_results = {}
    ext_grid_export_P_results = {}
    ext_grid_export_Q_results = {}
    V_results = {}
    line_pl_results = {}
    line_ql_results = {}
    line_current_results = {}
    line_loading_results = {}
    transformer_loading_results = {}
    ts_capacity_results = {}
    ts_in_results = {}
    ts_out_results = {}
    ts_sof_results = {}

    # Temporary dictionary to store updated load values per time step
    flexible_time_synchronized_loads_P = {t: {} for t in time_steps}
    flexible_time_synchronized_loads_Q = {t: {} for t in time_steps}
    non_flexible_time_synchronized_loads_P = {t: {} for t in time_steps}
    non_flexible_time_synchronized_loads_Q = {t: {} for t in time_steps}

    # Identify buses with flexible loads
    flexible_load_buses = list(set(net.load[net.load['controllable'] == True].bus.values))
    non_flexible_load_buses = list(set(net.load[net.load['controllable'] == False].bus.values))
    #print(f"Flexible load buses: {flexible_load_buses}")

    # Add thermal storage variables
    ts_capacity_vars = model.addVars(
        flexible_load_buses,
        lb=0.0001,
        name=f'ts_capacity'
    )

    heat_demand_scaling = 1 / par.tsnet_eff
    COP = {}  # Coefficient of Performance (COP) for heat pumps
    for t in time_steps:
        COP[t] = par.eta_c0 * (T_amb[t] + par.DeltaT)/(par.T_S - T_amb[t] + 2*par.DeltaT)
    #print(f"COP: {COP}")

    # Add variables for each time step
    for t in time_steps:
        # Update const_pv and const_load for this time step
        const_load_heatpump.time_step(net, time=t)
        const_load_household_P.time_step(net, time=t)
        const_load_household_Q.time_step(net, time=t)
        const_pv.time_step(net, time=t)


        # Initialize dictionaries for time-synchronized loads
        flexible_time_synchronized_loads_P[t] = {}
        flexible_time_synchronized_loads_Q[t] = {}
        non_flexible_time_synchronized_loads_P[t] = {}
        non_flexible_time_synchronized_loads_Q[t] = {}

        # Iterate over all loads
        for load in net.load.itertuples():
            bus = load.bus
            if load.controllable:
                # Flexible load
                flexible_time_synchronized_loads_P[t][bus] = (
                    flexible_time_synchronized_loads_P[t].get(bus, 0.0) + load.p_mw
                )
                flexible_time_synchronized_loads_Q[t][bus] = (
                    flexible_time_synchronized_loads_Q[t].get(bus, 0.0) + load.q_mvar
                )
            else:
                # Non-flexible load
                non_flexible_time_synchronized_loads_P[t][bus] = (
                    non_flexible_time_synchronized_loads_P[t].get(bus, 0.0) + load.p_mw
                )
                non_flexible_time_synchronized_loads_Q[t][bus] = (
                    non_flexible_time_synchronized_loads_Q[t].get(bus, 0.0) + load.q_mvar
                )


        # Ensure all buses have an entry, even if no loads are connected
        for bus in net.bus.index:
            if bus not in flexible_time_synchronized_loads_P[t]:
                flexible_time_synchronized_loads_P[t][bus] = 0.0
            if bus not in flexible_time_synchronized_loads_Q[t]:
                flexible_time_synchronized_loads_Q[t][bus] = 0.0
            if bus not in non_flexible_time_synchronized_loads_P[t]:
                non_flexible_time_synchronized_loads_P[t][bus] = 0.0
            if bus not in non_flexible_time_synchronized_loads_Q[t]:
                non_flexible_time_synchronized_loads_Q[t][bus] = 0.0

        # Extract the bus indices where PV generators are connected (from net.sgen.bus)
        # Ensure unique PV buses
        pv_buses = list(set(net.sgen.bus.values))

        # Define PV generation upper bounds per bus
        pv_bus_limits = {bus: net.sgen.loc[net.sgen.bus == bus, 'p_mw'].sum() for bus in pv_buses}

        # Create PV generation variables for this time step
        if len(pv_buses) > 0:
            pv_gen_vars[t] = model.addVars(pv_buses, lb=0, ub=pv_bus_limits, name=f'pv_gen_{t}')
            curtailment_vars[t] = model.addVars(pv_buses, lb=0, ub=pv_bus_limits, name=f'curtailment_{t}')

            for bus in pv_buses:
                sgen_index = net.sgen.index[net.sgen.bus == bus].tolist()[0]  # Get first occurrence
                model.addConstr(
                    curtailment_vars[t][bus] == net.sgen.at[sgen_index, 'p_mw'] - pv_gen_vars[t][bus], 
                    name=f'curtailment_constraint_{t}_{bus}'
                )
            
        # External grid power variables for import and export at the slack bus (bus 0)
        ext_grid_import_P_vars[t] = model.addVar(lb=0, name=f'ext_grid_import_P_{t}')  # Import is non-negative
        ext_grid_import_Q_vars[t] = model.addVar(lb=0, name=f'ext_grid_import_Q_{t}')  # Import is non-negative
        ext_grid_export_P_vars[t] = model.addVar(lb=0, name=f'ext_grid_export_P_{t}')  # Export is non-negative
        ext_grid_export_Q_vars[t] = model.addVar(lb=0, name=f'ext_grid_export_Q_{t}')  # Export is non-negative

        model.addConstr(ext_grid_import_P_vars[t] + ext_grid_export_Q_vars[t] >= epsilon, name=f'nonzero_ext_grid_P_usage_{t}')
        model.addConstr(ext_grid_import_Q_vars[t] + ext_grid_export_Q_vars[t] >= epsilon, name=f'nonzero_ext_grid_Q_usage_{t}')

        # Define flexible load variables with global peak limit (par.hp_max_power)
        flexible_load_P_vars[t] = model.addVars(
            flexible_load_buses,
            ub=par.hp_max_power,
            name=f'flexible_load_P_{t}'
        )

        flexible_load_Q_vars[t] = model.addVars(
            flexible_load_buses,
            name=f'flexible_load_Q_{t}'
        )

        # Define HNS variables with global peak limit (par.hp_max_power)
        HNS_vars[t] = model.addVars(
            flexible_load_buses,
            lb=0,
            ub=par.hp_max_power,
            name=f'HNS_{t}'
        )

        eta_pl_vars[t] = model.addVars(
            flexible_load_buses,
            lb=0,
            name=f'partload_effeciency_{t}'
        )
                            
       
    # Update SOF constraints for each flexible load bus
    for t_idx, t in enumerate(time_steps):
        for bus in flexible_load_buses:
            ts_in_vars[t][bus] = model.addVar(lb=0, ub=par.ts_in_max, name=f'ts_in_{t}_{bus}')
            ts_out_vars[t][bus] = model.addVar(lb=0, ub=par.ts_out_max, name=f'ts_out_{t}_{bus}')
            ts_sof_vars[t][bus] = model.addVar(lb=0, ub=1.0, name=f'ts_sof_{t}_{bus}')  # SOF as percentage (0 to 1)
            energy_stored_vars[t][bus] = model.addVar(lb=0, name=f'energy_stored_{t}_{bus}')

    
    non_slack_buses = [bus for bus in net.bus.index if bus != slack_bus_index]

    
    V_vars = model.addVars(time_steps, net.bus.index, name="V")
    V_reduced_vars = model.addVars(time_steps, non_slack_buses, name="V_reduced")
    # Set slack bus voltage to 1.0 p.u. at all time steps
    for t in time_steps:
        model.addConstr(V_vars[t, slack_bus_index] == 1.0, name=f"slack_voltage_fixed_{t}")

    P_branch_vars = model.addVars(time_steps, net.line.index, lb=-GRB.INFINITY, name="P_branch")
    Q_branch_vars = model.addVars(time_steps, net.line.index, lb=-GRB.INFINITY, name="Q_branch")
    S_branch_vars = model.addVars(time_steps, net.line.index, lb=0, name="S_branch")

    P_trafo_vars = model.addVars(time_steps, net.line.index, lb=-GRB.INFINITY, name="P_trafo")
    Q_trafo_vars = model.addVars(time_steps, net.line.index, lb=-GRB.INFINITY, name="Q_trafo")
    S_trafo_vars = model.addVars(time_steps, net.line.index, lb=0, name="S_trafo")  

    #Transformer loading percentage
    transformer_loading_perc_vars = model.addVars(time_steps, net.trafo.index, lb=0, name="Trafo_loading_percent")
    Line_loading_vars = model.addVars(time_steps, net.line.index, name="Line_loading")
    # Initialize as a structured dictionary of linear expressions
    Line_loading_expr = {}
    # Define expressions for each time step and line
    for t in time_steps:
        for line_idx in net.line.index:
            Line_loading_expr[t, line_idx] = gp.LinExpr()  # Properly initialize each entry
    S_branch_approx_expr = {}
    for t in time_steps:
        for line_idx in net.line.index:
            S_branch_approx_expr[t, line_idx] = gp.LinExpr()



    # Accumulated power at each bus
    P_accumulated_vars = model.addVars(time_steps, net.bus.index, lb=-GRB.INFINITY, name="P_accumulated")
    Q_accumulated_vars = model.addVars(time_steps, net.bus.index, lb=-GRB.INFINITY, name="Q_accumulated")

    # Add power balance and load flow constraints for each time step
    for t in time_steps:
        # Power injection vector P
        P_injected = {bus: gp.LinExpr() for bus in net.bus.index}
        Q_injected = {bus: gp.LinExpr() for bus in net.bus.index}

        for i, bus in enumerate(net.bus.index):
            if bus in net.load.bus.values:
                if bus in flexible_load_buses:
                    # Define the heat demand for the bus, representing the electrical equivalent
                    heat_demand = flexible_time_synchronized_loads_P[t][bus]  # or load profile representing heat demand
                    eta_pl_vars[t][bus] = 1 - (par.psi*(1-(flexible_load_P_vars[t][bus]/par.hp_max_power))) 

                    # 1. Heat Demand Coverage: flexible_load_vars and/or thermal storage must meet the demand (all in electrical equivalent)
                    model.addConstr(
                        flexible_load_P_vars[t][bus] + ((ts_out_vars[t][bus] - ts_in_vars[t][bus]) / COP[t]) == (heat_demand * heat_demand_scaling) - HNS_vars[t][bus],
                        name=f'heat_demand_coverage_{t}_{bus}'
                    )

                    # 3. Storage Charging: use excess power for storage charging if available
                    model.addConstr(
                        ts_in_vars[t][bus] <= flexible_load_P_vars[t][bus] * COP[t] * eta_pl_vars[t][bus],
                        name=f'storage_charging_{t}_{bus}'
                    )

                    # model.addConstr(
                    #     ts_in_vars[t][bus] <= (ts_capacity_vars[bus] - energy_stored_vars[t][bus]),
                    #     name=f'storage_input_limit_{t}_{bus}'
                    # )
                    model.addConstr(
                        ts_out_vars[t][bus] <= energy_stored_vars[t][bus],
                        name=f'storage_output_limit_{t}_{bus}'
                    )

                    # Update the state of fill (SOF) of the storage
                    # Initial SOF Constraint for the first timestep
                    if t == 0:
                        model.addConstr(
                            energy_stored_vars[t][bus] == par.ts_sof_init * ts_capacity_vars[bus],
                            name=f'initial_energy_{bus}'
                        )
                    else:
                        model.addConstr(
                            energy_stored_vars[t][bus] ==
                            energy_stored_vars[time_steps[t - 1]][bus] +
                            (par.ts_eff * ts_in_vars[t][bus]) -
                            (ts_out_vars[t][bus] / par.ts_eff) -
                            (par.ts_alpha * energy_stored_vars[time_steps[t - 1]][bus] * (par.T_S-T_amb[t])),
                            name=f'storage_energy_update_{t}_{bus}'
                        )

                    model.addConstr(
                        ts_sof_vars[t][bus] * ts_capacity_vars[bus] == energy_stored_vars[t][bus],
                        name=f'sof_definition_{t}_{bus}'
                    )

                    model.addConstr(
                        flexible_load_Q_vars[t][bus] == flexible_load_P_vars[t][bus] * par.Q_scaling,
                        name=f'flexible_load_Q_{t}_{bus}'
                    )
                    # Use the flexible load variable for controllable loads
                    P_injected[bus] -= flexible_load_P_vars[t][bus]
                    Q_injected[bus] += flexible_load_Q_vars[t][bus]

                if bus in non_flexible_load_buses:
                    # For non-flexible loads, use the time-synchronized load
                    P_injected[bus] -= non_flexible_time_synchronized_loads_P[t][bus]
                    Q_injected[bus] -= non_flexible_time_synchronized_loads_Q[t][bus]

            if len(pv_buses) > 0 and bus in pv_buses:
                if bus in pv_buses:
                    # Only add PV generation if the bus has PV (i.e., in net.sgen.bus)
                    P_injected[bus] += pv_gen_vars[t][bus]

        model.update()

        #for bus in net.bus.index:
            #print(f"Time step {t}, Bus {bus}: Power injected (MW) = {P_injected[bus]}")

        # Convert P_injected to per unit
        P_pu = {bus: P_injected[bus] / net.sn_mva for bus in net.bus.index}
        Q_pu = {bus: Q_injected[bus] / net.sn_mva for bus in net.bus.index}

        # Compute the reduced impedance matrices
        Zbus_reduced = np.linalg.inv(Ybus_reduced)
        R = np.real(Zbus_reduced)
        X = np.imag(Zbus_reduced)

        # Define voltage magnitude constraints using correct indexing
        for i, bus in enumerate(non_slack_buses):
            model.addConstr(
                V_reduced_vars[t, bus] == 1 +
                gp.quicksum(R[i, j] * P_pu[non_slack_buses[j]] for j in range(len(non_slack_buses))) +
                gp.quicksum(X[i, j] * Q_pu[non_slack_buses[j]] for j in range(len(non_slack_buses))),
                name=f"voltage_magnitude_{t}_{bus}"
        )

        # Map V_reduced_vars to V_vars for non-slack buses
        for bus in non_slack_buses:
                model.addConstr(V_vars[t, bus] == V_reduced_vars[t, bus], name=f"voltage_assignment_{t}_{bus}")
        
        # model.addConstr(
        #     ext_grid_import_P_vars[t] - ext_grid_export_P_vars[t] ==
        #     gp.quicksum(P_injected[bus] for bus in net.bus.index),
        #     name=f"P_balance_slack_{t}"
        # )
        # model.addConstr(
        #     ext_grid_import_Q_vars[t] - ext_grid_export_Q_vars[t] ==
        #     gp.quicksum(Q_injected[bus] for bus in net.bus.index),
        #     name=f"Q_balance_slack_{t}"
        #)
        model.addConstr(
            (ext_grid_import_P_vars[t] * ext_grid_export_P_vars[t]) == 0, 
            name=f"import_export_exclusivity_P_{t}"
        )
        model.addConstr(
            (ext_grid_import_Q_vars[t] * ext_grid_export_Q_vars[t]) == 0, 
            name=f"import_export_exclusivity_Q_{t}"
        )


        # Accumulate power for each bus (excluding slack)
        sorted_buses = sorted(net.bus.index, key=lambda bus: len(downstream_map[bus]))  # Sort from leaves to root
        #print(f"Sorted Buses: {sorted_buses}")

        for bus in sorted_buses:
            if bus != slack_bus_index:
                #print(f"Bus {bus}: P_accumulated = P_injected[{bus}] + sum of downstream buses {downstream_map[bus]}")
                # Ensure it starts with its own injection
                model.addConstr(
                    P_accumulated_vars[t, bus] == P_injected[bus] + 
                    gp.quicksum(P_injected[child_bus] for child_bus in downstream_map[bus]),
                    name=f"P_accumulated_{t}_{bus}"
                )
                model.addConstr(
                    Q_accumulated_vars[t, bus] == Q_injected[bus] + 
                    gp.quicksum(Q_injected[child_bus] for child_bus in downstream_map[bus]),
                    name=f"Q_accumulated_{t}_{bus}"
                )

    # Enforce final state of fill to match initial state (0.5) for all flexible load buses
    for bus in flexible_load_buses:
        model.addConstr(ts_sof_vars[time_steps[-1]][bus] == 0.5, name=f'final_sof_balance_{bus}')


    #Line power flow and loading constraints (with the corrected expression)
    for t in time_steps:

        for line in net.line.itertuples():
            line_idx = line.Index  # Extract correct index
            from_bus = line.from_bus
            to_bus = line.to_bus

            # Compute Sending-End Power
            model.addConstr(
                P_branch_vars[t, line_idx] == P_accumulated_vars[t, to_bus],
                name=f"P_send_calc_{line_idx}"
            )

            model.addConstr(
                Q_branch_vars[t, line_idx] == Q_accumulated_vars[t, to_bus],
                name=f"Q_send_calc_{line_idx}"
            )

            # Enforce |P| = max(P, -P)
            # Force correct absolute value using big-M formulation
            # model.addConstr(P_abs_vars[t, line_idx] >= P_branch_vars[t, line_idx], name=f"P_abs_pos_{t}_{line_idx}")
            # model.addConstr(P_abs_vars[t, line_idx] >= -P_branch_vars[t, line_idx], name=f"P_abs_neg_{t}_{line_idx}")

            # Enforce that exactly one of the conditions holds
            # model.addConstr(P_abs_vars[t, line_idx] <= P_branch_vars[t, line_idx] + M * (1 - P_sign[t, line_idx]),
            #                 name=f"P_abs_helper1_{t}_{line_idx}")
            # model.addConstr(P_abs_vars[t, line_idx] <= -P_branch_vars[t, line_idx] + M * P_sign[t, line_idx],
            #                 name=f"P_abs_helper2_{t}_{line_idx}")
            
            # # Force correct absolute value using big-M formulation for Q
            # model.addConstr(Q_abs_vars[t, line_idx] >= Q_branch_vars[t, line_idx], name=f"Q_abs_pos_{t}_{line_idx}")
            # model.addConstr(Q_abs_vars[t, line_idx] >= -Q_branch_vars[t, line_idx], name=f"Q_abs_neg_{t}_{line_idx}")

            # model.addConstr(Q_abs_vars[t, line_idx] <= Q_branch_vars[t, line_idx] + M * (1 - Q_sign[t, line_idx]),
            #                 name=f"Q_abs_helper1_{t}_{line_idx}")
            # model.addConstr(Q_abs_vars[t, line_idx] <= -Q_branch_vars[t, line_idx] + M * Q_sign[t, line_idx],
            #                 name=f"Q_abs_helper2_{t}_{line_idx}")

            # Approximate apparent power using linear relaxation
            # S_branch_approx_expr[t, line_idx] = P_abs_vars[t, line_idx] + Q_abs_vars[t, line_idx]

            # Compute apparent power (S_line)
            #S_line_vars = model.addVar(lb=0, name=f"S_line_{t}_{line_idx}")
            model.addGenConstrNorm(S_branch_vars[t, line_idx], [P_branch_vars[t, line_idx], Q_branch_vars[t, line_idx]], 2, name=f"S_branch_calc_{t}_{line_idx}")

            # Define line rating based on voltage and current limits
            S_rated_line = np.sqrt(3) * line.max_i_ka * net.bus.at[from_bus, 'vn_kv']

            model.addConstr(Line_loading_vars[t, line_idx] == (S_branch_vars[t, line_idx] / S_rated_line) * 100, name=f"line_loading_{t}_{line_idx}")

            model.addConstr(
            Line_loading_vars[t, line_idx] <= 80,  # Enforce 80% limit
            name=f"line_loading_limit_{t}_{line_idx}"
            )
            

            # # Compute line loading percentage as a linear expression
            # Line_loading_expr[t, line_idx] = (S_branch_approx_vars[t, line_idx] / S_rated) * 100

            # # Enforce the 80% limit correctly
            # model.addConstr(
            #     Line_loading_expr[t, line_idx] <= 80,
            #     name=f"line_loading_limit_{t}_{line_idx}"
            # )
        # Transformer loading constraints
        for trafo in net.trafo.itertuples():
            trafo_idx = trafo.Index
            lv_bus = trafo.lv_bus
            hv_bus = trafo.hv_bus

            # Transformer losses
            # R_trafo = np.real(Z[len(net.line) + trafo_idx])
            # X_trafo = np.imag(Z[len(net.line) + trafo_idx])

            #P_loss_trafo = model.addVar(lb=0, name=f"P_loss_trafo_{t}_{trafo_idx}")
            #Q_loss_trafo = model.addVar(lb=0, name=f"Q_loss_trafo_{t}_{trafo_idx}")

            # Auxiliary variable for squared voltage
            #V_lv_squared = model.addVar(name=f"V_lv_squared_{t}_{trafo_idx}")
            #model.addConstr(V_lv_squared == V_vars[t, lv_bus] * V_vars[t, lv_bus])

            # Reformulate constraints without division
            #model.addConstr(P_loss_trafo * V_lv_squared == R_trafo * (P_accumulated_vars[t, lv_bus] ** 2 + Q_accumulated_vars[t, lv_bus] ** 2))
            #model.addConstr(Q_loss_trafo * V_lv_squared == X_trafo * (P_accumulated_vars[t, lv_bus] ** 2 + Q_accumulated_vars[t, lv_bus] ** 2))

            # Transformer HV-side power flow
            model.addConstr(P_trafo_vars[t, trafo_idx] == P_accumulated_vars[t, lv_bus])
            model.addConstr(Q_trafo_vars[t, trafo_idx] == Q_accumulated_vars[t, lv_bus])

            # Compute apparent power (S_trafo)
            S_trafo_vars = model.addVar(lb=0, name=f"S_trafo_{t}_{trafo_idx}")
            model.addGenConstrNorm(S_trafo_vars, [P_trafo_vars[t, trafo_idx], Q_trafo_vars[t, trafo_idx]], 2, name=f"S_trafo_calc_{t}_{trafo_idx}")

            # Compute transformer loading percentage
            S_rated = net.trafo.sn_mva.iloc[trafo_idx]
            model.addConstr(transformer_loading_perc_vars[t, trafo_idx] == (S_trafo_vars / S_rated) * 100, name=f"trafo_loading_{t}_{trafo_idx}")

            model.addConstr(
            transformer_loading_perc_vars[t, trafo_idx] <= 80,  # Enforce 80% limit
            name=f"trafo_loading_limit_{t}_{trafo_idx}"
        )

        # External Grid Balance
        model.addConstr(
            ext_grid_import_P_vars[t] - ext_grid_export_P_vars[t] == 
            gp.quicksum(P_trafo_vars[t, trafo_idx] for trafo_idx in range(len(net.trafo))),
            name=f"P_balance_slack_{t}"
        )

        model.addConstr(
            ext_grid_import_Q_vars[t] - ext_grid_export_Q_vars[t] ==
            gp.quicksum(Q_trafo_vars[t, trafo_idx] for trafo_idx in range(len(net.trafo))),
            name=f"Q_balance_slack_{t}"
        )

    # Objective: Minimize total cost (import, export, and curtailment costs)
    total_cost = gp.quicksum(
        electricity_price[t] * ext_grid_import_P_vars[t] +
        electricity_price[t] * ext_grid_export_P_vars[t] +
        (gp.quicksum(curtailment_cost * curtailment_vars[t][bus] for bus in pv_buses) if len(pv_buses) > 0 else 0) +
        gp.quicksum(electricity_price[t] * (flexible_load_P_vars[t][bus]) for bus in flexible_load_buses) +
        gp.quicksum(HNS_price * HNS_vars[t][bus] for bus in flexible_load_buses)
        for t in time_steps
    ) + gp.quicksum(storage_cost_levelized * ts_capacity_vars[bus] for bus in flexible_load_buses)
    model.setObjective(total_cost, GRB.MINIMIZE)

    # After adding all constraints and variables
    #model.setParam('OutputFlag', 0)
    #model.setParam('Presolve', 0)
    model.setParam('NonConvex', 2)

    model.update()

    # Optimize the model
    model.optimize()

    # Check if optimization was successful
    if model.status == gp.GRB.OPTIMAL:
        print(f"OPF Optimal Objective Value: {model.ObjVal}")
        #print("\n--- Debugging P_abs and P_branch Values ---\n")
    
        # Extract optimized values for PV generation, external grid power, loads, and theta
        for t in time_steps:
            pv_gen_results[t] = {bus: pv_gen_vars[t][bus].x for bus in pv_buses}
            ext_grid_import_P_results[t] = ext_grid_import_P_vars[t].x
            ext_grid_import_Q_results[t] = ext_grid_import_Q_vars[t].x
            ext_grid_export_P_results[t] = ext_grid_export_P_vars[t].x
            ext_grid_export_Q_results[t] = ext_grid_export_Q_vars[t].x
            V_results[t] = {bus: V_vars[t, bus].x for bus in net.bus.index}
            transformer_loading_results[t] = {
                trafo_idx: transformer_loading_perc_vars[t, trafo_idx].x for trafo_idx in net.trafo.index
            }
            
            # Extract load results as **flat dictionaries**
            flexible_load_P_results[t] = {bus: flexible_load_P_vars[t][bus].x for bus in flexible_load_buses}
            flexible_load_Q_results[t] = {bus: flexible_load_Q_vars[t][bus].x for bus in flexible_load_buses}
            non_flexible_load_P_results[t] = {bus: non_flexible_time_synchronized_loads_P[t][bus] for bus in non_flexible_load_buses}
            non_flexible_load_Q_results[t] = {bus: non_flexible_time_synchronized_loads_Q[t][bus] for bus in non_flexible_load_buses}
            HNS_results[t] = {bus: HNS_vars[t][bus].x for bus in flexible_load_buses}

            # Extract thermal storage results for each flexible load bus
            ts_in_results[t] = {bus: ts_in_vars[t][bus].x for bus in flexible_load_buses}
            ts_out_results[t] = {bus: ts_out_vars[t][bus].x for bus in flexible_load_buses}
            ts_sof_results[t] = {bus: ts_sof_vars[t][bus].x for bus in flexible_load_buses}

            ts_capacity_results['capacity'] = {bus: ts_capacity_vars[bus].x for bus in flexible_load_buses}

            line_pl_results[t] = {
                line_idx: -1 * P_branch_vars[t, line_idx].x for line_idx in net.line.index
            }
            line_ql_results[t] = {
                line_idx: -1 * Q_branch_vars[t, line_idx].x for line_idx in net.line.index
            }
            line_loading_results[t] = {
                line_idx: Line_loading_vars[t, line_idx].x for line_idx in net.line.index
            }

            # line_loading_results[t] = {
            #     line_idx: (
            #         np.sqrt(P_branch_vars[t, line_idx].x**2 + Q_branch_vars[t, line_idx].x**2) /
            #         (np.sqrt(3) * V_results[t][net.line.at[line_idx, 'from_bus']] * net.bus.at[net.line.at[line_idx, 'from_bus'], 'vn_kv'])
            #     ) * 100
            #     for line_idx in net.line.index
            # }

            line_current_results[t] = {
                line_idx: (
                    S_branch_vars[t, line_idx].x /
                    (np.sqrt(3) * V_results[t][net.line.at[line_idx, 'from_bus']] * net.bus.at[net.line.at[line_idx, 'from_bus'], 'vn_kv'])
                )
                for line_idx in net.line.index
            }

        # Return results in a structured format
        results = {
            'pv_gen': pv_gen_results,
            'flexible_load_p': flexible_load_P_results,
            'flexible_load_q': flexible_load_Q_results,
            'non_flexible_load_p': non_flexible_load_P_results,
            'non_flexible_load_q': non_flexible_load_Q_results,
            'HNS': HNS_results,
            'ext_grid_import_p': ext_grid_import_P_results,
            'ext_grid_import_q': ext_grid_import_Q_results,
            'ext_grid_export_p': ext_grid_export_P_results,
            'ext_grid_export_q': ext_grid_export_Q_results,
            'voltage': V_results, 
            'line_P': line_pl_results,
            'line_Q': line_ql_results,
            'line_current': line_current_results,
            'line_loading': line_loading_results,
            'transformer_loading': transformer_loading_results,
            'thermal_storage_capacity': ts_capacity_results,
            'thermal_storage_in': ts_in_results,
            'thermal_storage_out': ts_out_results,
            'thermal_storage_sof': ts_sof_results
        }

        # # Save the results to a file
        # if results is not None:
        #     rs.save_optim_results(results, "drcc_results.pkl")

        print(f"thermal storage capacity: {ts_capacity_results['capacity']}")
        
        return results
    
    elif model.status == gp.GRB.INFEASIBLE:
        # If the model is infeasible, write the model to an ILP file for debugging
        print("OPF Optimization failed - model is infeasible. Writing model to 'infeasible_model.ilp'")
        model.computeIIS()  # Compute IIS to identify the infeasible set
        model.write("infeasible_model.ilp")
        return None
    else:
        print(f"OPF Optimization failed with status: {model.status}")
        return None
