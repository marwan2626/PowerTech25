"""
Code for the publication Reliability-Constrained Thermal Storage Sizing in
Multi-Energy Systems under Uncertainty
Optimization by Marwan Mostafa 

Reliability File
"""

###############################################################################
## IMPORT PACKAGES & SCRIPTS ##
###############################################################################
#### PACKAGES ####
import time
import numpy as np
import pandas as pd
import pandapower as pp
from joblib import Parallel, delayed
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import ldf_lc as ldf



#### SCRIPTS ####
import parameters as par
import results as rs
import data as dt
import plot as pl

###############################################################################
## Generate Samples ##
###############################################################################
# Generate failure scenarios
def generate_failure_schedule(failure_rate, repair_time, time_steps):
    n_steps = len(time_steps)
    failure_schedule = np.ones(n_steps)  # Initialize as all operational
    
    t = 0
    while t < n_steps:
        # If the system is operational, determine the next failure
        if failure_schedule[t] == 1:
            # Time to next failure (exponentially distributed)
            time_to_failure = np.random.exponential(1 / failure_rate)
            t_fail = int(t + time_to_failure)
            
            # If failure happens within the simulation time
            if t_fail < n_steps:
                # Duration of failure (normally distributed with cutoffs)
                min_repair_time = 2  # Minimum repair time in hours
                max_repair_time = 3 * repair_time  # Maximum repair time
                repair_duration = np.clip(
                    np.random.normal(repair_time), min_repair_time, max_repair_time
                )
                t_repair = int(t_fail + repair_duration)
                
                # Set failure period in the schedule
                failure_schedule[t_fail:min(t_repair, n_steps)] = 0
                
                # Move time pointer to the end of the repair period
                t = t_repair
            else:
                break  # No more failures within the time horizon
        else:
            t += 1  # If already in failure, move to the next step

    return failure_schedule

def generate_all_failure_schedules(failure_rate_trafo, repair_time_trafo, 
                                   failure_rate_hp, repair_time_hp,
                                   failure_rate_ts, repair_time_ts, 
                                   time_steps, n_scenarios):
 
    trafo_failures = [
        generate_failure_schedule(failure_rate_trafo, repair_time_trafo, time_steps)
        for _ in range(n_scenarios)
    ]
    hp_failures = [
        generate_failure_schedule(failure_rate_hp, repair_time_hp, time_steps)
        for _ in range(n_scenarios)
    ]
    ts_failures = [
        generate_failure_schedule(failure_rate_ts, repair_time_ts, time_steps)
        for _ in range(n_scenarios)
    ]
    return trafo_failures, hp_failures, ts_failures

###############################################################################
## DEBUG ##
###############################################################################
def generate_deterministic_failure_schedule(failure_timestep, repair_time, time_steps):
    n_steps = len(time_steps)
    failure_schedule = np.ones(n_steps)  # Initialize as all operational
    
    # Calculate the end of the repair period
    t_repair = int(failure_timestep + repair_time)
    
    # Set failure period in the schedule
    if failure_timestep < n_steps:
        failure_schedule[failure_timestep:min(t_repair, n_steps)] = 0  # Mark failure period
    
    return failure_schedule

def generate_all_deterministic_failure_schedules(failure_timestep_trafo, repair_time_trafo, 
                                                 failure_timestep_hp, repair_time_hp, 
                                                 failure_timestep_ts, repair_time_ts,
                                                 time_steps, n_scenarios):
    trafo_failures = [
        generate_deterministic_failure_schedule(failure_timestep_trafo, repair_time_trafo, time_steps)
        for _ in range(n_scenarios)
    ]
    hp_failures = [
        generate_deterministic_failure_schedule(failure_timestep_hp, repair_time_hp, time_steps)
        for _ in range(n_scenarios)
    ]
    ts_failures = [
        generate_deterministic_failure_schedule(failure_timestep_ts, repair_time_ts, time_steps)
        for _ in range(n_scenarios)
    ]
    return trafo_failures, hp_failures, ts_failures


###############################################################################
## OPF ##
###############################################################################

### solve the OPF problem ###
def solve_opf_with_failures(net, time_steps, electricity_price, T_amb,  TRAFO_FAIL, HP_FAIL, TS_FAIL):

    pd.set_option('display.precision', 10)
    model = gp.Model("opf_with_ldf_lc")

    # Define the costs
    curtailment_cost = electricity_price

    HNS_price = par.HNS_cost

    ### Define the variables ###
    epsilon = 100e-9  # Small positive value to ensure some external grid usage


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
    total_heat_demand_results = {}


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
        lb=0.00001,
        name=f'ts_capacity'
    )

    heat_demand_scaling = 1 / par.tsnet_eff
    COP = {}  # Coefficient of Performance (COP) for heat pumps
    for t in time_steps:
        COP[t] = par.eta_c0 * (T_amb[t] + par.DeltaT)/(par.T_S - T_amb[t] + 2*par.DeltaT)
    #print(f"COP: {COP}")

    # Add variables for each time step
    for t in time_steps:
        if not net.controller.empty:
            for _, controller in net.controller.iterrows():
                controller.object.time_step(net, time=t)


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

        model.addConstr(ext_grid_import_P_vars[t] + ext_grid_export_P_vars[t] >= epsilon, name=f'nonzero_ext_grid_P_usage_{t}')
        model.addConstr(ext_grid_import_Q_vars[t] + ext_grid_export_Q_vars[t] >= epsilon, name=f'nonzero_ext_grid_Q_usage_{t}')

        # Define flexible load variables with global peak limit (par.hp_max_power)
        flexible_load_P_vars[t] = model.addVars(
            flexible_load_buses,
            lb=0,
            ub=par.hp_max_power*(1-HP_FAIL[t])*(1-TRAFO_FAIL[t]),
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
    #S_trafo_vars = model.addVars(time_steps, net.line.index, lb=0, name="S_trafo")  

    #Transformer loading percentage
    transformer_loading_perc_vars = model.addVars(time_steps, net.trafo.index, lb=0, name="Trafo_loading_percent")
    #Line_loading_vars = model.addVars(time_steps, net.line.index, name="Line_loading")
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

        total_heat_supply = gp.quicksum(flexible_load_P_vars[t][bus]  for bus in flexible_load_buses) + \
                    gp.quicksum((ts_out_vars[t][bus] - ts_in_vars[t][bus]) / COP[t] for bus in flexible_load_buses)

        total_heat_demand = gp.quicksum(flexible_time_synchronized_loads_P[t][bus] * heat_demand_scaling for bus in flexible_load_buses) - \
                            gp.quicksum(HNS_vars[t][bus] for bus in flexible_load_buses)
        
        model.addConstr(
            total_heat_supply == total_heat_demand,
            name=f"global_heat_balance_{t}"
        )

        for i, bus in enumerate(net.bus.index):
            if bus in net.load.bus.values:
                if bus in flexible_load_buses:
                    # Define the heat demand for the bus, representing the electrical equivalent
                    # heat_demand = flexible_time_synchronized_loads_P[t][bus]  # or load profile representing heat demand
                    # #eta_pl_vars[t][bus] = 1 - (par.psi*(1-(flexible_load_P_vars[t][bus]/par.hp_max_power))) 
                    # model.addGenConstrPWL(
                    #     flexible_load_P_vars[t][bus],
                    #     eta_pl_vars[t][bus],
                    #     xpts=[0, par.hp_max_power],
                    #     ypts=[1 - par.psi, 1],  # Because eta = 1 - ψ*(1 - load/p_max)
                    #     name=f"eta_pl_def_{t}_{bus}"
                    # )

                    # # 1. Heat Demand Coverage: flexible_load_vars and/or thermal storage must meet the demand (all in electrical equivalent)
                    # model.addConstr(
                    #     flexible_load_P_vars[t][bus] + ((ts_out_vars[t][bus] - ts_in_vars[t][bus]) / COP[t]) == (heat_demand * heat_demand_scaling) - HNS_vars[t][bus],
                    #     name=f'heat_demand_coverage_{t}_{bus}'
                    # )

                    # # 3. Storage Charging: use excess power for storage charging if available
                    # model.addConstr(
                    #     ts_in_vars[t][bus] <= flexible_load_P_vars[t][bus] * COP[t] * eta_pl_vars[t][bus],
                    #     name=f'storage_charging_{t}_{bus}'
                    # )

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

                tight_v_min = 0.95 + par.DRCC_FLG * k_epsilon * results_variance.loc[t, ("V_variance", bus)]
                tight_v_max = 1.05 - par.DRCC_FLG * k_epsilon * results_variance.loc[t, ("V_variance", bus)]

                model.addConstr(V_vars[t, bus] >= tight_v_min, name=f"voltage_min_drcc_{t}_{bus}")
                model.addConstr(V_vars[t, bus] <= tight_v_max, name=f"voltage_max_drcc_{t}_{bus}")
        

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

            S_rated_line = np.sqrt(3) * line.max_i_ka * net.bus.at[from_bus, 'vn_kv']
            tight_line_limit = (par.DRCC_FLG * k_epsilon * results_variance.loc[t, ("line_pl_mw", line_idx)])/par.hp_pf
            S_branch_limit = 0.8 * S_rated_line - tight_line_limit

            #model.addGenConstrNorm(S_branch_vars[t, line_idx], [P_branch_vars[t, line_idx], Q_branch_vars[t, line_idx]], 2, name=f"S_branch_calc_{t}_{line_idx}")
            model.addQConstr(
                P_branch_vars[t, line_idx]*P_branch_vars[t, line_idx] +
                Q_branch_vars[t, line_idx]*Q_branch_vars[t, line_idx]
                <= S_branch_limit**2,
                name=f"S_branch_limit_{t}_{line_idx}"
            )

            # Define line rating based on voltage and current limits

            #model.addConstr(S_branch_vars[t, line_idx] <= (0.8*S_rated_line)-tight_line_limit, name=f"S_branch_limit_{t}_{line_idx}")

            #model.addConstr(Line_loading_vars[t, line_idx] == (S_branch_vars[t, line_idx] / S_rated_line) * 100, name=f"line_loading_{t}_{line_idx}")


        # Transformer loading constraints
        for trafo in net.trafo.itertuples():
            trafo_idx = trafo.Index
            lv_bus = trafo.lv_bus
            hv_bus = trafo.hv_bus

            # Transformer HV-side power flow
            model.addConstr(P_trafo_vars[t, trafo_idx] == P_accumulated_vars[t, lv_bus])
            model.addConstr(Q_trafo_vars[t, trafo_idx] == Q_accumulated_vars[t, lv_bus])

            # Compute transformer loading percentage
            S_rated = net.trafo.sn_mva.iloc[trafo_idx]
        
            tight_trafo_limit = par.DRCC_FLG * k_epsilon * results_variance.loc[t, ("S_trafo", trafo_idx)]
            S_limit = 0.8*S_rated - tight_trafo_limit

            model.addQConstr(
                P_trafo_vars[t, trafo_idx]*P_trafo_vars[t, trafo_idx] +
                Q_trafo_vars[t, trafo_idx]*Q_trafo_vars[t, trafo_idx]
                <= S_limit**2,
                name=f"S_trafo_limit_{t}_{trafo_idx}"
            )


            # model.addConstr(S_trafo_vars <= (0.8*S_rated)-tight_trafo_limit, name=f"S_trafo_limit_{t}_{trafo_idx}")

            #model.addConstr(transformer_loading_perc_vars[t, trafo_idx] == (S_trafo_vars / S_rated) * 100, name=f"trafo_loading_{t}_{trafo_idx}")


        #     model.addConstr(
        #     transformer_loading_perc_vars[t, trafo_idx] <= tight_trafo_limit,  # Enforce 80% limit
        #     name=f"trafo_loading_limit_{t}_{trafo_idx}"
        # )

        # External Grid Balance
        model.addConstr(
            ext_grid_export_P_vars[t] - ext_grid_import_P_vars[t] == 
            gp.quicksum(P_trafo_vars[t, trafo_idx] for trafo_idx in range(len(net.trafo))),
            name=f"P_balance_slack_{t}"
        )

        model.addConstr(
            ext_grid_export_Q_vars[t] - ext_grid_import_Q_vars[t] ==
            gp.quicksum(Q_trafo_vars[t, trafo_idx] for trafo_idx in range(len(net.trafo))),
            name=f"Q_balance_slack_{t}"
        )

    # Objective: Minimize total cost (import, export, and curtailment costs)
    total_cost = gp.quicksum(
        electricity_price[t] * ext_grid_import_P_vars[t] +
        electricity_price[t] * ext_grid_export_P_vars[t] +
        (gp.quicksum(electricity_price[t] * curtailment_vars[t][bus] for bus in pv_buses) if len(pv_buses) > 0 else 0) +
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

            transformer_loading_results[t] = {
                trafo_idx: (
                    np.sqrt(
                        P_trafo_vars[t, trafo_idx].x ** 2 + Q_trafo_vars[t, trafo_idx].x ** 2
                    ) / net.trafo.at[trafo_idx, 'sn_mva']
                ) * 100
                for trafo_idx in net.trafo.index
            }

            line_pl_results[t] = {
                line_idx: -1 * P_branch_vars[t, line_idx].x for line_idx in net.line.index
            }
            line_ql_results[t] = {
                line_idx: -1 * Q_branch_vars[t, line_idx].x for line_idx in net.line.index
            }

            line_loading_results[t] = {
                line_idx: (
                    (
                    np.sqrt(P_branch_vars[t, line_idx].x ** 2 + Q_branch_vars[t, line_idx].x ** 2) /
                    (np.sqrt(3) * V_results[t][net.line.at[line_idx, 'from_bus']] * net.bus.at[net.line.at[line_idx, 'from_bus'], 'vn_kv'])
                    ) / net.line.at[line_idx, 'max_i_ka']
                ) * 100
                for line_idx in net.line.index
            }

            line_current_results[t] = {
                line_idx: (
                    np.sqrt(P_branch_vars[t, line_idx].x ** 2 + Q_branch_vars[t, line_idx].x ** 2) /
                    (np.sqrt(3) * V_results[t][net.line.at[line_idx, 'from_bus']] * net.bus.at[net.line.at[line_idx, 'from_bus'], 'vn_kv'])
                )
                for line_idx in net.line.index
            }

            total_heat_demand_results[t] = {}
            for bus in flexible_load_buses:
                total_heat_demand_results[t][bus] = (
                    flexible_time_synchronized_loads_P[t][bus] * heat_demand_scaling
                    - HNS_vars[t][bus].x
                )

            

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
            'thermal_storage_sof': ts_sof_results,
            'total_heat_demand': total_heat_demand_results
        }

        # # Save the results to a file
        # if results is not None:
        #     rs.save_optim_results(results, "drcc_results.pkl")
        
        # Save the results to a file
        if results is not None:
            filename = f"drcc_results_drcc_{par.DRCC_FLG}_{par.epsilon}.pkl"
            rs.save_optim_results(results, filename)

        print(f"thermal storage capacity: {ts_capacity_results['capacity']}")
        
        return results
        
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




###############################################################################
## Reliability ANALYSIS ##
###############################################################################

def run_single_scenario(
    scenario,
    net,
    time_steps,
    const_load_heatpump,
    const_load_household,
    T_amb,
    Bbus,
    TS_capacity,
    trafo_failure,
    hp_failure,
    ts_failure
):
    # Convert failure schedules to flags
    TRAFO_FAIL = {t: int(trafo_failure[i] == 0) for i, t in enumerate(time_steps)}
    HP_FAIL = {t: int(hp_failure[i] == 0) for i, t in enumerate(time_steps)}
    TS_FAIL = {t: int(ts_failure[i] == 0) for i, t in enumerate(time_steps)}

    # Check if there is no failure at all in the scenario
    if all(TRAFO_FAIL[t] == 0 for t in time_steps) and all(HP_FAIL[t] == 0 for t in time_steps):
        # If no failure, skip OPF and use the original results directly
        HNS_total = 0
        final_storage = 0.5

    else:
        # If there are failures, run OPF with failures
        results_df = solve_opf_with_failures(net, time_steps, const_load_heatpump, const_load_household, T_amb, Bbus, TS_capacity, TRAFO_FAIL, HP_FAIL, TS_FAIL)
    
        #HNS_total = sum(sum(results_df["load"][t]["HNS"].values()) for t in time_steps) + np.abs(sum(results_df["thermal_storage"]["ts_sof"][time_steps[-1]].values()) -0.5)*par.TS_capacity
        HNS_total = sum(sum(results_df["load"][t]["HNS"].values()) for t in time_steps) 
        final_storage = results_df["thermal_storage"]["ts_sof"][time_steps[-1]]

    return {
        "scenario": scenario,
        "HNS": HNS_total,
        "final_storage": final_storage
    }

###############################################################################

def reliability_analysis(
    net, time_steps, const_load_heatpump, const_load_household, 
    T_amb, Bbus, n_jobs
):
    TS_capacity = par.TS_capacity
    N_scenarios = par.N_scenarios

    # Generate all failure schedules
    trafo_failures, hp_failures, ts_failures = generate_all_failure_schedules(par.failure_rate_trafo, par.repair_time_trafo, 
                                   par.failure_rate_hp, par.repair_time_hp,
                                   par.failure_rate_ts, par.repair_time_ts, 
                                   time_steps, N_scenarios)

    print("generated failure schedules")

    # Use joblib for parallel processing
    scenarios = range(N_scenarios)
    results_rel = Parallel(n_jobs=n_jobs)(
        delayed(run_single_scenario)(
            scenario,
            net.deepcopy(),
            time_steps,
            const_load_heatpump,
            const_load_household,
            T_amb,
            Bbus,
            TS_capacity,
            trafo_failures[scenario],
            hp_failures[scenario],
            ts_failures[scenario]
        )
        for scenario in tqdm(scenarios, desc="Processing scenarios")
    )

    # Aggregate results
    total_HNS = sum(res["HNS"] for res in results_rel)
    EHNS = total_HNS / N_scenarios

    print(f"Reliability study for TS_capacity = {TS_capacity} MWh")
    print(f"  Total HNS = {total_HNS:.8f} MWh")
    print(f"  EHNS = {EHNS:.8f} MWh")

    # Save results
    rs.save_optim_results(results_rel, "results_rel.pkl")

    return results_rel

