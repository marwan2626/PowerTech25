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
    n_steps = len(time_steps)  # Total number of timesteps
    failure_schedule = np.ones(n_steps)  # Initialize as all operational
    t = 0
    while t < n_steps:
        time_to_failure = max(0, np.random.exponential(1 / failure_rate))  # Avoid negative times
        repair_duration = max(0, np.random.normal(repair_time))  # Avoid negative times
        t_fail = int(t + time_to_failure)
        t_repair = int(t_fail + repair_duration)
        if t_fail < n_steps:
            failure_schedule[t_fail:min(t_repair, n_steps)] = 0  # Mark failure period
        t = t_repair
    return failure_schedule

def generate_all_failure_schedules(failure_rate_trafo, repair_time_trafo, 
                                   failure_rate_hp, repair_time_hp, 
                                   time_steps, n_scenarios):
 
    trafo_failures = [
        generate_failure_schedule(failure_rate_trafo, repair_time_trafo, time_steps)
        for _ in range(n_scenarios)
    ]
    hp_failures = [
        generate_failure_schedule(failure_rate_hp, repair_time_hp, time_steps)
        for _ in range(n_scenarios)
    ]
    return trafo_failures, hp_failures

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

def generate_all_deterministic_failure_schedules(failure_timestep, repair_time_trafo, 
                                                 repair_time_hp, time_steps, n_scenarios):
    trafo_failures = [
        generate_deterministic_failure_schedule(failure_timestep, repair_time_trafo, time_steps)
        for _ in range(n_scenarios)
    ]
    hp_failures = [
        generate_deterministic_failure_schedule(failure_timestep, repair_time_hp, time_steps)
        for _ in range(n_scenarios)
    ]
    return trafo_failures, hp_failures

###############################################################################
## OPF ##
###############################################################################

### solve the OPF problem ###
def solve_opf_with_failures(net, time_steps, const_load_heatpump, const_load_household, heatpump_scaling_factors_df, T_amb, Bbus, TS_capacity, TRAFO_FAIL, HP_FAIL, initial_storage_state):

    model = gp.Model("opf_with_dc_load_flow")

    # Define the costs
    curtailment_cost = par.curtailment_cost  

    filepath = "electricityprice1h.csv"
    electricity_price = dt.get_electricity_price(filepath)['price']
    #print(f"Electricity Price: {electricity_price}")

    HNS_price = par.HNS_cost


    ### Define the variables ###
    epsilon = 100e-9  # Small positive value to ensure some external grid usage

    # Extract transformer capacity in MW (assuming sn_mva is in MVA)
    transformer_capacity_mw = net.trafo['sn_mva'].values[0]

    # Initialize decision variables
    pv_gen_vars = {}  # Store PV generation decision variables
    ext_grid_import_vars = {}  # Store external grid import power decision variables
    ext_grid_export_vars = {}  # Store external grid export power decision variables
    theta_vars = {}  # Store voltage angle decision variables (radians)
    curtailment_vars = {} # Store decision variables for curtailment
    flexible_load_vars = {}  # New flexible load variables
    transformer_loading_vars = {}  # Store transformer loading percentage decision variables 
    transformer_loading_perc_vars = {}  # Store transformer loading percentage decision variables  
    eta_pl_vars = {}  # Store partload electrical efficiency of heat pumps
    HNS_vars = {}  # Store HNS variables

    # Add thermal storage variables
    ts_in_vars = {t: {} for t in time_steps}
    ts_out_vars = {t: {} for t in time_steps}
    ts_sof_vars = {t: {} for t in time_steps}
    energy_stored_vars = {t: {} for t in time_steps}    

    slack_bus_index = net.ext_grid.bus.iloc[0]

    # Pre-process Bbus: Remove the row and column corresponding to the slack bus
    Bbus_reduced = np.delete(Bbus, slack_bus_index, axis=0)
    Bbus_reduced = np.delete(Bbus_reduced, slack_bus_index, axis=1)

    # Dictionaries to store results
    pv_gen_results = {}
    load_results = {}
    ext_grid_import_results = {}
    ext_grid_export_results = {}
    theta_results = {}
    line_results = {}
    transformer_loading_results = {}
    ts_capacity_results = {}
    thermal_storage_results = {
        'ts_in': {t: {} for t in time_steps},
        'ts_out': {t: {} for t in time_steps},
        'ts_sof': {t: {} for t in time_steps}
    }

    # Temporary dictionary to store updated load values per time step
    flexible_time_synchronized_loads = {t: {} for t in time_steps}
    non_flexible_time_synchronized_loads = {t: {} for t in time_steps}

    # Identify buses with flexible loads
    flexible_load_buses = list(set(net.load[net.load['controllable'] == True].bus.values))
    non_flexible_load_buses = list(set(net.load[net.load['controllable'] == False].bus.values))
    #print(f"Flexible load buses: {flexible_load_buses}")

    heat_demand_scaling = 1 / par.tsnet_eff
    COP = {}  # Coefficient of Performance (COP) for heat pumps
    for t in time_steps:
        COP[t] = par.eta_c0 * (T_amb[t] + par.DeltaT)/(par.T_S - T_amb[t] + 2*par.DeltaT)
    #print(f"COP: {COP}")

    # Add variables for each time step
    for t in time_steps:
        # Update const_pv and const_load for this time step
        const_load_heatpump.time_step(net, time=t)
        const_load_household.time_step(net, time=t)


        # Initialize dictionaries for time-synchronized loads
        flexible_time_synchronized_loads[t] = {}
        non_flexible_time_synchronized_loads[t] = {}

        # Iterate over all loads
        for load in net.load.itertuples():
            bus = load.bus
            if load.controllable:
                # Flexible load
                flexible_time_synchronized_loads[t][bus] = (
                    flexible_time_synchronized_loads[t].get(bus, 0.0) + load.p_mw
                )
            else:
                # Non-flexible load
                non_flexible_time_synchronized_loads[t][bus] = (
                    non_flexible_time_synchronized_loads[t].get(bus, 0.0) + load.p_mw
                )

        # Ensure all buses have an entry, even if no loads are connected
        for bus in net.bus.index:
            if bus not in flexible_time_synchronized_loads[t]:
                flexible_time_synchronized_loads[t][bus] = 0.0
            if bus not in non_flexible_time_synchronized_loads[t]:
                non_flexible_time_synchronized_loads[t][bus] = 0.0

        # Extract the bus indices where PV generators are connected (from net.sgen.bus)
        pv_buses = net.sgen.bus.values

        # Create PV generation variables for this time step
        if len(pv_buses) > 0:
            pv_gen_vars[t] = model.addVars(pv_buses, lb=0, ub=net.sgen.p_mw.values, name=f'pv_gen_{t}')
            curtailment_vars[t] = model.addVars(pv_buses, lb=0, ub=net.sgen.p_mw.values, name=f'curtailment_{t}')
            for bus in pv_buses:
                # Find the index in sgen corresponding to this bus
                sgen_index = np.where(net.sgen.bus.values == bus)[0][0]
                model.addConstr(curtailment_vars[t][bus] == net.sgen.p_mw.values[sgen_index] - pv_gen_vars[t][bus], 
                                name=f'curtailment_constraint_{t}_{bus}')
            
        # External grid power variables for import and export at the slack bus (bus 0)
        ext_grid_import_vars[t] = model.addVar(lb=0, name=f'ext_grid_import_{t}')  # Import is non-negative
        ext_grid_export_vars[t] = model.addVar(lb=0, name=f'ext_grid_export_{t}')  # Export is non-negative
        model.addConstr(ext_grid_import_vars[t] + ext_grid_export_vars[t] >= epsilon, name=f'nonzero_ext_grid_usage_{t}')

        # Limit grid import and export based on transformer capacity
        model.addConstr(
            ext_grid_import_vars[t] <= transformer_capacity_mw, 
            name=f'limit_import_{t}'
        )
        model.addConstr(
            ext_grid_export_vars[t] <= transformer_capacity_mw, 
            name=f'limit_export_{t}'
        )
        # Voltage angle variables for all buses
        theta_vars[t] = model.addVars(net.bus.index, lb=-GRB.INFINITY, name=f'theta_{t}')

        # Fix the slack bus angle to 0 radians
        model.addConstr(theta_vars[t][slack_bus_index] == 0, name=f'slack_theta_{t}')

        # Define flexible load variables with global peak limit (par.hp_max_power)
        flexible_load_vars[t] = model.addVars(
            flexible_load_buses,
            lb=0,
            ub=par.hp_max_power,
            name=f'flexible_load_{t}'
        )

                # Define flexible load variables with global peak limit (par.hp_max_power)
        HNS_vars[t] = model.addVars(
            flexible_load_buses,
            lb=0,
            ub=par.hp_max_power,
            name=f'flexible_load_{t}'
        )

        eta_pl_vars[t] = model.addVars(
            flexible_load_buses,
            lb=0,
            name=f'partload_effeciency_{t}'
        )
                            
    # Create a dictionary mapping bus indices to scaling factors using heatpump_scaling_factors_df
    heatpump_scaling_factors_dict = {
        bus: heatpump_scaling_factors_df.loc[heatpump_scaling_factors_df['bus'] == bus, 'p_mw'].values[0]
        for bus in flexible_load_buses
    }
       
    # Update SOF constraints for each flexible load bus
    for t_idx, t in enumerate(time_steps):
        for bus in flexible_load_buses:
            ts_in_vars[t][bus] = model.addVar(lb=0, ub=par.ts_in_max, name=f'ts_in_{t}_{bus}')
            ts_out_vars[t][bus] = model.addVar(lb=0, ub=par.ts_out_max, name=f'ts_out_{t}_{bus}')
            ts_sof_vars[t][bus] = model.addVar(lb=0, ub=1.0, name=f'ts_sof_{t}_{bus}')  # SOF as percentage (0 to 1)
            energy_stored_vars[t][bus] = model.addVar(lb=0, name=f'energy_stored_{t}_{bus}')

    # Add power balance and load flow constraints for each time step
    for t in time_steps:
        # Power injection vector P
        P_injected = {bus: gp.LinExpr() for bus in net.bus.index}

        for i, bus in enumerate(net.bus.index):
            if bus in net.load.bus.values:
                if bus in flexible_load_buses:
                    # Define the heat demand for the bus, representing the electrical equivalent
                    heat_demand = flexible_time_synchronized_loads[t][bus]  # or load profile representing heat demand
                    eta_pl_vars[t][bus] = 1 - (par.psi*(1-(flexible_load_vars[t][bus]/par.hp_max_power))) 

                    # 1. Heat Demand Coverage: flexible_load_vars and/or thermal storage must meet the demand (all in electrical equivalent)
                    model.addConstr(
                        flexible_load_vars[t][bus]*(1 - TRAFO_FAIL[t])*(1 - HP_FAIL[t]) + ((ts_out_vars[t][bus] - ts_in_vars[t][bus]) / COP[t]) == (heat_demand * heat_demand_scaling) - HNS_vars[t][bus],
                        name=f'heat_demand_coverage_{t}_{bus}'
                    )

                    # 3. Storage Charging: use excess power for storage charging if available
                    model.addConstr(
                        ts_in_vars[t][bus] <= flexible_load_vars[t][bus] * COP[t] * eta_pl_vars[t][bus]*(1 - TRAFO_FAIL[t])*(1 - HP_FAIL[t]),
                        name=f'storage_charging_{t}_{bus}'
                    )

                    model.addConstr(
                        ts_out_vars[t][bus] <= energy_stored_vars[t][bus],
                        name=f'storage_output_limit_{t}_{bus}'
                    )

                    # Update the state of fill (SOF) of the storage
                    # Initial SOF Constraint for the first timestep
                    if t_idx == 0:
                        model.addConstr(
                            energy_stored_vars[t][bus] == initial_storage_state[bus],
                            name=f'initial_energy_{t}_{bus}'
                        )
                    else:
                        prev_t = time_steps[t_idx - 1]
                        model.addConstr(
                            energy_stored_vars[t][bus] ==
                            energy_stored_vars[prev_t][bus] +
                            (par.ts_eff * ts_in_vars[t][bus]) -
                            (ts_out_vars[t][bus] / par.ts_eff) -
                            (par.ts_alpha * energy_stored_vars[prev_t][bus] * (par.T_S-T_amb[t])),
                            name=f'storage_energy_update_{t}_{bus}'
                        )

                    model.addConstr(
                        ts_sof_vars[t][bus] * TS_capacity == energy_stored_vars[t][bus],
                        name=f'sof_definition_{t}_{bus}'
                    )
                    # Use the flexible load variable for controllable loads
                    P_injected[bus] -= flexible_load_vars[t][bus]

                if bus in non_flexible_load_buses:
                    # For non-flexible loads, use the time-synchronized load
                    P_injected[bus] -= non_flexible_time_synchronized_loads[t][bus]

            if len(pv_buses) > 0 and bus in pv_buses:
                if bus in pv_buses:
                    # Only add PV generation if the bus has PV (i.e., in net.sgen.bus)
                    P_injected[bus] += pv_gen_vars[t][bus]

            if bus == slack_bus_index:
                # Add the import minus export for the external grid power at the slack bus
                P_injected[bus] += ext_grid_import_vars[t] - ext_grid_export_vars[t]

        model.update()

        #for bus in net.bus.index:
            #print(f"Time step {t}, Bus {bus}: Power injected (MW) = {P_injected[bus]}")

        # Convert P_injected to per unit
        P_pu = {bus: P_injected[bus] / net.sn_mva for bus in net.bus.index}

        # Reduce the power vector by removing the slack bus entry
        P_pu_reduced = [P_pu[bus] for bus in net.bus.index if bus != slack_bus_index]

        # Power flow constraint: P_pu_reduced = Bbus_reduced * theta_reduced
        theta_reduced_vars = [theta_vars[t][i] for i in net.bus.index if i != slack_bus_index]

        # Apply power balance constraints for each non-slack bus
        for i in range(len(Bbus_reduced)):
            power_balance_expr = gp.LinExpr()
            for j in range(len(Bbus_reduced)):
                power_balance_expr += Bbus_reduced[i, j] * theta_reduced_vars[j]

            model.addConstr(P_pu_reduced[i] == power_balance_expr, name=f'power_flow_{t}_{i}')


    # Enforce final state of fill to match initial state (0.5) for all flexible load buses
    # for bus in flexible_load_buses:
    #     model.addConstr(ts_sof_vars[time_steps[-1]][bus] == 0.5, name=f'final_sof_balance_{bus}')

    # Line power flow and loading constraints (with the corrected expression)
    for t in time_steps:
        line_results[t] = {
            "line_pl_mw": {},
            "line_loading_percent": {},
            "line_current_mag": {}
        }

        #
        for line in net.line.itertuples():
            from_bus = line.from_bus
            to_bus = line.to_bus
            base_voltage = net.bus.at[from_bus, 'vn_kv'] * 1e3  # Convert kV to V
            x_pu = line.x_ohm_per_km * line.length_km / ((base_voltage ** 2) / net.sn_mva)

            # Power flow on this line: (theta_from - theta_to) / X
            power_flow_expr = (theta_vars[t][from_bus] - theta_vars[t][to_bus]) / x_pu
            power_flow_mw = power_flow_expr * net.sn_mva / 1e6  # Convert to MW

            sqrt3 = np.sqrt(3)
            current_mag_ka = power_flow_mw / (sqrt3 * (base_voltage / 1e3))

            # Create an auxiliary variable for the absolute value of the current magnitude
            abs_current_mag_ka = model.addVar(lb=0, name=f'abs_current_mag_ka_{line.Index}_{t}')
            model.addConstr(abs_current_mag_ka >= current_mag_ka, name=f'abs_current_mag_ka_pos_{line.Index}_{t}')
            model.addConstr(abs_current_mag_ka >= -current_mag_ka, name=f'abs_current_mag_ka_neg_{line.Index}_{t}')

            abs_power_flow_mw = model.addVar(lb=0, name=f'abs_power_flow_mw_{line.Index}_{t}')
            model.addConstr(abs_power_flow_mw >= power_flow_mw, name=f'abs_power_flow_mw_pos_{line.Index}_{t}')
            model.addConstr(abs_power_flow_mw >= -power_flow_mw, name=f'abs_power_flow_mw_neg_{line.Index}_{t}')

            # Now, calculate the line loading percentage using the auxiliary variable
            if hasattr(line, 'max_i_ka'):
                line_loading_percent = 100 * (abs_current_mag_ka / line.max_i_ka)
                # model.addConstr(abs_current_mag_ka <= (line.max_i_ka), 
                #             name=f'abs_current_mag_constraint_{t}_{line.Index}')

                model.addConstr(abs_power_flow_mw <= (line.max_i_ka * (sqrt3 * (base_voltage / 1e3))), 
                            name=f'abs_power_flow_constraint_{t}_{line.Index}')

            # Store results for each line in the time step
            line_results[t]["line_pl_mw"][line.Index] = power_flow_mw
            line_results[t]["line_loading_percent"][line.Index] = line_loading_percent            
            line_results[t]["line_current_mag"][line.Index] = current_mag_ka

        # Transformer loading constraints
        for trafo in net.trafo.itertuples():
            x_pu = (trafo.vk_percent / 100) / trafo.sn_mva
            power_flow_pu = (theta_vars[t][trafo.hv_bus] - theta_vars[t][trafo.lv_bus]) / x_pu
            power_flow_mw = power_flow_pu * net.sn_mva
            model.addConstr(ext_grid_import_vars[t] - ext_grid_export_vars[t] == power_flow_mw, name=f'power_balance_slack_{t}')

        transformer_loading_vars[t] = model.addVar(lb=0, ub=(par.max_trafo_loading*transformer_capacity_mw), name=f'transformer_loading_{t}')
        transformer_loading_perc_vars[t] = model.addVar(lb=0, name=f'transformer_loading_percent_{t}')
        model.addConstr(
            transformer_loading_vars[t] == (ext_grid_import_vars[t] + ext_grid_export_vars[t]),
            name=f'transformer_loading_{t}'
        )
        model.addConstr(
            transformer_loading_perc_vars[t] == (transformer_loading_vars[t] / transformer_capacity_mw) * 100,
            name=f'transformer_loading_percent_constr_{t}'
        )

    # Objective: Minimize total cost (import, export, and curtailment costs)
    total_cost = gp.quicksum(
        electricity_price[t] * ext_grid_import_vars[t] +
        electricity_price[t] * ext_grid_export_vars[t] +
        (gp.quicksum(curtailment_cost * curtailment_vars[t][bus] for bus in pv_buses) if len(pv_buses) > 0 else 0) +
        gp.quicksum(electricity_price[t] * (flexible_load_vars[t][bus]) for bus in flexible_load_buses) +
        gp.quicksum(HNS_price * HNS_vars[t][bus] for bus in flexible_load_buses)
        for t in time_steps
    )
    model.setObjective(total_cost, GRB.MINIMIZE)

    # After adding all constraints and variables
    model.setParam('OutputFlag', 0)
    model.setParam('Presolve', 0)
    model.setParam('NonConvex', 2)

    model.update()

    # Optimize the model
    model.optimize()

    # Check if optimization was successful
    if model.status == gp.GRB.OPTIMAL:
        #print(f"OPF Optimal Objective Value: {model.ObjVal}")
        # Extract optimized values for PV generation, external grid power, loads, and theta
        for t in time_steps:
            pv_gen_results[t] = {bus: pv_gen_vars[t][bus].x for bus in pv_buses}
            ext_grid_import_results[t] = ext_grid_import_vars[t].x
            ext_grid_export_results[t] = ext_grid_export_vars[t].x
            theta_results[t] = {bus: theta_vars[t][bus].x for bus in net.bus.index}
            transformer_loading_results[t] = transformer_loading_perc_vars[t].x
            
            # Separate flexible and non-flexible load results
            load_results[t] = {
                'flexible_loads': {
                    bus: flexible_load_vars[t][bus].x if bus in flexible_load_buses else 0.0
                    for bus in flexible_load_buses
                },
                'non_flexible_loads': {
                    bus: non_flexible_time_synchronized_loads[t][bus]
                    for bus in non_flexible_load_buses
                },
                'HNS': {
                    bus: HNS_vars[t][bus].x if bus in flexible_load_buses else 0.0
                    for bus in flexible_load_buses
                },
            }

            # Extract thermal storage results for each flexible load bus
            thermal_storage_results['ts_in'][t] = {bus: ts_in_vars[t][bus].x for bus in flexible_load_buses}
            thermal_storage_results['ts_out'][t] = {bus: ts_out_vars[t][bus].x for bus in flexible_load_buses}
            thermal_storage_results['ts_sof'][t] = {bus: ts_sof_vars[t][bus].x for bus in flexible_load_buses}

            ts_capacity_results['capacity'] = {bus: TS_capacity for bus in flexible_load_buses}

            # Extract numerical values for line results
            for line in net.line.itertuples():
                line_results[t]["line_pl_mw"][line.Index] = line_results[t]["line_pl_mw"][line.Index].getValue()
                line_results[t]["line_loading_percent"][line.Index] = line_results[t]["line_loading_percent"][line.Index].getValue()
                line_results[t]["line_current_mag"][line.Index] = line_results[t]["line_current_mag"][line.Index].getValue()

        # Return results in a structured format
        results = {
            'pv_gen': pv_gen_results,
            'load': load_results,
            'ext_grid_import': ext_grid_import_results,
            'ext_grid_export': ext_grid_export_results,
            'theta': theta_results,  # Add theta results to the final results
            'line_results': line_results,  # Line-specific results added
            'transformer_loading': transformer_loading_results,
            'thermal_storage': thermal_storage_results,  #  thermal storage results
            'thermal_storage_capacity': ts_capacity_results
        }
        
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

def reliability_analysis(net, time_steps, const_load_heatpump, const_load_household, heatpump_scaling_factors_df, T_amb, Bbus):
    # Define thermal storage capacity
    TS_capacity = par.TS_capacity
    # Define the number of scenarios
    N_scenarios = par.N_scenarios

    # Generate failure schedules for all scenarios
    # trafo_failures, hp_failures = generate_all_failure_schedules(
    #     par.failure_rate_trafo, par.repair_time_trafo,
    #     par.failure_rate_hp, par.repair_time_hp,
    #     time_steps, N_scenarios
    # )

    # Generate deterministic failure schedules for all scenarios
    trafo_failures, hp_failures = generate_all_deterministic_failure_schedules(
        par.failure_timestep, par.repair_time_trafo,
        par.repair_time_hp, time_steps, N_scenarios
    )
    ## debug failure schedule
    # Plot the failure schedules
    plt.figure(figsize=(10, 6))

    # Plot transformer failures
    plt.plot(time_steps, trafo_failures[0], label="Transformer Failures", drawstyle="steps-post", linewidth=2)

    # Plot heat pump failures
    plt.plot(time_steps, hp_failures[0], label="Heat Pump Failures", drawstyle="steps-post", linewidth=2, linestyle="--")

    # Customize the plot
    plt.title("Failure Schedules for Transformer and Heat Pump", fontsize=14)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Operational Status (1=Operational, 0=Failed)", fontsize=12)
    plt.ylim(-0.1, 1.1)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Show the plot
    plt.show()

    print("generated failure schedules")

    # Run Sequential Monte Carlo Simulation
    results_rel = []

    for scenario in range(N_scenarios):
        print(f"Running scenario {scenario + 1}/{N_scenarios}...")

        # Use pre-generated failure schedules
        trafo_failure = trafo_failures[scenario]
        hp_failure = hp_failures[scenario]

        # Convert failure schedules to flags
        TRAFO_FAIL = {t: int(trafo_failure[i] == 0) for i, t in enumerate(time_steps)}  # 1 if failed
        HP_FAIL = {t: int(hp_failure[i] == 0) for i, t in enumerate(time_steps)}        # 1 if failed

        # Solve original OPF without failures
        original_results = rs.load_optim_results("drcc_results.pkl")

        # Initialize results dataframe with original results
        results_df = {
            "pv_gen": original_results["pv_gen"].copy(),
            "load": original_results["load"].copy(),
            "ext_grid_import": original_results["ext_grid_import"].copy(),
            "ext_grid_export": original_results["ext_grid_export"].copy(),
            "theta": original_results["theta"].copy(),
            "line_results": original_results["line_results"].copy(),
            "transformer_loading": original_results["transformer_loading"].copy(),
            "thermal_storage": original_results["thermal_storage"].copy(),
            "thermal_storage_capacity": original_results["thermal_storage_capacity"].copy(),
        }

        # Initialize state variables
        failure_triggered = False


        for t in time_steps:
            # Check if a failure has occurred
            if not failure_triggered and TRAFO_FAIL[t] == 0 and HP_FAIL[t] == 0:
                # No failure yet: Use original results
                continue

            # A failure has occurred or failure mode is active
            failure_triggered = True

            # Define rolling horizon
            rolling_horizon_steps = [s for s in time_steps if t <= s < t + par.horizon]

            # Extract initial storage state directly from results_df
            initial_storage_state = {
                bus: results_df["thermal_storage"]["ts_sof"][t][bus] * TS_capacity
                for bus in results_df["thermal_storage"]["ts_sof"][t]
            }

            # Solve OPF for the rolling horizon
            results_opf = solve_opf_with_failures(
                net, rolling_horizon_steps, const_load_heatpump, const_load_household, 
                heatpump_scaling_factors_df, T_amb, Bbus, TS_capacity, 
                TRAFO_FAIL, HP_FAIL, initial_storage_state
            )
            # Update results for the rolling horizon steps
            for t_rolling in rolling_horizon_steps:
                if t_rolling >= len(time_steps):
                    break
                results_df["pv_gen"][t_rolling] = results_opf["pv_gen"][t_rolling]
                results_df["load"][t_rolling]["flexible_load"] = results_opf["load"][t_rolling]["flexible_loads"]
                results_df["load"][t_rolling]["non_flexible_load"] = results_opf["load"][t_rolling]["non_flexible_loads"]
                results_df["load"][t_rolling]["HNS"] = results_opf["load"][t_rolling]["HNS"]
                results_df["ext_grid_import"][t_rolling] = results_opf["ext_grid_import"][t_rolling]
                results_df["ext_grid_export"][t_rolling] = results_opf["ext_grid_export"][t_rolling]
                results_df["theta"][t_rolling] = results_opf["theta"][t_rolling]
                results_df["line_results"][t_rolling]["line_pl_mw"] = results_opf["line_results"][t_rolling]["line_pl_mw"]
                results_df["line_results"][t_rolling]["line_loading_percent"] = results_opf["line_results"][t_rolling]["line_loading_percent"]
                results_df["line_results"][t_rolling]["line_current_mag"] = results_opf["line_results"][t_rolling]["line_current_mag"]
                results_df["transformer_loading"][t_rolling] = results_opf["transformer_loading"][t_rolling]
                results_df["thermal_storage"]["ts_in"][t_rolling] = results_opf["thermal_storage"]["ts_in"][t_rolling]
                results_df["thermal_storage"]["ts_out"][t_rolling] = results_opf["thermal_storage"]["ts_out"][t_rolling]
                results_df["thermal_storage"]["ts_sof"][t_rolling] = results_opf["thermal_storage"]["ts_sof"][t_rolling]

        # Collect scenario results
        HNS_total = sum(
            sum(results_df["load"][t]["HNS"].values()) for t in time_steps
        )        
        final_storage = results_df["thermal_storage"]["ts_sof"][time_steps[-1]]

        results_rel.append({
            "scenario": scenario,
            "HNS": HNS_total,
            "final_storage": final_storage
        })



    # After the for loop over scenarios, aggregate results
    total_HNS = sum([res["HNS"] for res in results_rel])  # Total HNS across all scenarios
    EHNS = total_HNS / N_scenarios  # Expected HNS

    # Print reliability study summary
    print(f"Reliability study for TS_capacity = {TS_capacity} MWh")
    print(f"  Total HNS = {total_HNS:.2f} MWh (Total Heat Not Supplied)")
    print(f"  EHNS = {EHNS:.2f} MWh (Expected Heat Not Supplied)")

    if results_rel is not None:
        rs.save_optim_results(results_rel, "results_rel.pkl")

    return results_rel
