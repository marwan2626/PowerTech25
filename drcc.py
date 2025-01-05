"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
Optimization by Marwan Mostafa 

DRCC-OPF File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### PACKAGES ####
import gurobipy as gp
from gurobipy import GRB
import pickle as pkl
import numpy as np
import copy

#### SCRIPTS ####
import parameters as par 
import results as rs
###############################################################################
## AUXILIARY FUNCTIONS ##
###############################################################################

def extract_line_power(line_results, time_step):
    # Extracts the current magnitude for all lines at a given time_step
    return {line: line_results[time_step]['line_pl_mw'][line] for line in line_results[time_step]['line_pl_mw'].keys()}

###############################################################################
## DRCC-OPF FUNCTIONS ##
###############################################################################

### calculate the covariance matrix ###
def calculate_covariance_matrix(heatpumpForecast):
    # Ensure 'sigma P' is present in the DataFrame
    if 'stdP_NORM' not in heatpumpForecast.columns:
        raise ValueError("DataFrame must contain 'stdP' for standard deviation values.")
    
    #max_mean = heatpumpForecast['meanP'].max()
    #heatpumpForecast['meanP_norm'] = heatpumpForecast['meanP'] / max_mean
    #heatpumpForecast['stdP_NORM'] = heatpumpForecast['stdP'] / max_mean

    # Extract the 'stdP' values and square them to get the variance
    variance = (heatpumpForecast['stdP_NORM']) ** 2

    # Create a diagonal covariance matrix from the variance
    covariance_matrix = np.diag(variance)
    #print(f'covariance matrix: {covariance_matrix}')

    return covariance_matrix

### calculate the sensitivity ###
def calculate_sensitivity(heatpumpForecast, opf_results, time_steps):
    # Check if 'line_results' exists in opf_results
    if 'line_results' not in opf_results:
        raise KeyError("'line_results' key missing from opf_results. Ensure solve_opf populates it correctly.")
    
    # Initialize line currents at the first timestep
    powerflows_prev = extract_line_power(opf_results['line_results'], time_steps[0])
    sensitivity_results = {t: {} for t in time_steps[1:]}  # Structured by timestep first

    # Iterate over each timestep to calculate sensitivity
    for t in range(1, len(time_steps)):
        # Calculate forecasting error ω for current and previous timestep
        w_t = float(heatpumpForecast['stdP_NORM'].iloc[t])

        # Get current line currents from OPF results
        powerflows_t = extract_line_power(opf_results['line_results'], time_steps[t])

        # Calculate sensitivity for each line relative to the previous timestep
        for line in powerflows_prev.keys():
            sensitivity_value = (powerflows_t[line] - powerflows_prev[line]) / ((w_t))
            sensitivity_results[time_steps[t]][line] = float(sensitivity_value)

        # Update `line_currents_prev` to the current timestep’s line currents
        powerflows_prev = powerflows_t

    return sensitivity_results

def calculate_sensitivity_trafo(heatpumpForecast, opf_results, time_steps):
    # Check if 'line_results' exists in opf_results
    if 'transformer_loading' not in opf_results:
        raise KeyError("'transformer_loading' key missing from opf_results. Ensure solve_opf populates it correctly.")
    
    # Initialize line currents at the first timestep
    transformer_loading_prev = opf_results['transformer_loading'][0]
    print(f"transformer loading prev: {transformer_loading_prev}")
    sensitivity_trafo_results = {t: {} for t in time_steps[1:]}  # Structured by timestep first

    # Iterate over each timestep to calculate sensitivity
    for t in range(1, len(time_steps)):
        # Calculate forecasting error ω for current and previous timestep
        w_t = float(heatpumpForecast['stdP_NORM'].iloc[t])

        # Get current line currents from OPF results
        transformer_loading_t = opf_results['transformer_loading'][t]
        print(f"transformer loading: {transformer_loading_t}")

        # Calculate sensitivity for each line relative to the previous timestep

        sensitivity_trafo_value = (transformer_loading_t - transformer_loading_prev) / ((w_t))
        sensitivity_trafo_results[time_steps[t]] = float(sensitivity_trafo_value)

        # Update `line_currents_prev` to the current timestep’s line currents
        transformer_loading_prev = transformer_loading_t

    return sensitivity_trafo_results

### calculate the omega I ###
def calculate_omega_P(alpha, sensitivity, cov_matrix, Omega_P):
    print('Calculating omega I')
    
    # Initialize Omega_I_new based on the provided structure of Omega_I
    Omega_P_new = {t: {line: 0 for line in Omega_P[t]} for t in Omega_P}
    
    # Compute scaling factor
    scaling_factor = (1 - alpha) / (alpha) # 95% confidence interval

    # Compute the square root of the covariance matrix
    cov_sqrt = np.sqrt(cov_matrix)

    # Iterate over each timestep and line sensitivity
    for t, sensitivity_t in sensitivity.items():
        for line, sensitivity_value in sensitivity_t.items():
            # Scale each sensitivity value with the square root of the covariance matrix
            scaled_sensitivity = sensitivity_value * cov_sqrt 

            # Calculate the L2 norm of the scaled sensitivity vector
            Omega_P_new[t][line] = scaling_factor * np.linalg.norm(scaled_sensitivity) * 1e-3 #kA

    return Omega_P_new

def calculate_omega_P_trafo(alpha, sensitivity_trafo, cov_matrix, Omega_P_trafo, time_steps):
    print('Calculating omega I')
    print(f"sensitivity_trafo: {sensitivity_trafo}")
    # Initialize Omega_I_new based on the provided structure of Omega_I
    Omega_P_trafo_new = {t: 0 for t in time_steps}
    
    
    # Compute scaling factor
    scaling_factor = (1 - alpha) / (alpha) # 95% confidence interval

    # Compute the square root of the covariance matrix
    cov_sqrt = np.sqrt(cov_matrix)

    # Iterate over each timestep and line sensitivity
    for t in time_steps[1:]:
        scaled_trafo_sensitivity = sensitivity_trafo[t] * cov_sqrt 
        # Calculate the L2 norm of the scaled sensitivity vector
        Omega_P_trafo_new[t] = scaling_factor * np.linalg.norm(scaled_trafo_sensitivity) * 1e-5
        Omega_P_trafo_new[0] = 0

    return Omega_P_trafo_new

### solve the OPF problem ###

def solve_opf(net, time_steps, const_load_heatpump, const_load_household, heatpump_scaling_factors_df, T_amb, Bbus, Omega_P, Omega_P_Trafo):
    model = gp.Model("opf_with_dc_load_flow")

    # Define the costs for import and export
    import_cost = par.import_cost 
    export_cost = par.export_cost  
    curtailment_cost = par.curtailment_cost  
    flexibility_cost = par.flexibility_cost  
    storage_cost = par.c_cost

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
    ts_capacity_vars = {}  # Store thermal storage capacity decision variables

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

    # Add thermal storage variables
    ts_capacity_vars = model.addVars(
        flexible_load_buses,
        lb=0.001,
        name=f'ts_capacity'
    )

    COP = {}  # Coefficient of Performance (COP) for heat pumps
    for t in time_steps:
        COP[t] = par.eta_c0 * (T_amb[t] + par.DeltaT)/(par.T_S - T_amb[t] + 2*par.DeltaT)
    print(f"COP: {COP}")

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

        # Compute maximum heat demand for each flexible load bus over the day
        # max_heat_demand_per_bus = {
        #     bus: max(flexible_time_synchronized_loads[t].get(bus, 0.0) for t in time_steps)
        #     for bus in flexible_load_buses
        # }

        # Define flexible load variables with global peak limit (par.hp_max_power)
        flexible_load_vars[t] = model.addVars(
            flexible_load_buses,
            lb=0,
            # ub={bus: max_heat_demand_per_bus[bus] for bus in flexible_load_buses},
            name=f'flexible_load_{t}'
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

                    # 1. Heat Demand Coverage: flexible_load_vars and/or thermal storage must meet the demand (all in electrical equivalent)
                    model.addConstr(
                        flexible_load_vars[t][bus] + ((ts_out_vars[t][bus] - ts_in_vars[t][bus]) / COP[t]) == heat_demand,
                        name=f'heat_demand_coverage_{t}_{bus}'
                    )

                    # 3. Storage Charging: use excess power for storage charging if available
                    model.addConstr(
                        ts_in_vars[t][bus] <= flexible_load_vars[t][bus] * COP[t],
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
                            ts_out_vars[t][bus] / par.ts_eff,
                            name=f'storage_energy_update_{t}_{bus}'
                        )

                    model.addConstr(
                        ts_sof_vars[t][bus] * ts_capacity_vars[bus] == energy_stored_vars[t][bus],
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

        # Total power balance constraint at the slack bus
        # This enforces that the slack bus always balances generation and demand
        total_generation = gp.quicksum(pv_gen_vars[t][bus] for bus in pv_buses) if pv_buses.size > 0 else 0
        total_load = gp.quicksum(flexible_load_vars[t][bus] for bus in flexible_load_buses) + gp.quicksum(net.load.loc[net.load.bus == bus, 'p_mw'].values[0] 
                                for bus in net.load.bus.values if bus not in flexible_load_buses)        
        model.addConstr(ext_grid_import_vars[t] - ext_grid_export_vars[t] == total_load - total_generation, name=f'power_balance_slack_{t}')

    # Enforce final state of fill to match initial state (0.5) for all flexible load buses
    for bus in flexible_load_buses:
        model.addConstr(ts_sof_vars[time_steps[-1]][bus] == 0.5, name=f'final_sof_balance_{bus}')


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
                model.addConstr(abs_power_flow_mw <= (line.max_i_ka * (sqrt3 * (base_voltage / 1e3))-Omega_P[t][line.Index]), 
                            name=f'abs_power_flow_constraint_{t}_{line.Index}')

            # Store results for each line in the time step
            line_results[t]["line_pl_mw"][line.Index] = power_flow_mw
            line_results[t]["line_loading_percent"][line.Index] = line_loading_percent            
            line_results[t]["line_current_mag"][line.Index] = current_mag_ka

        transformer_loading_vars[t] = model.addVar(lb=0, ub=transformer_capacity_mw, name=f'transformer_loading_{t}')
        transformer_loading_perc_vars[t] = model.addVar(lb=0, ub=100, name=f'transformer_loading_percent_{t}')
        model.addConstr(
            transformer_loading_vars[t] == (ext_grid_import_vars[t] + ext_grid_export_vars[t]) / (transformer_capacity_mw-Omega_P_Trafo[t]),
            name=f'transformer_loading_{t}'
        )
        model.addConstr(
            transformer_loading_perc_vars[t] == (transformer_loading_vars[t] / transformer_capacity_mw) * 100,
            name=f'transformer_loading_percent_constr_{t}'
        )

    # Objective: Minimize total cost (import, export, and curtailment costs)
    total_cost = gp.quicksum(
        import_cost * ext_grid_import_vars[t] +
        export_cost * ext_grid_export_vars[t] +
        (gp.quicksum(curtailment_cost * curtailment_vars[t][bus] for bus in pv_buses) if len(pv_buses) > 0 else 0) +
        gp.quicksum(flexibility_cost * (ts_out_vars[t][bus]+ts_in_vars[t][bus]) for bus in flexible_load_buses)
        for t in time_steps
    ) + gp.quicksum(storage_cost * ts_capacity_vars[bus] for bus in flexible_load_buses)
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
        print(f"OPF Optimal Objective Value: {model.ObjVal}")
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
            }

            # Extract thermal storage results for each flexible load bus
            thermal_storage_results['ts_in'][t] = {bus: ts_in_vars[t][bus].x for bus in flexible_load_buses}
            thermal_storage_results['ts_out'][t] = {bus: ts_out_vars[t][bus].x for bus in flexible_load_buses}
            thermal_storage_results['ts_sof'][t] = {bus: ts_sof_vars[t][bus].x for bus in flexible_load_buses}

            ts_capacity_results['capacity'] = {bus: ts_capacity_vars[bus] for bus in flexible_load_buses}

            # Extract numerical values for line results
            for line in net.line.itertuples():
                line_results[t]["line_pl_mw"][line.Index] = line_results[t]["line_pl_mw"][line.Index].getValue()
                line_results[t]["line_loading_percent"][line.Index] = line_results[t]["line_loading_percent"][line.Index].getValue()
                line_results[t]["line_current_mag"][line.Index] = line_results[t]["line_current_mag"][line.Index].getValue()

            # After optimization, print the key variable results
            #print(f"Time Step {t}:")
            #print(f"PV Generation: {[pv_gen_vars[t][bus].x for bus in pv_buses]}")
            #print(f"External Grid Import: {ext_grid_import_vars[t].x}")
            #print(f"External Grid Export: {ext_grid_export_vars[t].x}")
            #print(f"Theta (angles): {[theta_vars[t][bus].x for bus in net.bus.index]}")
            #print(f"Thermal Storage In: {thermal_storage_results['ts_in'][t]}")
            #print(f"Thermal Storage Out: {thermal_storage_results['ts_out'][t]}")
            #print(f"Thermal Storage SOF: {thermal_storage_results['ts_sof'][t]}")
            print(f"thermal storage capacity: {ts_capacity_results['capacity']}")
            #print(f"Transformer Loading: {transformer_loading_vars[t].x}")

            #for line in net.line.itertuples():
                #print(f"Line {line.Index}: Power Flow MW = {line_results[t]['line_pl_mw'][line.Index]}, Loading % = {line_results[t]['line_loading_percent'][line.Index]}")


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

        # Save the results to a file
        if results is not None:
            rs.save_optim_results(results, "opf_results.pkl")
        
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
    



### DRCC-OPF function ###
def drcc_opf(net, time_steps, const_load_heatpump, const_load_household, Bbus, heatpumpForecast, heatpump_scaling_factors_df, T_amb, max_iter_drcc, alpha, eta):
    # Step 1: Calculate covariance matrix once
    cov_matrix = calculate_covariance_matrix(heatpumpForecast)
    
    # Step 2: Initial OPF run with Omega_I = 0 (no constraint tightening)
    Omega_P_init = {t: {line.Index: 0 for line in net.line.itertuples()} for t in time_steps}
    Omega_P_trafo_init = {t: 0 for t in time_steps}
    print(f"Omega_P_init {Omega_P_trafo_init}")

    drcc_opf_results = solve_opf(net, time_steps, const_load_heatpump, const_load_household, heatpump_scaling_factors_df, T_amb, Bbus, Omega_P_init, Omega_P_trafo_init)
    if drcc_opf_results is None:
        print("Initial OPF failed with Omega_P = 0")
        return None

    Omega_P_prev = Omega_P_init  # Store initial Omega_I as previous for first iteration
    Omega_P_trafo_prev = Omega_P_trafo_init  # Store initial Omega_I as previous for first iteration

    previous_max_diff = None  # To store max_diff from the previous iteration

    for drcc_iter in range(max_iter_drcc):
        print(f"Calculating sensitivity for iteration {drcc_iter + 1}")
        
        # Step 3: Calculate sensitivity based on the latest OPF results
        sensitivity = calculate_sensitivity(heatpumpForecast, drcc_opf_results, time_steps)
        sensitivity_trafo = calculate_sensitivity_trafo(heatpumpForecast, drcc_opf_results, time_steps)
        print(f"Sensitivity calculated for DRCC iteration {drcc_iter + 1}")
        
        # Step 4: Calculate Omega_I using the updated sensitivity and covariance matrix
        Omega_P = calculate_omega_P(alpha, sensitivity, cov_matrix, Omega_P_init)
        Omega_P_trafo = calculate_omega_P_trafo(alpha, sensitivity_trafo, cov_matrix, Omega_P_trafo_init, time_steps)
        print(f"Omega_P_trafo: {Omega_P_trafo}")
        # Apply threshold limit to Omega_I
        for t in Omega_P:
            for line in Omega_P[t]:
                if Omega_P[t][line] >= 1:
                    Omega_P[t][line] = 1

        # for t in Omega_P_trafo[1:]:
        #     if Omega_P_trafo[t] >= 10:
        #         Omega_P_trafo[t] = 10

        print(f"Omega_P calculated for DRCC iteration {drcc_iter + 1}")
        # for t in Omega_I:
        #     omega_values = list(Omega_I[t].values())
        #     print(f"Timestep {t}: Omega_I min={min(omega_values)}, max={max(omega_values)}, mean={np.mean(omega_values)}")

        max_diff = 0
        max_diff_timestep = None
        max_diff_line = None
        for t in time_steps:
            for line in Omega_P[t].keys():
                diff = np.abs(Omega_P[t][line] - Omega_P_prev[t][line])
                if diff > max_diff:
                    max_diff = diff
                    max_diff_timestep = t
                    max_diff_line = line

        # Print the max difference and the location details
        print(f"DRCC Iteration {drcc_iter + 1}: Max Omega_P difference = {max_diff} at Timestep {max_diff_timestep}, Line {max_diff_line}")

        # Check convergence based on change in max_diff between consecutive iterations
        if previous_max_diff is not None and abs(max_diff - previous_max_diff) < eta:
            print(f"Converged in {drcc_iter + 1} DRCC iterations based on max difference convergence.")
            break

        # Update Omega_P_prev and previous_max_diff for next iteration
        Omega_P_prev = copy.deepcopy(Omega_P)
        previous_max_diff = max_diff

        # Re-run OPF with the updated Omega_P
        
        drcc_opf_results = solve_opf(net, time_steps, const_load_heatpump, const_load_household, heatpump_scaling_factors_df, T_amb, Bbus, Omega_P, Omega_P_trafo)
        
        if drcc_opf_results is None:
            print(f"OPF infeasible in DRCC iteration {drcc_iter + 1}")
            return None
        
    # Save the results to a file
    if drcc_opf_results is not None:
        rs.save_optim_results(drcc_opf_results, "drcc_opf_results.pkl")

    return drcc_opf_results


import griddata as gd

### calculate the sensitivity ###
def calculate_sensitivity2(heatpumpForecast, opf_results, initial_opf_results, time_steps):
    # Check if 'line_results' exists in opf_results
    if 'line_results' not in opf_results:
        raise KeyError("'line_results' key missing from opf_results. Ensure solve_opf populates it correctly.")
    
    sensitivity_results = {t: {} for t in time_steps}
    #debug_line = 24
    
    for t in time_steps:
        # Calculate forecasting error ω for current and previous timestep
        w_t = float(heatpumpForecast['stdP_NORM'].iloc[t])

        current_line_powers = extract_line_power(opf_results['line_results'], t)
        initial_line_powers= extract_line_power(initial_opf_results['line_results'], t)
        
        # if debug_line in current_line_currents and debug_line in initial_line_currents:
        #     print(f"Time step {t}, Line {debug_line}:")
        #     print(f"  Current Line Current = {current_line_currents[debug_line]:.6f}")
        #     print(f"  Initial Line Current = {initial_line_currents[debug_line]:.6f}")
        #     print(f"  Forecasting Error w_t = {w_t:.6f}")


        # Calculate sensitivity for each line relative to the previous timestep
        for line in initial_line_powers.keys():
            sensitivity_value = (current_line_powers[line] - initial_line_powers[line]) / (w_t)
            sensitivity_results[time_steps[t]][line] = float(sensitivity_value)

    # for t in time_steps:
    #     for line, value in sensitivity_results[t].items():
    #         print(f"Time step {t}, Line {line}: Sensitivity = {value:.6f}")
    
    return sensitivity_results

def drcc_opf2(net, time_steps, const_load_heatpump, const_load_household, Bbus, heatpumpForecast, heatpumpReal, heatpump_scaling_factors_df, max_iter_drcc, alpha, eta, T_amb):
    # Step 1: Initialize covariance matrix
    season = 'winter'
    #net_forecast, const_load_household_fc, const_load_heatpump_fc, time_steps_fc, df_season_heatpump_prognosis, df_household, df_heatpump, heatpump_scaling_factors_fc_df = gd.setup_grid_irep_forecast(season)

    Omega_P_init = {t: {line.Index: 0 for line in net.line.itertuples()} for t in time_steps}
    print("Initializing Omega_P = 0")
    print("Solving forecast OPF")
    initial_results = solve_opf(net, time_steps, const_load_heatpump, const_load_household, heatpump_scaling_factors_df, T_amb, Bbus, Omega_P_init)
    print("Solving Initial OPF with Omega_P = 0")
    drcc_opf_results = solve_opf(net, time_steps, const_load_heatpump, const_load_household, heatpump_scaling_factors_df, T_amb, Bbus, Omega_P_init)
    #calcualte sensitivity from forecast & real data
    sensitivity = calculate_sensitivity2(heatpumpForecast, heatpumpReal, drcc_opf_results, initial_results, time_steps)
    print(f"Initial Sensitivity calculated")
    cov_matrix = calculate_covariance_matrix(heatpumpForecast)
    print(f"Covariance Matric Calculated")
    Omega_P_prev = Omega_P_init
    previous_max_diff = None
    if drcc_opf_results is None:
        print("Initial OPF failed with Omega_P = 0")
        return None

    for drcc_iter in range(max_iter_drcc):
        print(f"Starting DRCC iteration {drcc_iter + 1}")

        # Step 4: Calculate Omega_I using the updated sensitivity and covariance matrix
        Omega_P = calculate_omega_P(alpha, sensitivity, cov_matrix, Omega_P_prev)
        #Apply threshold limit to Omega_I
        for t in Omega_P:
            for line in Omega_P[t]:
                if Omega_P[t][line] >= 1:
                    Omega_P[t][line] = 1

        print(f"Omega_P calculated for DRCC iteration {drcc_iter + 1}")
        # for t in Omega_I:
        #      omega_values = list(Omega_I[t].values())
        #      print(f"Timestep {t}: Omega_I min={min(omega_values)}, max={max(omega_values)}, mean={np.mean(omega_values)}")

        max_diff = 0
        max_diff_timestep = None
        max_diff_line = None
        all_diffs_zero = True
        for t in time_steps:
            for line in Omega_P[t].keys():
                diff = np.abs(Omega_P[t][line] - Omega_P_prev[t][line])
                if diff > max_diff:
                    max_diff = diff
                    max_diff_timestep = t
                    max_diff_line = line
                if diff > 0:  # Check if any difference is non-zero
                    all_diffs_zero = False
                # Debugging for each line
                #print(f"Timestep {t}, Line {line}: Omega_I = {Omega_I[t][line]:.6f}, Previous = {Omega_I_prev[t][line]:.6f}, Diff = {diff:.6f}")

        # Print the max difference and the location details
        print(f"DRCC Iteration {drcc_iter + 1}: Max Omega_P difference = {max_diff} at Timestep {max_diff_timestep}, Line {max_diff_line}")

        if all_diffs_zero:
            print(f"Converged in {drcc_iter + 1} DRCC iterations as all differences are zero.")
            break
        # Check convergence based on change in max_diff between consecutive iterations
        if previous_max_diff is not None and abs(max_diff - previous_max_diff) < eta:
            print(f"Converged in {drcc_iter + 1} DRCC iterations based on max difference convergence.")
            break

        if (drcc_iter > 1) and abs(max_diff < eta):
            print(f"Converged in {drcc_iter + 1} DRCC iterations based on max difference convergence.")
            break

        # Update Omega_I_prev and previous_max_diff for next iteration
        Omega_P_prev = copy.deepcopy(Omega_P)
        previous_max_diff = copy.deepcopy(max_diff)

        # Re-run OPF with the updated Omega_I
        drcc_opf_results = solve_opf(net, time_steps, const_load_heatpump, const_load_household, heatpump_scaling_factors_df, T_amb, Bbus, Omega_P)
        
        #Calculate sensitivity based on the latest OPF results
        sensitivity = calculate_sensitivity2(heatpumpForecast, heatpumpReal, drcc_opf_results, initial_results, time_steps)
        print(f"Sensitivity calculated for DRCC iteration {drcc_iter + 1}")

        if drcc_opf_results is None:
            print(f"OPF infeasible in DRCC iteration {drcc_iter + 1}")
            return None

    # Save the results to a file
    if drcc_opf_results is not None:
        rs.save_optim_results(drcc_opf_results, "drcc_opf_results.pkl")


    return drcc_opf_results
