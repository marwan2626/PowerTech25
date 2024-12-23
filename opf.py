# Function to solve OPF problem over a time series
"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
Optimization by Marwan Mostafa 

OPF File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### PACKAGES ####
import gurobipy as gp
from gurobipy import GRB
import numpy as np


#### SCRIPTS ####
import parameters as par
import results as rs

###############################################################################
## OPTIMIZATION FUNCTION ## 
###############################################################################

def solve_opf(net, time_steps, const_load_heatpump, const_load_household, heatpump_scaling_factors_df, Bbus):
    model = gp.Model("opf_with_dc_load_flow")

    # Define the costs for import and export
    import_cost = par.import_cost 
    export_cost = par.export_cost  
    curtailment_cost = par.curtailment_cost  
    flexibility_cost = par.flexibility_cost  

    epsilon = 100e-9  # Small positive value to ensure some external grid usage

    # Extract transformer capacity in MW (assuming sn_mva is in MVA)
    transformer_capacity_mw = net.trafo['sn_mva'].values[0]*2


    # Initialize decision variables
    pv_gen_vars = {}  # Store PV generation decision variables
    ext_grid_import_vars = {}  # Store external grid import power decision variables
    ext_grid_export_vars = {}  # Store external grid export power decision variables
    theta_vars = {}  # Store voltage angle decision variables (radians)
    curtailment_vars = {} # Store decision variables for curtailment
    flexible_load_vars = {}  # New flexible load variables

    # Add thermal storage variables
    ts_in_vars = {t: {} for t in time_steps}
    ts_out_vars = {t: {} for t in time_steps}
    ts_sof_vars = {t: {} for t in time_steps}

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
        max_heat_demand_per_bus = {
            bus: max(flexible_time_synchronized_loads[t].get(bus, 0.0) for t in time_steps)
            for bus in flexible_load_buses
        }

        # Define flexible load variables with global peak limit (par.hp_max_power)
        flexible_load_vars[t] = model.addVars(
            flexible_load_buses,
            lb=0,
            ub={bus: max_heat_demand_per_bus[bus] for bus in flexible_load_buses},
            name=f'flexible_load_{t}'
        )
                            
    # Create a dictionary mapping bus indices to scaling factors using heatpump_scaling_factors_df
    heatpump_scaling_factors_dict = {
        bus: heatpump_scaling_factors_df.loc[heatpump_scaling_factors_df['bus'] == bus, 'p_mw'].values[0]
        for bus in flexible_load_buses
    }
    
    # Add thermal storage variables
    ts_size_mwh_scaled_dict = {
        bus: par.ts_size_mwh * heatpump_scaling_factors_dict[bus] for bus in flexible_load_buses
    }     
    # Update SOF constraints for each flexible load bus
    for t_idx, t in enumerate(time_steps):
        for bus in flexible_load_buses:
            ts_in_vars[t][bus] = model.addVar(lb=0, ub=par.ts_in_max, name=f'ts_in_{t}_{bus}')
            ts_out_vars[t][bus] = model.addVar(lb=0, ub=par.ts_out_max, name=f'ts_out_{t}_{bus}')
            ts_sof_vars[t][bus] = model.addVar(lb=0, ub=1.0, name=f'ts_sof_{t}_{bus}')  # SOF as percentage (0 to 1)

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
                        flexible_load_vars[t][bus] + ((ts_out_vars[t][bus] - ts_in_vars[t][bus]) / par.COP) >= heat_demand,
                        name=f'heat_demand_coverage_{t}_{bus}'
                    )

                    # 3. Storage Charging: use excess power for storage charging if available
                    model.addConstr(
                        ts_in_vars[t][bus] <= flexible_load_vars[t][bus] * par.COP,
                        name=f'storage_charging_{t}_{bus}'
                    )

                    model.addConstr(
                        ts_out_vars[t][bus] <= ts_size_mwh_scaled_dict[bus]/24,
                        name=f'storage_charging_{t}_{bus}'
                    )

                    # Update the state of fill (SOF) of the storage
                    # Initial SOF Constraint for the first timestep
                    if t == time_steps[0]:
                        model.addConstr(
                            ts_sof_vars[t][bus] == par.ts_sof_init,
                            name=f'storage_initial_sof_{bus}'
                        )
                    else:
                        # Update SOF based on the previous timestep
                        model.addConstr(
                            ts_sof_vars[t][bus] == ts_sof_vars[time_steps[t - 1]][bus] + (par.ts_eff * ts_in_vars[t][bus] - (ts_out_vars[t][bus]/par.ts_eff)) / ts_size_mwh_scaled_dict[bus],
                            name=f'storage_state_update_{t}_{bus}'
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

            # Now, calculate the line loading percentage using the auxiliary variable
            if hasattr(line, 'max_i_ka'):
                line_loading_percent = 100 * (abs_current_mag_ka / line.max_i_ka)
                model.addConstr(abs_current_mag_ka <= (line.max_i_ka), 
                            name=f'abs_current_mag_constraint_{t}_{line.Index}')

            # Store results for each line in the time step
            line_results[t]["line_pl_mw"][line.Index] = power_flow_mw
            line_results[t]["line_loading_percent"][line.Index] = line_loading_percent            
            line_results[t]["line_current_mag"][line.Index] = current_mag_ka

        transformer_loading_results[t] = model.addVar(lb=0, ub=100, name=f'transformer_loading_{t}')
        model.addConstr(
            transformer_loading_results[t] == ((ext_grid_import_vars[t] + ext_grid_export_vars[t]) / transformer_capacity_mw) * 100,
            name=f'transformer_loading_percentage_{t}'
        )

    # Objective: Minimize total cost (import, export, and curtailment costs)
    total_cost = gp.quicksum(
        import_cost * ext_grid_import_vars[t] +
        export_cost * ext_grid_export_vars[t] +
        (gp.quicksum(curtailment_cost * curtailment_vars[t][bus] for bus in pv_buses) if len(pv_buses) > 0 else 0) +
        gp.quicksum(flexibility_cost * (ts_out_vars[t][bus]+ts_in_vars[t][bus]) for bus in flexible_load_buses)
        for t in time_steps
    )
    model.setObjective(total_cost, GRB.MINIMIZE)

    # After adding all constraints and variables
    model.setParam('OutputFlag', 0)
    model.setParam('Presolve', 0)
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
            transformer_loading_results[t] = transformer_loading_results[t].x
            
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
            'thermal_storage': thermal_storage_results  #  thermal storage results
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
    
