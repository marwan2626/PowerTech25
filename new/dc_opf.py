import numpy as np
import pandas as pd
import pandapower as pp

import numpy as np
import pandas as pd
import pandapower as pp

def dc_load_flow_calculation(net):
    """
    Perform a DC load flow calculation based on the pandapower network.
    """

    # Extract necessary data from pandapower network
    B = pp.topology.calc_B_matrix(net, sparse=False)  # Admittance matrix
    P = np.zeros(len(net.bus))

    # Assume that `net.sgen` contains the generators and `net.load` contains the loads
    for _, sgen in net.sgen.iterrows():
        P[sgen['bus']] += sgen['p_mw']

    for _, load in net.load.iterrows():
        P[load['bus']] -= load['p_mw']

    # Identify the slack bus (typically the ext_grid bus in pandapower)
    slack_bus = net.ext_grid['bus'].iloc[0]

    # Solve the DC load flow, excluding the slack bus
    non_slack_buses = [bus for bus in net.bus.index if bus != slack_bus]
    B_reduced = B[np.ix_(non_slack_buses, non_slack_buses)]
    P_reduced = P[non_slack_buses]

    # Solve for angles at non-slack buses
    theta_non_slack = np.linalg.solve(B_reduced, P_reduced)

    # Insert the slack bus angle (which is zero by definition)
    theta = np.zeros(len(net.bus))
    theta[non_slack_buses] = theta_non_slack

    # Calculate line flows
    line_flows = B.dot(theta)

    # Calculate loading percentages
    line_loading_percent = 100 * np.abs(line_flows) / net.line.max_i_ka

    return theta, line_loading_percent


import gurobipy as gp
from gurobipy import GRB

def optimize_generation(net, pv_generators, time_steps, df_pv, df_load, power_cost=100):
    """
    Optimization function to curtail PV generation to prevent overloading of lines.
    """

    model = gp.Model("opf")
    sgen_p_mw = model.addVars(time_steps, len(pv_generators), lb=0, name="sgen_p_mw")
    ext_grid_p_mw = model.addVars(time_steps, lb=-GRB.INFINITY, name="ext_grid_p_mw")
    ext_grid_abs_p_mw = model.addVars(time_steps, lb=0, name="ext_grid_abs_p_mw")

    # Add constraints
    for t in time_steps:
        total_load_p_mw = df_load.loc[t, 'mult']
        model.addConstr(gp.quicksum(sgen_p_mw[t, i] for i in range(len(pv_generators))) + ext_grid_p_mw[t] == total_load_p_mw, name=f"power_balance_{t}")

        model.addConstr(ext_grid_abs_p_mw[t] >= ext_grid_p_mw[t], name=f"abs_ext_grid_pos_{t}")
        model.addConstr(ext_grid_abs_p_mw[t] >= -ext_grid_p_mw[t], name=f"abs_ext_grid_neg_{t}")

    # Objective: minimize cost
    total_cost = gp.quicksum(power_cost * ext_grid_abs_p_mw[t] for t in time_steps)
    model.setObjective(total_cost, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        optimized_values = {f"sgen_p_mw_{t}": [sgen_p_mw[t, i].X for i in range(len(pv_generators))] for t in time_steps}
        return optimized_values

    return None
