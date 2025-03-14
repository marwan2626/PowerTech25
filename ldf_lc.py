import pandas as pd
import numpy as np
import networkx as nx

def calculate_z_matrix(net):
    """Calculate the impedance matrix (Z) for all lines and transformers."""
    num_branches = len(net.line) + len(net.trafo)  # Total branches
    Z = np.zeros(num_branches, dtype=complex)  # Initialize impedance vector
    
    # System base values
    base_MVA = net.sn_mva  

    # Process lines
    for i, line in enumerate(net.line.itertuples()):
        from_bus = line.from_bus
        to_bus = line.to_bus
        
        # Base impedance for per-unit conversion
        base_voltage = net.bus.at[from_bus, 'vn_kv']  # Use "from bus" voltage as reference
        Z_base = base_voltage ** 2 / base_MVA
        
        # Compute per-unit impedance
        r_pu = line.r_ohm_per_km * line.length_km / Z_base
        x_pu = line.x_ohm_per_km * line.length_km / Z_base
        Z[i] = r_pu + 1j * x_pu  # Store complex impedance

    # Process transformers
    for j, trafo in enumerate(net.trafo.itertuples(), start=len(net.line)):
        trafo_base_mva = trafo.sn_mva  # Transformer rated MVA
        system_base_mva = base_MVA  # System base MVA
        
        # Compute per-unit impedance
        z_pu = (trafo.vk_percent / 100) * (system_base_mva / trafo_base_mva)
        r_pu = (trafo.vkr_percent / 100) * (system_base_mva / trafo_base_mva)
        x_pu = np.sqrt(z_pu**2 - r_pu**2)  # Solve for reactance
        
        Z[j] = r_pu + 1j * x_pu  # Store complex impedance
    
    return Z

def compute_incidence_matrix(net):
    """Constructs the correct incidence matrix A for the network."""
    A_bus = np.zeros((len(net.bus), len(net.line) + len(net.trafo)))  # Extend for transformers

    # Process lines
    for idx, line in net.line.iterrows():
        A_bus[int(line.from_bus), idx] = 1   # Sending end
        A_bus[int(line.to_bus), idx] = -1    # Receiving end

    # Process transformers
    for idx, trafo in net.trafo.iterrows():
        trafo_index = len(net.line) + idx  # Continue indexing after lines
        A_bus[int(trafo.hv_bus), trafo_index] = 1    # High-voltage side
        A_bus[int(trafo.lv_bus), trafo_index] = -1   # Low-voltage side

    # Save for debugging
    A_df = pd.DataFrame(A_bus, index=net.bus.index, columns=list(net.line.index) + list(net.trafo.index))
    A_df.to_csv("fixed_incidence_matrix_A.csv")
    print("Fixed incidence matrix saved as 'fixed_incidence_matrix_A.csv'.")

    return A_bus

def calculate_gbus_matrix(net):
    """Calculate the conductance matrix (Gbus) for LDF-LC."""
    num_buses = len(net.bus)
    Gbus = np.zeros((num_buses, num_buses))  # Initialize Gbus matrix
    
    base_MVA = net.sn_mva  # System base MVA
    
    # Add line resistances
    for line in net.line.itertuples():
        from_bus = line.from_bus
        to_bus = line.to_bus
        
        # Convert resistance to per-unit (R_pu = R_ohm / (Base Voltage^2 / Base MVA))
        base_voltage = net.bus.at[from_bus, 'vn_kv']  # Base voltage for the from bus
        Z_base = base_voltage ** 2 / net.sn_mva  # Calculate base impedance
        Y_base = 1 / Z_base  # Calculate base admittance
        x_pu = line.x_ohm_per_km * line.length_km / Z_base
        r_pu = line.r_ohm_per_km * line.length_km / Z_base
        
        Y_series = 1 / (r_pu + 1j * x_pu)  # Series admittance
        #print(f"Y_series: {Y_series}")
        G_pu = Y_series.real  # Conductance in per-unit
        #print(f"G_pu: {G_pu}")
        
        # Gbus off-diagonal elements
        Gbus[from_bus, to_bus] -= G_pu 
        Gbus[to_bus, from_bus] -= G_pu 
        
        # Gbus diagonal elements
        Gbus[from_bus, from_bus] += G_pu 
        Gbus[to_bus, to_bus] += G_pu 
    
    # Add transformer resistances
    for trafo in net.trafo.itertuples():
        hv_bus = trafo.hv_bus
        lv_bus = trafo.lv_bus
        
        trafo_base_mva = trafo.sn_mva  # Extract transformer base MVA
        system_base_mva = net.sn_mva  # System base MVA


        # Compute correct transformer impedance values
        z_pu = (trafo.vk_percent / 100)*(system_base_mva/trafo_base_mva)  # Total impedance in per-unit
        #print(f"z_pu: {z_pu}")

        r_pu = (trafo.vkr_percent / 100)*(system_base_mva/trafo_base_mva)  # Resistance in per-unit
        #print(f"r_pu: {r_pu}")

        x_pu = np.sqrt(z_pu**2 - r_pu**2)  # Reactance computed from Z and R
        #print(f"x_pu: {x_pu}")

        G_pu = r_pu / (r_pu**2 + x_pu**2)  # Conductance in per-unit

        # Gbus off-diagonal elements
        Gbus[hv_bus, lv_bus] -= G_pu 
        Gbus[lv_bus, hv_bus] -= G_pu 
    
        # Gbus diagonal elements
        Gbus[hv_bus, hv_bus] += G_pu 
        Gbus[lv_bus, lv_bus] += G_pu 

    return Gbus


def calculate_bbus_matrix(net):
    """Calculate the susceptance matrix (Bbus) for LDF-LC."""
    num_buses = len(net.bus)
    Bbus = np.zeros((num_buses, num_buses))  # Initialize Bbus matrix
    
    base_MVA = net.sn_mva  # System base MVA
    
    # Add line reactances
    for line in net.line.itertuples():
        from_bus = line.from_bus
        to_bus = line.to_bus
        
        # Convert reactance to per-unit (X_pu = X_ohm / (Base Voltage^2 / Base MVA))
        base_voltage = net.bus.at[from_bus, 'vn_kv']  # Base voltage for the from bus
        Z_base = base_voltage ** 2 / net.sn_mva  # Calculate base impedance
        Y_base = 1 / Z_base  # Calculate base admittance
        x_pu = line.x_ohm_per_km * line.length_km / Z_base
        r_pu = line.r_ohm_per_km * line.length_km / Z_base
        
        Y_series = 1 / (r_pu + 1j * x_pu)  # Series admittance
        #print(f"Y_series: {Y_series}")
        B_pu = Y_series.imag  # Susceptance in per-unit

        
        # Bbus off-diagonal elements
        Bbus[from_bus, to_bus] -= B_pu 
        Bbus[to_bus, from_bus] -= B_pu 
        
        # Bbus diagonal elements
        Bbus[from_bus, from_bus] += B_pu 
        Bbus[to_bus, to_bus] += B_pu 
    
    # Add transformer reactances
    for trafo in net.trafo.itertuples():
        hv_bus = trafo.hv_bus
        lv_bus = trafo.lv_bus
        
        trafo_base_mva = trafo.sn_mva  # Extract transformer base MVA
        system_base_mva = net.sn_mva  # System base MVA


        # Compute correct transformer impedance values
        z_pu = (trafo.vk_percent / 100)*(system_base_mva/trafo_base_mva)  # Total impedance in per-unit
        #print(f"z_pu: {z_pu}")

        r_pu = (trafo.vkr_percent / 100)*(system_base_mva/trafo_base_mva)  # Resistance in per-unit
        #print(f"r_pu: {r_pu}")

        x_pu = np.sqrt(z_pu**2 - r_pu**2)  # Reactance computed from Z and R
        #print(f"x_pu: {x_pu}")

        B_pu = -x_pu / (r_pu**2 + x_pu**2)  # Conductance in per-unit
    
        # Bbus off-diagonal elements
        Bbus[hv_bus, lv_bus] -= B_pu
        Bbus[lv_bus, hv_bus] -= B_pu
    
        # Bbus diagonal elements
        Bbus[hv_bus, hv_bus] += B_pu
        Bbus[lv_bus, lv_bus] += B_pu
    
    return Bbus

import numpy as np
import pandas as pd
import networkx as nx

def compute_downstream_nodes(A, net):
    """Returns a mapping of each bus to its downstream nodes and prints the hierarchy."""
    num_buses, num_branches = A.shape
    graph = nx.DiGraph()

    # Construct directed graph from incidence matrix
    for branch_idx in range(num_branches):
        from_bus = np.where(A[:, branch_idx] == 1)[0][0]
        to_bus = np.where(A[:, branch_idx] == -1)[0][0]
        graph.add_edge(from_bus, to_bus)

    # Perform BFS from the slack bus
    slack_bus = net.ext_grid.bus.iloc[0]
    downstream_map = {bus: [] for bus in range(num_buses)}

    for bus in graph.nodes:
        if bus == slack_bus:
            continue
        predecessors = list(nx.ancestors(graph, bus))
        for pred in predecessors:
            downstream_map[pred].append(bus)

    # Print hierarchy
    #print("\n--- Downstream Node Hierarchy ---")
    #for bus, children in downstream_map.items():
    #    print(f"Bus {bus}: {children}")

    return downstream_map


def compute_branch_power_flow(A, P_mw, Q_mw, downstream_map):
    """Computes accumulated power flow at each branch by summing downstream node demands."""
    num_branches = A.shape[1]
    
    P_branch = np.zeros(num_branches)
    Q_branch = np.zeros(num_branches)

    # Traverse buses in **reverse** order (bottom-up accumulation)
    buses = list(downstream_map.keys())[::-1]  # Reverse hierarchy

    # Create mapping from branches to their buses
    for branch_idx in range(num_branches):
        from_bus = np.where(A[:, branch_idx] == 1)[0][0]
        to_bus = np.where(A[:, branch_idx] == -1)[0][0]

        # Add direct node power demand
        P_branch[branch_idx] += P_mw[to_bus]
        Q_branch[branch_idx] += Q_mw[to_bus]

        # Accumulate downstream power
        for child_bus in downstream_map[to_bus]:
            downstream_branch_idx = np.where((A[:, :] == -1) & (np.arange(A.shape[1]) == child_bus))[1]
            if len(downstream_branch_idx) > 0:
                P_branch[branch_idx] += P_branch[downstream_branch_idx[0]]
                Q_branch[branch_idx] += Q_branch[downstream_branch_idx[0]]

    return P_branch, Q_branch

def compute_Ybus(Gbus, Bbus):
    """Computes the admittance matrix Ybus = Gbus + jBbus."""
    return Gbus + 1j * Bbus



def accumulate_downstream_power(A, P_mw, Q_mw, net):
    """Accumulates downstream power flows using the correct downstream node mapping."""
    num_buses = A.shape[0]

    # Initialize accumulated power with the nodal values
    P_accumulated = P_mw.copy()
    Q_accumulated = Q_mw.copy()

    # Identify slack bus
    slack_bus_index = net.ext_grid.bus.iloc[0]

    # Get correct downstream mappings
    downstream_map = compute_downstream_nodes(A, net)

    # Traverse each bus and accumulate power from its downstream buses
    for bus in range(num_buses):
        if bus == slack_bus_index:
            continue  # Skip slack bus

        for child_bus in downstream_map[bus]:  # All downstream buses
            P_accumulated[bus] += P_mw[child_bus]
            Q_accumulated[bus] += Q_mw[child_bus]

    # Compute power flow contributions for each branch
    branch_to_buses = {idx: set() for idx in net.line.index.tolist() + net.trafo.index.tolist()}
    
    for branch_idx in net.line.index:
        from_bus = np.where(A[:, branch_idx] == 1)[0][0]
        to_bus = np.where(A[:, branch_idx] == -1)[0][0]
        branch_to_buses[branch_idx].update(downstream_map[to_bus])
        branch_to_buses[branch_idx].add(to_bus)

    for trafo_idx in net.trafo.index:
        from_bus = net.trafo.hv_bus.loc[trafo_idx]
        to_bus = net.trafo.lv_bus.loc[trafo_idx]
        
        # Ensure trafo_idx exists in branch_to_buses
        if trafo_idx not in branch_to_buses:
            branch_to_buses[trafo_idx] = set()

        branch_to_buses[trafo_idx].update(downstream_map[to_bus])
        branch_to_buses[trafo_idx].add(to_bus)

    # Print debug info
    #print("\n--- Accumulated Power Contributions ---")
    #for branch_idx, buses in branch_to_buses.items():
    #    if branch_idx in net.line.index:
    #        print(f"Line {branch_idx} accumulates power from buses {sorted(buses)}")
    #    elif int(branch_idx) in net.trafo.index.tolist():
    #        print(f"Transformer {branch_idx} handles power from buses {sorted(buses)}")

    #print("\n--- Transformer Power Contributions ---")
    #for trafo_idx in net.trafo.index:
    #    if trafo_idx in branch_to_buses:
    #        print(f"Transformer {trafo_idx} handles power from buses {sorted(branch_to_buses[trafo_idx])}")
    #    else:
    #        print(f"Transformer {trafo_idx} is missing from branch_to_buses!")

    return P_accumulated, Q_accumulated


def run_lindistflow(Gbus, Bbus, net, P_mw, Q_mw):
    """Runs the LinDistFlow calculation with correct power accumulation, including transformers."""

    import numpy as np
    import pandas as pd

    # Convert power injections to per unit
    S_pu = (P_mw + 1j * Q_mw) / net.sn_mva  
    Ybus = compute_Ybus(Gbus, Bbus)

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

    # Compute incidence matrix
    A = compute_incidence_matrix(net)

    # Compute accumulated power flows
    P_accumulated, Q_accumulated = accumulate_downstream_power(A, P_mw, Q_mw, net)

    # === Compute P_branch and Q_branch for Lines and Transformers ===
    num_lines = len(net.line)
    num_trafo = len(net.trafo)

    P_branch = np.zeros(num_lines)
    Q_branch = np.zeros(num_lines)

    for line_idx in range(num_lines):
        from_bus = net.line.from_bus[line_idx]
        to_bus = net.line.to_bus[line_idx]
        P_branch[line_idx] = P_accumulated[to_bus]
        Q_branch[line_idx] = Q_accumulated[to_bus]

    # Transformer Power (Only for output, not current calculation)
    P_trafo = np.zeros(num_trafo)
    Q_trafo = np.zeros(num_trafo)
    for trafo_idx in range(num_trafo):
        lv_bus = net.trafo.lv_bus[trafo_idx]
        P_trafo[trafo_idx] = P_accumulated[lv_bus]  
        Q_trafo[trafo_idx] = Q_accumulated[lv_bus]

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
    #I_branch_ka = I_branch_pu / (V_base_kv * np.sqrt(3))   # Convert to kA


    # Create final results dictionary
    results = {
        'V_magnitude': pd.Series(np.abs(V_nodes), index=net.bus.index),  # p.u.
        'P_node': pd.Series(P_mw, index=net.bus.index),  # MW
        'Q_node': pd.Series(Q_mw, index=net.bus.index),  # MVar
        'P_line_flow': pd.Series(P_branch, index=net.line.index),  # MW
        'Q_line_flow': pd.Series(Q_branch, index=net.line.index),  # MVar
        'P_trafo_flow': pd.Series(P_trafo, index=net.trafo.index) if num_trafo > 0 else pd.Series([], dtype=np.float64),  # MW
        'Q_trafo_flow': pd.Series(Q_trafo, index=net.trafo.index) if num_trafo > 0 else pd.Series([], dtype=np.float64),  # MVar
        'I_branch': pd.Series(np.abs(I_branch_ka), index=net.line.index)  # kA (ONLY FOR LINES)
    }

    print("\n=== Final Power Flow Results ===")
    print("Transformer power flow:\n", results['P_trafo_flow'])
    print("Line power flow:\n", results['P_line_flow'])

    return results


def manual_lindistflow_timeseries(time_steps, net, const_load_household_P, const_load_household_Q, const_load_heatpump, Gbus, Bbus):
    results = {
        "time_step": [],
        "V_magnitude": [],
        "P_node": [],
        "Q_node": [],
        "line_pl_mw": [],
        "line_ql_mvar": [],
        "trafo_pl_mw": [],
        "trafo_ql_mvar": [],
        "I_branch": [],  # Store only line currents
        "load_p_mw": [],
        "sgen_p_mw": []
    }

    # Compute Z matrix once
    Z = calculate_z_matrix(net)

    for t in time_steps:
        # Update loads
        const_load_household_P.time_step(net, time=t)
        const_load_household_Q.time_step(net, time=t)
        const_load_heatpump.time_step(net, time=t)

        # Compute power injections
        P = np.zeros(len(net.bus), dtype=np.float64)
        Q = np.zeros(len(net.bus), dtype=np.float64)
        
        if not net.load.empty:
            for i, bus in enumerate(net.load.bus.values.astype(int)):
                P[bus] -= net.load.p_mw.iloc[i]
                Q[bus] -= net.load.q_mvar.iloc[i]  # Ensure consistent indexing for Q

        if not net.sgen.empty:
            for i, bus in enumerate(net.sgen.bus.values.astype(int)):
                P[bus] += net.sgen.p_mw.iloc[i]
                Q[bus] += net.sgen.q_mvar.iloc[i]  # Ensure consistent indexing for Q

        # Run LinDistFlow calculation with the corrected Z matrix
        flow_results = run_lindistflow(Gbus, Bbus, net, P, Q)

        # Extract relevant results
        I_branch = flow_results["I_branch"]  # Get branch currents

        num_lines = len(net.line)  # Number of lines
        results["I_branch"].append(I_branch[:num_lines])  # Store only line currents

        results["time_step"].append(t)
        results["V_magnitude"].append(flow_results["V_magnitude"])
        results["P_node"].append(flow_results["P_node"])
        results["Q_node"].append(flow_results["Q_node"])
        results["line_pl_mw"].append(flow_results["P_line_flow"])  
        results["line_ql_mvar"].append(flow_results["Q_line_flow"])  
        results["trafo_pl_mw"].append(flow_results["P_trafo_flow"])  
        results["trafo_ql_mvar"].append(flow_results["Q_trafo_flow"])  
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

    # Store I_branch only for lines
    I_branch_df = pd.DataFrame(results["I_branch"], index=results["time_step"], columns=line_indices)

    # Create final results DataFrame
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
        "I_branch": I_branch_df  # Include I_branch in results
    }, axis=1)

    return results_df
