
"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
Optimization by Marwan Mostafa 

Grid Data File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### PACKAGES ####
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
from pandapower.control import ConstControl
from pandapower.timeseries import DFData
import simbench as sb

#### SCRIPTS ####
import parameters as par

###############################################################################
## FUNCTIONS ##
###############################################################################

def reorder_buses_and_update_references(net):
    # Create a copy of net.bus and reset the index
    reordered_bus = pd.concat([net.bus.loc[129:129], net.bus.drop(129)]).reset_index(drop=False)

    # Mapping of old bus indices to new ones
    old_to_new_indices = {row['index']: idx for idx, row in reordered_bus.iterrows()}

    # Update net.bus with the reordered DataFrame (drop the old index)
    reordered_bus.drop(columns="index", inplace=True)
    net.bus = reordered_bus

    # Update all dependent DataFrames with the new mapping
    def update_indices(df, col):
        if col in df:
            df[col] = df[col].map(old_to_new_indices)

    # Update bus references in various components
    update_indices(net.load, 'bus')
    update_indices(net.line, 'from_bus')
    update_indices(net.line, 'to_bus')
    update_indices(net.trafo, 'hv_bus')
    update_indices(net.trafo, 'lv_bus')
    update_indices(net.ext_grid, 'bus')
    update_indices(net.shunt, 'bus') if 'shunt' in net else None
    update_indices(net.ward, 'bus') if 'ward' in net else None
    update_indices(net.xward, 'bus') if 'xward' in net else None

    return net

def reorder_lines(net):
    for line in net.line.itertuples():
        # If from_bus > to_bus, swap them
        if line.from_bus > line.to_bus:
            # Swap from_bus and to_bus
            net.line.loc[line.Index, ['from_bus', 'to_bus']] = line.to_bus, line.from_bus

    return net

def setup_grid_powertech25(season):
    sb_code1 = "1-LV-semiurb4--0-no_sw"  # rural MV grid of scenario 0 with full switches
    net = sb.get_simbench_net(sb_code1)
    net = reorder_buses_and_update_references(net)
    #net = reorder_lines(net)
    line_indices = [24, 28, 23, 0, 4, 19, 11, 5, 22, 18, 6, 20, 31, 13, 17, 29, 7, 12, 16]  # List of line indices to reverse

    for idx in line_indices:
        net.line.loc[idx, ['from_bus', 'to_bus']] = net.line.loc[idx, ['to_bus', 'from_bus']].values

    # Set ext_grid vm_pu to 1.0
    net.ext_grid['vm_pu'] = 1.0

    # Remove Sgen
    net.sgen.drop(net.sgen.index, inplace=True)


    return net
