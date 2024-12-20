
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

    # Buses and lines to remove
    buses_to_drop = [14, 34, 9, 19]

    # Remove lines associated with the buses to drop
    lines_to_drop = net.line[
        (net.line.from_bus.isin(buses_to_drop)) | (net.line.to_bus.isin(buses_to_drop))
    ].index

    # Drop the identified lines
    pp.drop_lines(net, lines_to_drop)

    # Drop the identified buses
    pp.drop_buses(net, buses_to_drop)

    # Set ext_grid vm_pu to 1.0
    net.ext_grid['vm_pu'] = 1.0

    # Remove Sgen
    net.sgen.drop(net.sgen.index, inplace=True)

    ############################################################################################################
    # Add Household Loads
    ############################################################################################################
    # Load the normalized household profile
    df_household = pd.read_csv("realData_winter.csv", sep=';')
    df_household['P_HOUSEHOLD'] = df_household['P_HOUSEHOLD'].str.replace(",", ".").astype(float)
    df_household['P_HOUSEHOLD_NORM'] = df_household['P_HOUSEHOLD'] / df_household['P_HOUSEHOLD'].max()
    time_steps = df_household.index


    household_loads = net.load[net.load['name'].str.startswith("LV4.101")]
    household_scaling_factors = household_loads['p_mw'].values
    for load_idx in household_loads.index:
        net.load.at[load_idx, 'controllable'] = False

        # Create a scaled profile DataFrame
    scaled_household_profiles = pd.DataFrame(
        df_household['P_HOUSEHOLD_NORM'].values[:, None] * household_scaling_factors / par.hh_scaling,
        columns=household_loads.index
    )

    # Convert to DFData for dynamic control
    ds_scaled_household_profiles = DFData(scaled_household_profiles)

    # Add a single ConstControl to update p_mw
    const_load_household = ConstControl(
        net,
        element="load",
        variable="p_mw",  # Update p_mw directly
        element_index=household_loads.index,  # Apply to all loads
        profile_name=scaled_household_profiles.columns.tolist(),  # Profile for each load
        data_source=ds_scaled_household_profiles
    )

    ############################################################################################################
    # Add Heat Pump
    ############################################################################################################
    # Locate the load at bus 24
    hp_index = net.load[net.load.bus == 24].index

    # Update the load name to start with "HP" instead of "LV4"
    net.load.loc[hp_index, 'name'] = net.load.loc[hp_index, 'name'].str.replace(r"^LV4", "HP", regex=True)

    # Set 'controllable' to True
    net.load.loc[hp_index, 'controllable'] = True

    heatpump_loads = net.load[net.load['name'].str.startswith("HP.101")]

    # Load the heatpump prognosis profile CSV and filter by season
    df_heatpump_prognosis = pd.read_csv("heatpumpPrognosis.csv", sep=';')
    df_season_heatpump_prognosis = df_heatpump_prognosis[df_heatpump_prognosis['season'] == season]
        
    # Process load profile for bus 1
    df_season_heatpump_prognosis['meanP'] = df_season_heatpump_prognosis['meanP'].str.replace(",", ".").astype(float)
    df_season_heatpump_prognosis['stdP'] = df_season_heatpump_prognosis['stdP'].str.replace(",", ".").astype(float)
    df_season_heatpump_prognosis['meanQ'] = df_season_heatpump_prognosis['meanQ'].str.replace(",", ".").astype(float)
    df_season_heatpump_prognosis['stdQ'] = df_season_heatpump_prognosis['stdQ'].str.replace(",", ".").astype(float)
        
    df_season_heatpump_prognosis['meanP_NORM'] = df_season_heatpump_prognosis['meanP'] / df_season_heatpump_prognosis['meanP'].max()
    df_season_heatpump_prognosis['stdP_NORM'] = df_season_heatpump_prognosis['stdP'] / df_season_heatpump_prognosis['meanP'].max()
    df_season_heatpump_prognosis['p_mw'] = df_season_heatpump_prognosis['meanP_NORM']


    # Generate heatpump scaling factors DataFrame
    heatpump_scaling_factors_df = pd.DataFrame({
        'load_idx': heatpump_loads.index,
        'p_mw': heatpump_loads['p_mw'].values,
        'bus': heatpump_loads['bus'].values
    }).set_index('load_idx')

    # Create a scaled heatpump profile DataFrame
    df_season_heatpump_prognosis_scaled = pd.DataFrame(
        df_season_heatpump_prognosis['p_mw'].values[:, None] * heatpump_scaling_factors_df['p_mw'].values,
        columns=heatpump_loads.index
    )

    # Convert to DFData for dynamic control
    ds_scaled_heatpump_profiles = DFData(df_season_heatpump_prognosis_scaled)

    # Add a single ConstControl to update p_mw
    const_load_heatpump = ConstControl(
        net,
        element="load",
        variable="p_mw",  # Update p_mw directly
        element_index=heatpump_loads.index,  # Apply to all loads
        profile_name=df_season_heatpump_prognosis_scaled.columns.tolist(),  # Profile for each load
        data_source=ds_scaled_heatpump_profiles
    )

    return net, const_load_household, const_load_heatpump, time_steps, df_household, df_season_heatpump_prognosis
