
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
## AUXILIARY FUNCTIONS ##
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

def append_zeros_to_buses(net):
    """
    Append '00' to all bus indices in a pandapower network and update all references.

    Parameters:
    - net: pandapower network

    Returns:
    - Updated pandapower network
    """
    # Create a mapping of old indices to new indices
    bus_mapping = {bus: int(f"{bus}00") for bus in net.bus.index}

    # Update references in all elements of the network
    for element in ["line", "trafo", "trafo3w", "load", "gen", "sgen", "shunt", "ext_grid", "ward", "xward", "dcline", "switch"]:
        if element in net and not net[element].empty:
            for col in net[element].columns:
                if "bus" in col:  # Look for columns containing "bus"
                    net[element][col] = net[element][col].replace(bus_mapping)

    # Update references in line "to_bus" and "from_bus"
    if "line" in net and not net.line.empty:
        net.line["from_bus"] = net.line["from_bus"].replace(bus_mapping)
        net.line["to_bus"] = net.line["to_bus"].replace(bus_mapping)

    # Update references in trafo "hv_bus" and "lv_bus"
    if "trafo" in net and not net.trafo.empty:
        net.trafo["hv_bus"] = net.trafo["hv_bus"].replace(bus_mapping)
        net.trafo["lv_bus"] = net.trafo["lv_bus"].replace(bus_mapping)

    # Update references in trafo3w "hv_bus", "mv_bus", and "lv_bus"
    if "trafo3w" in net and not net.trafo3w.empty:
        net.trafo3w["hv_bus"] = net.trafo3w["hv_bus"].replace(bus_mapping)
        net.trafo3w["mv_bus"] = net.trafo3w["mv_bus"].replace(bus_mapping)
        net.trafo3w["lv_bus"] = net.trafo3w["lv_bus"].replace(bus_mapping)

    # Rename the bus indices
    net.bus.rename(index=bus_mapping, inplace=True)

    return net

def reindex_bus(net, old_index, new_index):
    # Update references in all elements of the network
    for element in ["line", "trafo", "trafo3w", "load", "gen", "sgen", "shunt", "ext_grid", "ward", "xward", "dcline", "switch"]:
        if element in net and not net[element].empty:
            for col in net[element].columns:
                if "bus" in col:  # Look for columns containing "bus"
                    net[element][col] = net[element][col].replace({old_index: new_index})

    # Update references in line "to_bus" and "from_bus"
    if "line" in net and not net.line.empty:
        net.line["from_bus"] = net.line["from_bus"].replace({old_index: new_index})
        net.line["to_bus"] = net.line["to_bus"].replace({old_index: new_index})

    # Update references in trafo "hv_bus" and "lv_bus"
    if "trafo" in net and not net.trafo.empty:
        net.trafo["hv_bus"] = net.trafo["hv_bus"].replace({old_index: new_index})
        net.trafo["lv_bus"] = net.trafo["lv_bus"].replace({old_index: new_index})

    # Update references in trafo3w "hv_bus", "mv_bus", and "lv_bus"
    if "trafo3w" in net and not net.trafo3w.empty:
        net.trafo3w["hv_bus"] = net.trafo3w["hv_bus"].replace({old_index: new_index})
        net.trafo3w["mv_bus"] = net.trafo3w["mv_bus"].replace({old_index: new_index})
        net.trafo3w["lv_bus"] = net.trafo3w["lv_bus"].replace({old_index: new_index})

    # Rename the bus index
    net.bus.rename(index={old_index: new_index}, inplace=True)


###############################################################################
## MAIN FUNCTION ##
###############################################################################
def setup_grid_powertech255(season):
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

    net = append_zeros_to_buses(net)
    reindex_bus(net, old_index=1500, new_index=1)
    reindex_bus(net, old_index=2100, new_index=2)
    reindex_bus(net, old_index=800, new_index=3)
    reindex_bus(net, old_index=3200, new_index=4)
    reindex_bus(net, old_index=200, new_index=5)
    reindex_bus(net, old_index=300, new_index=6)
    reindex_bus(net, old_index=1200, new_index=7)
    reindex_bus(net, old_index=100, new_index=8)
    reindex_bus(net, old_index=2300, new_index=9)
    reindex_bus(net, old_index=1600, new_index=10)
    reindex_bus(net, old_index=1800, new_index=11)
    reindex_bus(net, old_index=1700, new_index=12)
    reindex_bus(net, old_index=500, new_index=13)
    reindex_bus(net, old_index=2200, new_index=14)
    reindex_bus(net, old_index=2800, new_index=15)
    reindex_bus(net, old_index=700, new_index=16)
    reindex_bus(net, old_index=1000, new_index=17)
    reindex_bus(net, old_index=600, new_index=18)
    reindex_bus(net, old_index=2400, new_index=19)
    reindex_bus(net, old_index=3000, new_index=20)
    reindex_bus(net, old_index=1300, new_index=21)
    reindex_bus(net, old_index=400, new_index=22)
    reindex_bus(net, old_index=3100, new_index=23)
    reindex_bus(net, old_index=2700, new_index=24)
    reindex_bus(net, old_index=3300, new_index=25)
    reindex_bus(net, old_index=2500, new_index=26)
    reindex_bus(net, old_index=2000, new_index=27)
    reindex_bus(net, old_index=2600, new_index=28)
    reindex_bus(net, old_index=2900, new_index=29)
    reindex_bus(net, old_index=3500, new_index=30)
    reindex_bus(net, old_index=3600, new_index=31)
    reindex_bus(net, old_index=3700, new_index=32)
    reindex_bus(net, old_index=3800, new_index=33)
    reindex_bus(net, old_index=4100, new_index=34)
    reindex_bus(net, old_index=4200, new_index=35)
    reindex_bus(net, old_index=4300, new_index=36)
    reindex_bus(net, old_index=3900, new_index=37)
    reindex_bus(net, old_index=4000, new_index=38)
    reindex_bus(net, old_index=1100, new_index=39)

    ############################################################################################################
    # Add Household Loads
    ############################################################################################################
    # Load the normalized household profile
    df_household_prognosis = pd.read_csv("householdPrognosis.csv", sep=';')
    df_season_household_prognosis = df_household_prognosis[df_household_prognosis['season'] == season]
    df_season_household_prognosis['meanP'] = df_season_household_prognosis['meanP'].str.replace(",", ".").astype(float)
    df_season_household_prognosis['P_HOUSEHOLD_NORM'] = df_season_household_prognosis['meanP'] / df_season_household_prognosis['meanP'].max()
    time_steps = df_season_household_prognosis.index


    household_loads = net.load[net.load['name'].str.startswith("LV4.101")]
    household_scaling_factors = household_loads['p_mw'].values
    for load_idx in household_loads.index:
        net.load.at[load_idx, 'controllable'] = False

    # Create a scaled profile DataFrame
    scaled_household_profiles = pd.DataFrame(
        df_season_household_prognosis['P_HOUSEHOLD_NORM'].values[:, None] * household_scaling_factors / par.hh_scaling,
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
    # Locate the load at bus 29
    hp_index = net.load[net.load.bus == 29].index

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
        df_season_heatpump_prognosis['p_mw'].values[:, None] * heatpump_scaling_factors_df['p_mw'].values * par.hp_scaling,
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

    return net, const_load_household, const_load_heatpump, time_steps, df_household_prognosis, df_season_heatpump_prognosis, heatpump_scaling_factors_df

###############################################################################
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

    # Filter loads connected to the specified buses
    loads_to_modify = net.load[net.load.bus.isin(buses_to_drop)].index

    # Set the 'p_mw' and 'q_mvar' columns to zero
    net.load.loc[loads_to_modify, ['p_mw', 'q_mvar']] = 0

    # Rename the loads to "DEACTIVATE"
    net.load.loc[loads_to_modify, 'name'] = "DEACTIVATE"

    # Update line at index 9 to represent 2 parallel NAYY 4x300 cables
    line_idx = 9
    net.line.loc[line_idx, ['name', 'type', 'length_km', 'r_ohm_per_km', 'x_ohm_per_km', 'c_nf_per_km', 'max_i_ka']] = [
        "LV4.101 Line 7 (Parallel 2x NAYY 4x300)",  # New name
        "cs",  # Type remains the same
        0.001552,  # Length (unchanged)
        0.1 / 2,  # resistance for parallel cables
        0.080425 / 2,  # Halve reactance for parallel cables
        829.999394 * 2,  # Double capacitance for parallel cables
        0.838  # Max current capacity in kA (2x NAYY 4x300 ~ 450 A per cable)
    ]

    # Set ext_grid vm_pu to 1.0
    net.ext_grid['vm_pu'] = 1.0

    # Remove Sgen
    net.sgen.drop(net.sgen.index, inplace=True)

    ############################################################################################################
    # Add Household Loads
    ############################################################################################################
    # Load the normalized household profile
    df_household_prognosis = pd.read_csv("householdPrognosis1h.csv", sep=';')
    df_season_household_prognosis = df_household_prognosis[df_household_prognosis['season'] == season]
    #df_season_household_prognosis['meanP'] = df_season_household_prognosis['meanP'].str.replace(",", ".").astype(float)
    df_season_household_prognosis['meanP'] = df_season_household_prognosis['meanP'].astype(float)
    df_season_household_prognosis['P_HOUSEHOLD_NORM'] = df_season_household_prognosis['meanP'] / df_season_household_prognosis['meanP'].max()
    time_steps = df_season_household_prognosis.index


    # Exclude the heat pump load index from the household loads
    household_loads = net.load[(net.load['name'].str.startswith("LV4.101")) & (net.load.index != 21)]
    household_scaling_factors = household_loads['p_mw'].values
    for load_idx in household_loads.index:
        net.load.at[load_idx, 'controllable'] = False

    # Create a scaled profile DataFrame
    scaled_household_profiles = pd.DataFrame(
        df_season_household_prognosis['P_HOUSEHOLD_NORM'].values[:, None] * household_scaling_factors / par.hh_scaling,
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
    # Set p_mw of load with index 21 to 0
    net.load.loc[22, 'p_mw'] = 0
    ############################################################################################################
    # Add Heat Pump
    ############################################################################################################
    # Locate the load at bus 29
    # Locate the specific household load at bus 29
    #hp_index = net.load[(net.load.bus == 29) & net.load['name'].str.startswith("LV4")].index

    # Identify the loads at bus 29
    loads_at_bus_29 = net.load[net.load.bus == 29]

    # Select the load to modify (e.g., the one starting with "LV4")
    target_load_index = loads_at_bus_29[loads_at_bus_29['name'].str.startswith("LV4")].index

    if len(target_load_index) > 0:
        # Modify only the first matched load (or modify logic as needed)
        target_load_index = target_load_index[0]
        
        # Rename the load and make it controllable
        net.load.at[target_load_index, 'name'] = net.load.at[target_load_index, 'name'].replace("LV4", "HP")
        net.load.at[target_load_index, 'controllable'] = True
    else:
        print("No load at bus 29 matches the criteria to be renamed.")

    heatpump_loads = net.load[net.load['name'].str.startswith("HP.101")]

    # Load the heatpump prognosis profile CSV and filter by season
    df_heatpump_prognosis = pd.read_csv("heatpumpPrognosis1h.csv", sep=';')
    df_season_heatpump_prognosis = df_heatpump_prognosis[df_heatpump_prognosis['season'] == season]
        
    # Process load profile for bus 1
    # df_season_heatpump_prognosis['meanP'] = df_season_heatpump_prognosis['meanP'].str.replace(",", ".").astype(float)
    # df_season_heatpump_prognosis['stdP'] = df_season_heatpump_prognosis['stdP'].str.replace(",", ".").astype(float)
    # df_season_heatpump_prognosis['meanQ'] = df_season_heatpump_prognosis['meanQ'].str.replace(",", ".").astype(float)
    # df_season_heatpump_prognosis['stdQ'] = df_season_heatpump_prognosis['stdQ'].str.replace(",", ".").astype(float)
    df_season_heatpump_prognosis['meanP'] = df_season_heatpump_prognosis['meanP'].astype(float)
    df_season_heatpump_prognosis['stdP'] = df_season_heatpump_prognosis['stdP'].astype(float)
    df_season_heatpump_prognosis['meanQ'] = df_season_heatpump_prognosis['meanQ'].astype(float)
    df_season_heatpump_prognosis['stdQ'] = df_season_heatpump_prognosis['stdQ'].astype(float)
        
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
        df_season_heatpump_prognosis['p_mw'].values[:, None] * heatpump_scaling_factors_df['p_mw'].values * par.hp_scaling,
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

    ############################################################################################################
    # Ambient Temperature
    ############################################################################################################
    T_amb =  pd.read_csv("temperatureWinter1h.csv")
    T_amb = T_amb['APPARENT_TEMPERATURE:TOTAL']+273.15

    return net, const_load_household, const_load_heatpump, time_steps, df_household_prognosis, df_season_heatpump_prognosis, heatpump_scaling_factors_df, T_amb

def setup_grid_powertech25_variance(net,df_season_heatpump_prognosis,heatpump_scaling_factors_df):
    heatpump_loads = net.load[net.load['name'].str.startswith("HP.101")]
    df_season_heatpump_prognosis['p_mw'] = df_season_heatpump_prognosis['stdP_NORM']
    household_loads = net.load[(net.load['name'].str.startswith("LV4.101"))]
    for load_idx in household_loads.index:
        net.load.at[load_idx, 'p_mw'] = 0
    #print("grid loads",net.load)
    #print("network loads",net.load)
    # Create a scaled variance DataFrame
    df_variance = pd.DataFrame(
        (df_season_heatpump_prognosis['p_mw'].values[:, None] * heatpump_scaling_factors_df['p_mw'].values * par.hp_scaling)**2,
        columns=heatpump_loads.index
    )
    #print("df_variance",df_variance.columns.tolist())
    # Convert to DFData for dynamic control
    ds_variance= DFData(df_variance)
    #print("ds_variance",ds_variance)
    net.controller = net.controller[~net.controller['object'].apply(lambda ctrl: isinstance(ctrl, ConstControl))]
    const_variance = ConstControl(
        net,
        element="load",
        variable="p_mw",  # Update p_mw directly
        element_index=heatpump_loads.index,  # Apply to all loads
        profile_name=df_variance.columns.tolist(),  # Profile for each load
        data_source=ds_variance
    )
    variance_net = net
    return variance_net, const_variance


###############################################################################
def setup_grid_IAS(season):
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
    net.controller.drop(net.controller[net.controller["object"].apply(lambda x: isinstance(x, ConstControl.ConstControl))].index, inplace=True)


    ############################################################################################################
    # Add Household Loads
    ############################################################################################################
    # Load the normalized household profile
    df_household_prognosis = pd.read_csv("householdPrognosis1h.csv", sep=';')
    df_season_household_prognosis = df_household_prognosis[df_household_prognosis['season'] == season].reset_index(drop=True)
    #df_season_household_prognosis['meanP'] = df_season_household_prognosis['meanP'].str.replace(",", ".").astype(float)
    df_season_household_prognosis[['meanP', 'meanQ']] = df_season_household_prognosis[['meanP', 'meanQ']].astype(float)
    df_season_household_prognosis['P_HOUSEHOLD_NORM'] = df_season_household_prognosis['meanP'] / df_household_prognosis['meanP'].max()
    df_season_household_prognosis['Q_HOUSEHOLD_NORM'] = df_season_household_prognosis['meanQ'] / df_household_prognosis['meanP'].max()
    time_steps = df_season_household_prognosis.index


    # Exclude the heat pump load index from the household loads
    household_loads = net.load[(net.load['name'].str.startswith("LV4.101")) & (net.load.index != 21)]
    household_scaling_factors_P = household_loads['p_mw'].values  # Extract only active power scaling
    household_scaling_factors_Q = household_loads['p_mw'].values / 5  # Extract only reactive power scaling

    for load_idx in household_loads.index:
        net.load.at[load_idx, 'controllable'] = False

    # Create Separate Scaled Profiles for P and Q
    scaled_household_profiles_P = pd.DataFrame(
        df_season_household_prognosis['P_HOUSEHOLD_NORM'].values[:, None] * household_scaling_factors_P / par.hh_scaling,
        columns=household_loads.index
    )

    scaled_household_profiles_Q = pd.DataFrame(
        df_season_household_prognosis['Q_HOUSEHOLD_NORM'].values[:, None] * household_scaling_factors_P * -1 / par.hh_scaling,
        columns=household_loads.index
    )

        
    # Convert to DFData for dynamic control
    ds_scaled_household_profiles_P = DFData(scaled_household_profiles_P)
    ds_scaled_household_profiles_Q = DFData(scaled_household_profiles_Q)

    # Add Separate ConstControls for P and Q
    const_load_household_P = ConstControl(
        net,
        element="load",
        variable="p_mw",
        element_index=household_loads.index,
        profile_name=scaled_household_profiles_P.columns.tolist(),
        data_source=ds_scaled_household_profiles_P
    )

    const_load_household_Q = ConstControl(
        net,
        element="load",
        variable="q_mvar",
        element_index=household_loads.index,
        profile_name=scaled_household_profiles_Q.columns.tolist(),
        data_source=ds_scaled_household_profiles_Q
    )
    # Set p_mw of load with index 21 to 0
    #net.load.loc[22, 'p_mw'] = 0
    #net.load.loc[22, 'q_mvar'] = 0
    ############################################################################################################
    # Add PV Generation
    ############################################################################################################
    # Load PV Prognosis Data
    df_pv_prognosis = pd.read_csv("pvPrognosis1h.csv", sep=';')
    df_pv_prognosis['meanP'] = df_pv_prognosis['meanP'].clip(lower=0)


    
    # Define PV Capacity Categories
    pv_capacity_mapping = {
        (0.00001, 0.003): 10,  # 0 kWp
        (0.003, 0.006): 15,  # 4 kWp
        (0.006, 0.012): 20,  # 6 kWp
        (0.012, float('inf')): 25  # 10 kWp
    }

    # Function to determine PV capacity
    def get_pv_capacity(p_mw):
        for (low, high), capacity in pv_capacity_mapping.items():
            if low <= p_mw < high:
                return capacity / 1000  # Convert kWp to MW
        return 0
    
    # Add PV Generation at Household Load Nodes
    pv_nodes = []
    pv_indices = []
    for load_idx, load_row in household_loads.iterrows():
        bus = load_row['bus']
        p_mw = load_row['p_mw']
        pv_capacity = get_pv_capacity(p_mw)
        
        if pv_capacity > 0:
            pv_idx = net.sgen.shape[0]  # Get new sgen index
            net.sgen = net.sgen.append(pd.Series({
                "bus": bus,
                "p_mw": pv_capacity,
                "q_mvar": 0,
                "name": f"PV_{load_idx}",
                "in_service": True,
                "sn_mva": 1.0,  # Default value, adjust if needed
                "scaling": 1.0  # Default scaling factor
            }, name=net.sgen.shape[0]))            
            pv_nodes.append(bus)
            pv_indices.append(pv_idx)
    # Explicitly set the sgen at bus 29 to 0
    #net.sgen.loc[net.sgen["bus"] == 29, "p_mw"] = 0

    # Scale PV Profiles
    df_season_pv_prognosis = df_pv_prognosis[df_pv_prognosis['season'] == season].reset_index(drop=True)
    df_season_pv_prognosis['P_PV_NORM'] = df_season_pv_prognosis['meanP'] / df_pv_prognosis['meanP'].max()
    df_season_pv_prognosis['stdP_NORM'] = df_season_pv_prognosis['stdP'] / df_pv_prognosis['meanP'].max()
    scaled_pv_profiles = pd.DataFrame(
        df_season_pv_prognosis['P_PV_NORM'].values[:, None] * np.array([net.sgen.loc[i, 'p_mw'] for i in pv_indices]) / par.hh_scaling,
        columns=pv_indices
    )
    # Convert to DFData
    ds_scaled_pv_profiles = DFData(scaled_pv_profiles)

    # Create PV Controller
    const_pv = ConstControl(
        net,
        element="sgen",
        variable="p_mw",
        element_index=pv_indices,
        profile_name=scaled_pv_profiles.columns.tolist(),
        data_source=ds_scaled_pv_profiles
    )


    ############################################################################################################
    # Add Heat Pump
    ############################################################################################################

    # Identify the loads at the specified buses
    aggregation_map = {
        37: [35, 36, 38, 39],
        42: [40, 41, 42, 43],
        34: [29, 14, 9, 19],
        33: [31, 27, 25, 20, 26],
        24: [10, 6, 30, 13, 4],
        32: [21, 8, 2, 3, 12, 1, 11],
        16: [23, 18, 17, 5, 22, 28, 7]
    }

    # Target parent buses
    target_buses = list(aggregation_map.keys())

    # Get loads at target buses whose names start with "LV4"
    loads_at_target_buses = net.load[net.load.bus.isin(target_buses)]
    target_load_indices = loads_at_target_buses[loads_at_target_buses['name'].str.startswith("LV4")].index

    if len(target_load_indices) > 0:
        for idx in target_load_indices:
            bus = net.load.at[idx, 'bus']

            # Get child buses for this parent bus
            child_buses = aggregation_map.get(bus, [])

            # Sum up p_mw values of loads at those child buses
            total_child_load = net.load[net.load.bus.isin(child_buses)]['p_mw'].sum()

            # Rename the load and make it controllable
            net.load.at[idx, 'name'] = net.load.at[idx, 'name'].replace("LV4", "HP")
            net.load.at[idx, 'controllable'] = True
            net.load.at[idx, 'p_mw'] = total_child_load
    else:
        print("No loads at the specified buses match the criteria to be renamed.")

    heatpump_loads = net.load[net.load['name'].str.startswith("HP.101")]

    # Load the heatpump prognosis profile CSV and filter by season
    df_heatpump_prognosis = pd.read_csv("heatpumpPrognosis1h.csv", sep=';')
    df_season_heatpump_prognosis = df_heatpump_prognosis[df_heatpump_prognosis['season'] == season].reset_index(drop=True)
        
    # Process load profile for bus 1
    # df_season_heatpump_prognosis['meanP'] = df_season_heatpump_prognosis['meanP'].str.replace(",", ".").astype(float)
    # df_season_heatpump_prognosis['stdP'] = df_season_heatpump_prognosis['stdP'].str.replace(",", ".").astype(float)
    # df_season_heatpump_prognosis['meanQ'] = df_season_heatpump_prognosis['meanQ'].str.replace(",", ".").astype(float)
    # df_season_heatpump_prognosis['stdQ'] = df_season_heatpump_prognosis['stdQ'].str.replace(",", ".").astype(float)
    df_season_heatpump_prognosis['meanP'] = df_season_heatpump_prognosis['meanP'].astype(float)
    df_season_heatpump_prognosis['stdP'] = df_season_heatpump_prognosis['stdP'].astype(float)
    df_season_heatpump_prognosis['meanQ'] = df_season_heatpump_prognosis['meanQ'].astype(float)
    df_season_heatpump_prognosis['stdQ'] = df_season_heatpump_prognosis['stdQ'].astype(float)
        
    df_season_heatpump_prognosis['meanP_NORM'] = df_season_heatpump_prognosis['meanP'] / df_heatpump_prognosis['meanP'].max()
    df_season_heatpump_prognosis['stdP_NORM'] = df_season_heatpump_prognosis['stdP'] / df_heatpump_prognosis['meanP'].max()
    df_season_heatpump_prognosis['p_mw'] = df_season_heatpump_prognosis['meanP_NORM']
    df_season_heatpump_prognosis['meanQ_NORM'] = df_season_heatpump_prognosis['meanQ'] / df_heatpump_prognosis['meanQ'].max()
    df_season_heatpump_prognosis['stdQ_NORM'] = df_season_heatpump_prognosis['stdQ'] / df_heatpump_prognosis['meanQ'].max()
    df_season_heatpump_prognosis['q_mvar'] = df_season_heatpump_prognosis['meanQ_NORM']


    # Generate heatpump scaling factors DataFrame
    heatpump_scaling_factors_df = pd.DataFrame({
        'load_idx': heatpump_loads.index,
        'p_mw': heatpump_loads['p_mw'].values,
        'bus': heatpump_loads['bus'].values
    }).set_index('load_idx')

    # Create a scaled heatpump profile DataFrame
    df_season_heatpump_prognosis_scaled = pd.DataFrame(
        df_season_heatpump_prognosis['p_mw'].values[:, None] * heatpump_scaling_factors_df['p_mw'].values * par.hp_scaling,
        columns=heatpump_loads.index
    )
    Q_scaling = par.Q_scaling
    #print("Q_scaling",Q_scaling)

    df_season_heatpump_prognosis_scaled_Q = pd.DataFrame(
        df_season_heatpump_prognosis['p_mw'].values[:, None] * heatpump_scaling_factors_df['p_mw'].values * -1 * par.hp_scaling * Q_scaling,
        columns=heatpump_loads.index
)

    # Convert to DFData for dynamic control
    ds_scaled_heatpump_profiles = DFData(df_season_heatpump_prognosis_scaled)
    ds_scaled_heatpump_profiles_Q = DFData(df_season_heatpump_prognosis_scaled_Q)

    # Add a single ConstControl to update p_mw
    const_load_heatpump = ConstControl(
        net,
        element="load",
        variable="p_mw",  # Update p_mw directly
        element_index=heatpump_loads.index,  # Apply to all loads
        profile_name=df_season_heatpump_prognosis_scaled.columns.tolist(),  # Profile for each load
        data_source=ds_scaled_heatpump_profiles
    )

    const_load_heatpump_Q = ConstControl(
        net,
        element="load",
        variable="q_mvar",  # Update q_mvar
        element_index=heatpump_loads.index,  # Apply to all heat pumps
        profile_name=df_season_heatpump_prognosis_scaled_Q.columns.tolist(),  # Profile for each load
        data_source=ds_scaled_heatpump_profiles_Q
    )

    ############################################################################################################
    # Ambient Temperature
    ############################################################################################################
    temperature_file = f"temperature_{season}1h.csv"

    try:
        T_amb = pd.read_csv(temperature_file)['APPARENT_TEMPERATURE:TOTAL'] + 273.15
    except FileNotFoundError:
        print(f"Warning: Temperature file '{temperature_file}' not found.")
        T_amb = None  # Set to None if missing


    ############################################################################################################
    # Electricity Price #
    ############################################################################################################
    electricity_price_file = f"electricityPrice_{season}1h.csv"

    try:
        electricity_price = pd.read_csv(electricity_price_file)['price']
    except FileNotFoundError:
        print(f"Warning: Electricity price file '{electricity_price_file}' not found.")
        electricity_price = None  # Set to None if missing

    ### Test PV
    original_sgen_p_mw = net.sgen['p_mw'].copy()
    #print(f"original_sgen_p_mw: {original_sgen_p_mw}")  # Debug statement


    return net, time_steps, const_pv, const_load_household_P, const_load_household_Q, const_load_heatpump, const_load_heatpump_Q ,df_household_prognosis, df_season_heatpump_prognosis, df_season_pv_prognosis ,heatpump_scaling_factors_df, T_amb, electricity_price, original_sgen_p_mw



def setup_grid_IAS_variance(season):
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
    net.controller.drop(net.controller[net.controller["object"].apply(lambda x: isinstance(x, ConstControl.ConstControl))].index, inplace=True)

    ############################################################################################################
    # Add Household Loads
    ############################################################################################################
    # Load the normalized household profile
    df_household_prognosis = pd.read_csv("householdPrognosis1h.csv", sep=';')
    df_season_household_prognosis = df_household_prognosis[df_household_prognosis['season'] == season].reset_index(drop=True)
    df_season_household_prognosis[['meanP', 'meanQ']] = df_season_household_prognosis[['meanP', 'meanQ']].astype(float)
    df_season_household_prognosis['P_HOUSEHOLD_NORM'] = df_season_household_prognosis['meanP'] / df_household_prognosis['meanP'].max()
    df_season_household_prognosis['Q_HOUSEHOLD_NORM'] = df_season_household_prognosis['meanQ'] / df_household_prognosis['meanP'].max()
    time_steps = df_season_household_prognosis.index


    # Exclude the heat pump load index from the household loads
    household_loads = net.load[(net.load['name'].str.startswith("LV4.101")) & (net.load.index != 21)]
    household_scaling_factors_P = household_loads['p_mw'].values  # Extract only active power scaling

    # Set p_mw of load with index 21 to 0
    #net.load.loc[22, 'p_mw'] = 0
    #net.load.loc[22, 'q_mvar'] = 0
    ############################################################################################################
    # Add PV Generation
    ############################################################################################################
    # Load PV Prognosis Data
    df_pv_prognosis = pd.read_csv("pvPrognosis1h.csv", sep=';')
    df_pv_prognosis['meanP'] = df_pv_prognosis['meanP'].clip(lower=0)
    df_pv_prognosis['stdP'] = df_pv_prognosis['stdP'].clip(lower=0)


    
    # Define PV Capacity Categories
    pv_capacity_mapping = {
        (0.00001, 0.003): 10,  # 0 kWp
        (0.003, 0.006): 15,  # 4 kWp
        (0.006, 0.012): 20,  # 6 kWp
        (0.012, float('inf')): 25  # 10 kWp
    }

    # Function to determine PV capacity
    def get_pv_capacity(p_mw):
        for (low, high), capacity in pv_capacity_mapping.items():
            if low <= p_mw < high:
                return capacity / 1000  # Convert kWp to MW
        return 0
    
    # Add PV Generation at Household Load Nodes
    pv_nodes = []
    pv_indices = []
    for load_idx, load_row in household_loads.iterrows():
        bus = load_row['bus']
        p_mw = load_row['p_mw']
        pv_capacity = get_pv_capacity(p_mw)
        
        if pv_capacity > 0:
            pv_idx = net.sgen.shape[0]  # Get new sgen index
            net.sgen = net.sgen.append(pd.Series({
                "bus": bus,
                "p_mw": pv_capacity,
                "q_mvar": 0,
                "name": f"PV_{load_idx}",
                "in_service": True,
                "sn_mva": 1.0,  # Default value, adjust if needed
                "scaling": 1.0  # Default scaling factor
            }, name=net.sgen.shape[0]))            
            pv_nodes.append(bus)
            pv_indices.append(pv_idx)
    # Explicitly set the sgen at bus 29 to 0
    #net.sgen.loc[net.sgen["bus"] == 29, "p_mw"] = 0

    # Scale PV Profiles
    df_season_pv_prognosis = df_pv_prognosis[df_pv_prognosis['season'] == season].reset_index(drop=True)
    df_season_pv_prognosis['meanP_NORM'] = df_season_pv_prognosis['meanP'] / df_pv_prognosis['meanP'].max()
    df_season_pv_prognosis['stdP_NORM'] = df_season_pv_prognosis['stdP'] / df_pv_prognosis['meanP'].max()
    pv_std_scaling = df_pv_prognosis['stdP'].max()/df_pv_prognosis['meanP'].max()
    #print("pv_std_scaling",pv_std_scaling)
    scaled_pv_profiles = pd.DataFrame(
        ((df_season_pv_prognosis['stdP_NORM'].values[:, None]*pv_std_scaling)) * np.array([net.sgen.loc[i, 'p_mw'] for i in pv_indices]) / par.hh_scaling,
        columns=pv_indices
    )
    # Convert to DFData
    ds_scaled_pv_profiles = DFData(scaled_pv_profiles)

    # Create PV Controller
    const_pv = ConstControl(
        net,
        element="sgen",
        variable="p_mw",
        element_index=pv_indices,
        profile_name=scaled_pv_profiles.columns.tolist(),
        data_source=ds_scaled_pv_profiles
    )


    ############################################################################################################
    # Add Heat Pump
    ############################################################################################################
    # Locate the load at bus 29
    # Locate the specific household load at bus 29
    #hp_index = net.load[(net.load.bus == 29) & net.load['name'].str.startswith("LV4")].index

    # Identify the loads at the specified buses
    aggregation_map = {
        37: [35, 36, 38, 39],
        42: [40, 41, 42, 43],
        34: [29, 14, 9, 19],
        33: [31, 27, 25, 20, 26],
        24: [10, 6, 30, 13, 4],
        32: [21, 8, 2, 3, 12, 1, 11],
        16: [23, 18, 17, 5, 22, 28, 7]
    }

    # Target parent buses
    target_buses = list(aggregation_map.keys())

    # Get loads at target buses whose names start with "LV4"
    loads_at_target_buses = net.load[net.load.bus.isin(target_buses)]
    target_load_indices = loads_at_target_buses[loads_at_target_buses['name'].str.startswith("LV4")].index

    if len(target_load_indices) > 0:
        for idx in target_load_indices:
            bus = net.load.at[idx, 'bus']

            # Get child buses for this parent bus
            child_buses = aggregation_map.get(bus, [])

            # Sum up p_mw values of loads at those child buses
            total_child_load = net.load[net.load.bus.isin(child_buses)]['p_mw'].sum()

            # Rename the load and make it controllable
            net.load.at[idx, 'name'] = net.load.at[idx, 'name'].replace("LV4", "HP")
            net.load.at[idx, 'controllable'] = True
            net.load.at[idx, 'p_mw'] = total_child_load
    else:
        print("No loads at the specified buses match the criteria to be renamed.")

    heatpump_loads = net.load[net.load['name'].str.startswith("HP.101")]

    household_loads = net.load[(net.load['name'].str.startswith("LV4.101"))]

    # Load the heatpump prognosis profile CSV and filter by season
    df_heatpump_prognosis = pd.read_csv("heatpumpPrognosis1h.csv", sep=';')
    df_season_heatpump_prognosis = df_heatpump_prognosis[df_heatpump_prognosis['season'] == season].reset_index(drop=True)
        

    df_season_heatpump_prognosis['meanP'] = df_season_heatpump_prognosis['meanP'].astype(float)
    df_season_heatpump_prognosis['stdP'] = df_season_heatpump_prognosis['stdP'].astype(float)
    df_season_heatpump_prognosis['meanQ'] = df_season_heatpump_prognosis['meanQ'].astype(float)
    df_season_heatpump_prognosis['stdQ'] = df_season_heatpump_prognosis['stdQ'].astype(float)
        
    df_season_heatpump_prognosis['meanP_NORM'] = df_season_heatpump_prognosis['meanP'] / df_heatpump_prognosis['meanP'].max()
    df_season_heatpump_prognosis['stdP_NORM'] = df_season_heatpump_prognosis['stdP'] / df_heatpump_prognosis['meanP'].max()
    df_season_heatpump_prognosis['p_mw'] = df_season_heatpump_prognosis['meanP_NORM']
    df_season_heatpump_prognosis['meanQ_NORM'] = df_season_heatpump_prognosis['meanQ'] / df_heatpump_prognosis['meanQ'].max()
    df_season_heatpump_prognosis['stdQ_NORM'] = df_season_heatpump_prognosis['stdQ'] / df_heatpump_prognosis['meanQ'].max()
    df_season_heatpump_prognosis['q_mvar'] = df_season_heatpump_prognosis['meanQ_NORM']

    std_mean_scaling = df_heatpump_prognosis['stdP'].max()/df_heatpump_prognosis['meanP'].max()
    #print("std_mean_scaling",std_mean_scaling)


    # Generate heatpump scaling factors DataFrame
    heatpump_scaling_factors_df = pd.DataFrame({
        'load_idx': heatpump_loads.index,
        'p_mw': heatpump_loads['p_mw'].values,
        'bus': heatpump_loads['bus'].values
    }).set_index('load_idx')

    # Create a scaled heatpump profile DataFrame
    df_season_heatpump_prognosis_scaled = pd.DataFrame(
        ((df_season_heatpump_prognosis['stdP_NORM'].values[:, None]*std_mean_scaling)) * heatpump_scaling_factors_df['p_mw'].values * par.hp_scaling,
        columns=heatpump_loads.index
    )
    Q_scaling = par.Q_scaling
    #print("Q_scaling",Q_scaling)

    df_season_heatpump_prognosis_scaled_Q = pd.DataFrame(
        ((df_season_heatpump_prognosis['stdP_NORM'].values[:, None]*std_mean_scaling)) * heatpump_scaling_factors_df['p_mw'].values * -1 * par.hp_scaling * Q_scaling,
        columns=heatpump_loads.index
)

    # Convert to DFData for dynamic control
    ds_scaled_heatpump_profiles = DFData(df_season_heatpump_prognosis_scaled)
    ds_scaled_heatpump_profiles_Q = DFData(df_season_heatpump_prognosis_scaled_Q)

    # Add a single ConstControl to update p_mw
    const_load_heatpump = ConstControl(
        net,
        element="load",
        variable="p_mw",  # Update p_mw directly
        element_index=heatpump_loads.index,  # Apply to all loads
        profile_name=df_season_heatpump_prognosis_scaled.columns.tolist(),  # Profile for each load
        data_source=ds_scaled_heatpump_profiles
    )

    const_load_heatpump_Q = ConstControl(
        net,
        element="load",
        variable="q_mvar",  # Update q_mvar
        element_index=heatpump_loads.index,  # Apply to all heat pumps
        profile_name=df_season_heatpump_prognosis_scaled_Q.columns.tolist(),  # Profile for each load
        data_source=ds_scaled_heatpump_profiles_Q
    )

    for load_idx in household_loads.index:
        net.load.at[load_idx, 'p_mw'] = 0
        net.load.at[load_idx, 'q_mvar'] = 0



    return net
