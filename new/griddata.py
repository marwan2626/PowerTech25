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

def setup_grid():
    net = pn.create_cigre_network_lv()

    # Switch off industrial and commercial loads
    net.switch.loc[1, "closed"] = False
    net.switch.loc[2, "closed"] = False

    # Iterate over all loads in the network and set controllable to False (i.e. not flexible)
    for load_idx in net.load.index:
        net.load.at[load_idx, 'controllable'] = False

    # Remove the switch between bus 0 and bus 1
    switch_to_remove = net.switch[(net.switch.bus == 0) & (net.switch.element == 1)].index
    net.switch.drop(switch_to_remove, inplace=True)
    
    # Change the transformer HV bus from bus 1 to bus 0
    net.trafo.at[0, 'hv_bus'] = 0

    # Remove bus 1 and any associated elements
    bus_to_remove = 1
    net.bus.drop(bus_to_remove, inplace=True)
    net.load = net.load[~net.load.bus.isin([bus_to_remove])]
    net.sgen = net.sgen[~net.sgen.bus.isin([bus_to_remove])]
    net.line = net.line[~net.line.from_bus.isin([bus_to_remove]) & ~net.line.to_bus.isin([bus_to_remove])]
    net.trafo = net.trafo[~net.trafo.hv_bus.isin([bus_to_remove]) & ~net.trafo.lv_bus.isin([bus_to_remove])]
    net.switch = net.switch[~net.switch.bus.isin([bus_to_remove]) & ~net.switch.element.isin([bus_to_remove])]

    # Renumber the remaining buses to fill the gap
    old_to_new_bus_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(net.bus.index))}
    net.bus.rename(index=old_to_new_bus_map, inplace=True)

    # Update all elements that reference bus indices
    def update_bus_references(df, columns):
        for col in columns:
            df[col] = df[col].map(old_to_new_bus_map)

    update_bus_references(net.load, ['bus'])
    update_bus_references(net.sgen, ['bus'])
    update_bus_references(net.line, ['from_bus', 'to_bus'])
    update_bus_references(net.trafo, ['hv_bus', 'lv_bus'])
    update_bus_references(net.switch, ['bus', 'element'])

    # Define the peak power for each bus
    peak_power_dict = {12: 200, 16: 600, 17: 500, 18: 350, 19: 350}

    # Add PV generators to the corresponding buses
    pv_buses = [12, 16, 17, 18, 19]
    pv_generators = []

    # Load and preprocess the PV generation profile CSV file
    df_pv = pd.read_csv("pv_generation_profile.csv")
    df_pv['time'] = pd.to_datetime(df_pv['time'], format='%H:%M:%S').dt.time
    df_pv['time_step'] = range(len(df_pv))  # Create a numerical index
    df_pv.set_index('time_step', inplace=True)

    # Instead of scaling in ConstControl, we directly scale the profiles in the DataFrame
    for bus in pv_buses:
        df_pv[f'pvgen_bus_{bus}'] = df_pv['pvgen'] * peak_power_dict[bus] / 1000  # Scale to peak power

    # Create a single DFData object containing all the bus profiles
    ds_pv = DFData(df_pv[[f'pvgen_bus_{bus}' for bus in pv_buses]])

    # Load and preprocess the load profile CSV file
    df = pd.read_csv("load_profile_1111.csv")
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    df['time_step'] = range(len(df))  # Create a numerical index
    df.set_index('time_step', inplace=True)
    df['mult'] = df['mult'] * 15 / 1000
    ds = DFData(df)

    # Add PV generators and set the correct limits
    for bus in pv_buses:
        pv_gen = pp.create_sgen(net, old_to_new_bus_map[bus], p_mw=0, q_mvar=0, type='pv', controllable=True)
        # Set initial limits
        net.sgen.at[pv_gen, 'min_p_mw'] = 0
        net.sgen.at[pv_gen, 'max_p_mw'] = df_pv[f'pvgen_bus_{bus}'].max()  # Max value from scaled profile
        net.sgen.at[pv_gen, 'min_q_mvar'] = -0.5 / 1000  # Example value, adjust as needed
        net.sgen.at[pv_gen, 'max_q_mvar'] = 0.5 / 1000  # Example value, adjust as needed
        pv_generators.append(pv_gen)

    # Add the Load profile to the network
    profile_loads = net.load.index.intersection([0, 1, 2, 3, 4, 5])
    const_load = ConstControl(net, element='load', element_index=profile_loads,
                              variable='p_mw', data_source=ds, profile_name=["mult"] * len(profile_loads))

    # Initialize ConstControl for PV and load profiles with correct time steps, using scaled profiles directly
    const_pv = ConstControl(net, element='sgen', element_index=pv_generators,
                            variable='p_mw', data_source=ds_pv,
                            profile_name=[f'pvgen_bus_{bus}' for bus in pv_buses])

    # Remove buses with prefixes "I" or "C" along with associated elements
    buses_to_remove = net.bus[net.bus['name'].str.startswith(('Bus I', 'Bus C'))].index

    if not buses_to_remove.empty:
        # Drop the buses
        net.bus.drop(buses_to_remove, inplace=True)
        
        # Remove loads, sgens, lines, transformers connected to those buses
        net.load = net.load[~net.load.bus.isin(buses_to_remove)]
        net.sgen = net.sgen[~net.sgen.bus.isin(buses_to_remove)]
        net.line = net.line[~net.line.from_bus.isin(buses_to_remove) & ~net.line.to_bus.isin(buses_to_remove)]
        net.trafo = net.trafo[~net.trafo.hv_bus.isin(buses_to_remove) & ~net.trafo.lv_bus.isin(buses_to_remove)]
    
    # Remove switches associated with the deleted buses
    switches_to_remove = net.switch[(net.switch.bus.isin(buses_to_remove)) | (net.switch.element.isin(buses_to_remove))].index
    net.switch.drop(switches_to_remove, inplace=True)
    
    return net, df_pv, df, pv_generators, const_load, const_pv





def setup_grid_transactions(season):
    net = pn.create_cigre_network_lv()

    # Switch off industrial and commercial loads
    net.switch.loc[1, "closed"] = False
    net.switch.loc[2, "closed"] = False

    # Move the load with index 0 from bus 1 to bus 2
    if 0 in net.load.index:
        net.load.at[0, 'bus'] = 3

    # Iterate over all loads in the network and set controllable to False (i.e. not flexible)
    for load_idx in net.load.index:
        net.load.at[load_idx, 'controllable'] = False

    # Remove the switch between bus 0 and bus 1
    switch_to_remove = net.switch[(net.switch.bus == 0) & (net.switch.element == 1)].index
    net.switch.drop(switch_to_remove, inplace=True)

    # Change the transformer HV bus from bus 1 to bus 0
    net.trafo.at[0, 'hv_bus'] = 0

    # Remove bus 1 and associated elements
    bus_to_remove = 1
    net.bus.drop(bus_to_remove, inplace=True)
    net.load = net.load[~net.load.bus.isin([bus_to_remove])]
    net.sgen = net.sgen[~net.sgen.bus.isin([bus_to_remove])]
    net.line = net.line[~net.line.from_bus.isin([bus_to_remove]) & ~net.line.to_bus.isin([bus_to_remove])]
    net.trafo = net.trafo[~net.trafo.hv_bus.isin([bus_to_remove]) & ~net.trafo.lv_bus.isin([bus_to_remove])]
    net.switch = net.switch[~net.switch.bus.isin([bus_to_remove]) & ~net.switch.element.isin([bus_to_remove])]

    # Renumber the remaining buses to fill the gap
    old_to_new_bus_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(net.bus.index))}
    net.bus.rename(index=old_to_new_bus_map, inplace=True)

    def update_bus_references(df, columns):
        for col in columns:
            df[col] = df[col].map(old_to_new_bus_map)

    update_bus_references(net.load, ['bus'])
    update_bus_references(net.sgen, ['bus'])
    update_bus_references(net.line, ['from_bus', 'to_bus'])
    update_bus_references(net.trafo, ['hv_bus', 'lv_bus'])
    update_bus_references(net.switch, ['bus', 'element'])

    # Load the heatpump prognosis profile CSV and filter by season
    df_heatpump_prognosis = pd.read_csv("heatpumpPrognosis.csv", sep=';')
    df_season_heatpump_prognosis = df_heatpump_prognosis[df_heatpump_prognosis['season'] == season]
        
    # Process load profile for bus 1
    df_season_heatpump_prognosis['meanP'] = df_season_heatpump_prognosis['meanP'].str.replace(",", ".").astype(float)
    df_season_heatpump_prognosis['stdP'] = df_season_heatpump_prognosis['stdP'].str.replace(",", ".").astype(float)
    df_season_heatpump_prognosis['meanQ'] = df_season_heatpump_prognosis['meanQ'].str.replace(",", ".").astype(float)
    df_season_heatpump_prognosis['stdQ'] = df_season_heatpump_prognosis['stdQ'].str.replace(",", ".").astype(float)
    time_steps = df_season_heatpump_prognosis.index
    
    # Load the real load profile CSV
    df_heatpump = pd.read_csv("realData_winter.csv", sep=';')
        
    # Process load profile for bus 1
    df_heatpump['P_HEATPUMP'] = df_heatpump['P_HEATPUMP'].str.replace(",", ".").astype(float)
    time_steps = df_heatpump.index
    threshold = 0.5  # Z-score threshold for identifying outliers
    mean = df_heatpump['P_HEATPUMP'].mean()
    std = df_heatpump['P_HEATPUMP'].std()
    z_scores = (df_heatpump['P_HEATPUMP'] - mean) / std
    # Replace outliers with a rolling average (smoothing) or cap them
    df_heatpump['P_HEATPUMP_smooth'] = np.where(
        abs(z_scores) > threshold,
        df_heatpump['P_HEATPUMP'].rolling(window=5, min_periods=1, center=True).mean(),
        df_heatpump['P_HEATPUMP']
    )

    # Replace outliers with a rolling average (smoothing) or cap them
    df_heatpump['P_HEATPUMP_smooth'] = np.where(
        abs(z_scores) > threshold,
        df_heatpump['P_HEATPUMP'].rolling(window=4, min_periods=1, center=True).mean(),
        df_heatpump['P_HEATPUMP']
    )
    # Create a DFData object for the load profile on bus 1
    ds_load_heatpump = DFData(df_heatpump[['P_HEATPUMP_smooth']]/par.hp_scaling)  # Convert to MW

    # Set the load on bus 1 to follow this profile
    load_bus_1 = net.load.index.intersection([0])
    net.load.loc[load_bus_1, 'controllable'] = True
    const_load_heatpump = ConstControl(net, element='load', element_index=load_bus_1,
                                    variable='p_mw', data_source=ds_load_heatpump, profile_name="P_HEATPUMP_smooth")

    # Load the potato load profile CSV for buses 11, 15, 16, 17
    df_household = pd.read_csv("realData_winter.csv", sep=';')
    
    # Process the potato load profile (assuming it contains the same columns)
    df_household['P_HOUSEHOLD'] = df_household['P_HOUSEHOLD'].str.replace(",", ".").astype(float)

    # Create a DFData object for the load profile on buses 11, 15, 16, and 17
    ds_load_household = DFData(df_household[['P_HOUSEHOLD']]/par.house_scaling)  # Convert to MW

    # Set the load on buses 11, 15, 16, and 17 to follow this profile
    load_buses = net.load.index.intersection([1, 2, 3, 4, 5])
    net.load.index.intersection([0, 1, 2, 3, 4, 5])
    const_load_household = ConstControl(net, element='load', element_index=load_buses,
                                     variable='p_mw', data_source=ds_load_household, profile_name="P_HOUSEHOLD")
    
       # Remove buses with prefixes "I" or "C" along with associated elements
    buses_to_remove = net.bus[net.bus['name'].str.startswith(('Bus I', 'Bus C'))].index

    if not buses_to_remove.empty:
        # Drop the buses
        net.bus.drop(buses_to_remove, inplace=True)
        
        # Remove loads, sgens, lines, transformers connected to those buses
        net.load = net.load[~net.load.bus.isin(buses_to_remove)]
        net.sgen = net.sgen[~net.sgen.bus.isin(buses_to_remove)]
        net.line = net.line[~net.line.from_bus.isin(buses_to_remove) & ~net.line.to_bus.isin(buses_to_remove)]
        net.trafo = net.trafo[~net.trafo.hv_bus.isin(buses_to_remove) & ~net.trafo.lv_bus.isin(buses_to_remove)]
    
    # Remove switches associated with the deleted buses
    switches_to_remove = net.switch[(net.switch.bus.isin(buses_to_remove)) | (net.switch.element.isin(buses_to_remove))].index
    net.switch.drop(switches_to_remove, inplace=True)


    return net, const_load_heatpump, const_load_household, time_steps, df_season_heatpump_prognosis, df_heatpump, df_household


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

def setup_grid_irep(season):
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

    # Load the heatpump prognosis profile CSV and filter by season
    df_heatpump_prognosis = pd.read_csv("heatpumpPrognosis.csv", sep=';')
    df_season_heatpump_prognosis = df_heatpump_prognosis[df_heatpump_prognosis['season'] == season]
        
    # Process load profile for bus 1
    df_season_heatpump_prognosis['meanP'] = df_season_heatpump_prognosis['meanP'].str.replace(",", ".").astype(float)
    df_season_heatpump_prognosis['stdP'] = df_season_heatpump_prognosis['stdP'].str.replace(",", ".").astype(float)
    df_season_heatpump_prognosis['meanQ'] = df_season_heatpump_prognosis['meanQ'].str.replace(",", ".").astype(float)
    df_season_heatpump_prognosis['stdQ'] = df_season_heatpump_prognosis['stdQ'].str.replace(",", ".").astype(float)
    time_steps = df_season_heatpump_prognosis.index

    # Load the normalized household profile
    df_household = pd.read_csv("realData_winter.csv", sep=';')
    df_household['P_HOUSEHOLD'] = df_household['P_HOUSEHOLD'].str.replace(",", ".").astype(float)
    df_household['P_HOUSEHOLD_NORM'] = df_household['P_HOUSEHOLD'] / df_household['P_HOUSEHOLD'].max()

    household_loads = net.load[net.load['name'].str.startswith("LV4.101")]
    household_scaling_factors = household_loads['p_mw'].values
    
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

    for load in household_loads.itertuples():
    # Create a new load with modified parameters
        pp.create_load(
            net,
            bus=load.bus,  # Use the same bus as the relevant load
            p_mw=load.p_mw * par.hp_scaling,  # scale p_mw of the relevant load
            q_mvar=load.q_mvar,  # Same q_mvar
            name=load.name.replace("LV4.101", "HP.101"),  # Change name prefix
            scaling=load.scaling,  # Same scaling
            const_z_percent=load.const_z_percent,  # Same const_z_percent
            const_i_percent=load.const_i_percent,  # Same const_i_percent
            voltLvl=load.voltLvl,  # Same voltLvl
            sn_mva=load.sn_mva,  # Same sn_mva
            subnet=load.subnet  # Same subnet
        )
    
    heatpump_loads = net.load[net.load['name'].str.startswith("HP.101")]
    # Load the real load profile CSV
    df_heatpump = pd.read_csv("realData_winter.csv", sep=';')
        
    # Process load profile for bus 1
    df_heatpump['P_HEATPUMP'] = df_heatpump['P_HEATPUMP'].str.replace(",", ".").astype(float)
    time_steps = df_heatpump.index
    threshold = 0.5  # Z-score threshold for identifying outliers
    mean = df_heatpump['P_HEATPUMP'].mean()
    std = df_heatpump['P_HEATPUMP'].std()
    z_scores = (df_heatpump['P_HEATPUMP'] - mean) / std
    # Replace outliers with a rolling average (smoothing) or cap them
    df_heatpump['P_HEATPUMP_smooth'] = np.where(
        abs(z_scores) > threshold,
        df_heatpump['P_HEATPUMP'].rolling(window=5, min_periods=1, center=True).mean(),
        df_heatpump['P_HEATPUMP']
    )

    # Replace outliers with a rolling average (smoothing) or cap them
    df_heatpump['P_HEATPUMP_smooth'] = np.where(
        abs(z_scores) > threshold,
        df_heatpump['P_HEATPUMP'].rolling(window=4, min_periods=1, center=True).mean(),
        df_heatpump['P_HEATPUMP']
    )

    df_heatpump['P_HEATPUMP_NORM'] = df_heatpump['P_HEATPUMP_smooth'] / df_season_heatpump_prognosis['meanP'].max()
    df_season_heatpump_prognosis['meanP_NORM'] = df_season_heatpump_prognosis['meanP'] / df_season_heatpump_prognosis['meanP'].max()

    # Generate heatpump scaling factors DataFrame
    heatpump_scaling_factors_df = pd.DataFrame({
        'load_idx': heatpump_loads.index,
        'p_mw': heatpump_loads['p_mw'].values,
        'bus': heatpump_loads['bus'].values
    }).set_index('load_idx')
    

    # Create a scaled heatpump profile DataFrame
    scaled_heatpump_profiles = pd.DataFrame(
        df_heatpump['P_HEATPUMP_NORM'].values[:, None] * heatpump_scaling_factors_df['p_mw'].values,
        columns=heatpump_loads.index
    )


    # Convert to DFData for dynamic control
    ds_scaled_heatpump_profiles = DFData(scaled_heatpump_profiles)

    # Add a single ConstControl to update p_mw
    const_load_heatpump = ConstControl(
        net,
        element="load",
        variable="p_mw",  # Update p_mw directly
        element_index=heatpump_loads.index,  # Apply to all loads
        profile_name=scaled_heatpump_profiles.columns.tolist(),  # Profile for each load
        data_source=ds_scaled_heatpump_profiles
    )

    
    # Iterate over all loads in the network and set controllable to False (i.e. not flexible)
    for load_idx in heatpump_loads.index:
        net.load.at[load_idx, 'controllable'] = True

    for load_idx in household_loads.index:
        net.load.at[load_idx, 'controllable'] = False


    return net, const_load_household, const_load_heatpump, time_steps, df_season_heatpump_prognosis, df_household, df_heatpump, heatpump_scaling_factors_df
#return net, const_load_heatpump, const_load_household, time_steps, df_season_heatpump_prognosis, df_heatpump, df_households



def setup_grid_irep_forecast(season):
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

    # Load the heatpump prognosis profile CSV and filter by season
    df_heatpump_prognosis = pd.read_csv("heatpumpPrognosis.csv", sep=';')
    df_season_heatpump_prognosis = df_heatpump_prognosis[df_heatpump_prognosis['season'] == season]
        
    # Process load profile for bus 1
    df_season_heatpump_prognosis['meanP'] = df_season_heatpump_prognosis['meanP'].str.replace(",", ".").astype(float)
    df_season_heatpump_prognosis['stdP'] = df_season_heatpump_prognosis['stdP'].str.replace(",", ".").astype(float)
    df_season_heatpump_prognosis['meanQ'] = df_season_heatpump_prognosis['meanQ'].str.replace(",", ".").astype(float)
    df_season_heatpump_prognosis['stdQ'] = df_season_heatpump_prognosis['stdQ'].str.replace(",", ".").astype(float)
    time_steps = df_season_heatpump_prognosis.index

    # Load the normalized household profile
    df_household = pd.read_csv("realData_winter.csv", sep=';')
    df_household['P_HOUSEHOLD'] = df_household['P_HOUSEHOLD'].str.replace(",", ".").astype(float)
    df_household['P_HOUSEHOLD_NORM'] = df_household['P_HOUSEHOLD'] / df_household['P_HOUSEHOLD'].max()

    household_loads = net.load[net.load['name'].str.startswith("LV4.101")]
    household_scaling_factors = household_loads['p_mw'].values
    
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

    for load in household_loads.itertuples():
    # Create a new load with modified parameters
        pp.create_load(
            net,
            bus=load.bus,  # Use the same bus as the relevant load
            p_mw=load.p_mw * par.hp_scaling,  # scale p_mw of the relevant load
            q_mvar=load.q_mvar,  # Same q_mvar
            name=load.name.replace("LV4.101", "HP.101"),  # Change name prefix
            scaling=load.scaling,  # Same scaling
            const_z_percent=load.const_z_percent,  # Same const_z_percent
            const_i_percent=load.const_i_percent,  # Same const_i_percent
            voltLvl=load.voltLvl,  # Same voltLvl
            sn_mva=load.sn_mva,  # Same sn_mva
            subnet=load.subnet  # Same subnet
        )
    
    heatpump_loads = net.load[net.load['name'].str.startswith("HP.101")]
    # Load the real load profile CSV
        
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

    
    # Iterate over all loads in the network and set controllable to False (i.e. not flexible)
    for load_idx in heatpump_loads.index:
        net.load.at[load_idx, 'controllable'] = True

    for load_idx in household_loads.index:
        net.load.at[load_idx, 'controllable'] = False


    return net, const_load_household, const_load_heatpump, time_steps, df_season_heatpump_prognosis, df_household, heatpump_scaling_factors_df
#return net, const_load_heatpump, const_load_household, time_steps, df_season_heatpump_prognosis, df_heatpump, df_households