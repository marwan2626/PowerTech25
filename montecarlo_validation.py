"""
Code for the publication Reliability-Constrained Thermal Storage Sizing in
Multi-Energy Systems under Uncertainty
Optimization by Marwan Mostafa 

Monecarlo Validation File
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

#### SCRIPTS ####
import parameters as par
import results as rs

###############################################################################
## Generate Samples ##
###############################################################################


import numpy as np
import pandas as pd

def generate_samples(df_season_heatpump_prognosis):
    # Normalize the input data for ease of use
    max_mean = df_season_heatpump_prognosis['meanP'].max()
    df_season_heatpump_prognosis['meanP_norm'] = df_season_heatpump_prognosis['meanP'] / max_mean
    df_season_heatpump_prognosis['stdP_norm'] = df_season_heatpump_prognosis['stdP'] / max_mean
    
    # Extract mean and std deviation as numpy arrays for efficient computation
    meanP = df_season_heatpump_prognosis['meanP_norm'].values
    stdP = df_season_heatpump_prognosis['stdP_norm'].values
    timesteps = df_season_heatpump_prognosis.index

    n_samples = par.N_MC

    # Initialize the list to store sample profiles
    sample_profiles = []

    # Randomly choose a distribution for each sample and generate data
    for i in range(n_samples):
        # Randomly choose the distribution for the current sample
        distribution_choice = np.random.choice(['normal', 'uniform', 'exponential', 'poisson', 'beta', 'gamma', 'lognormal', 'weibull'])

        if distribution_choice == 'normal':
            samples = np.random.normal(loc=meanP, scale=stdP, size=len(meanP))

        elif distribution_choice == 'uniform':
            # Define range for uniform distribution
            lowerP = meanP - stdP  # Lower bound based on the mean and std deviation
            upperP = meanP + stdP  # Upper bound based on the mean and std deviation
            samples = np.random.uniform(low=lowerP, high=upperP, size=len(meanP))

        elif distribution_choice == 'exponential':
            # Scale exponential distribution to match the mean and approximate variability
            samples = np.random.exponential(scale=stdP, size=len(meanP))

        elif distribution_choice == 'poisson':
            # Poisson's standard deviation is sqrt(lambda), so adjust lambda accordingly
            lambdaP = np.maximum(meanP, stdP)  # Ensure lambda is positive and reasonable
            samples = np.random.poisson(lam=lambdaP, size=len(meanP))

        elif distribution_choice == 'beta':
            a = (meanP * (1 - meanP) / stdP**2 - 1) * meanP
            b = a * (1 / meanP - 1)
            a = np.maximum(a, 1e-3)
            b = np.maximum(b, 1e-3)
            samples = np.random.beta(a=a, b=b, size=len(meanP))

        elif distribution_choice == 'gamma':
            shape = (meanP / stdP)**2
            scale = stdP**2 / meanP
            shape = np.maximum(shape, 1e-3)
            scale = np.maximum(scale, 1e-3)
            samples = np.random.gamma(shape=shape, scale=scale, size=len(meanP))

        elif distribution_choice == 'lognormal':
            sigma = np.sqrt(np.log(1 + (stdP / meanP)**2))
            mu = np.log(meanP) - 0.5 * sigma**2
            samples = np.random.lognormal(mean=mu, sigma=sigma, size=len(meanP))

        elif distribution_choice == 'weibull':
            shape = 1.5
            scale = meanP / np.exp(np.log(2) / shape)
            samples = np.random.weibull(a=shape, size=len(meanP)) * scale

        # Convert the sample to a DataFrame
        df_sample = pd.DataFrame({
            'P_HEATPUMP_NORM': samples
        }, index=timesteps)  # Use the same index as the input DataFrame
        
        # Append the generated sample profile to the list
        sample_profiles.append(df_sample)

    return sample_profiles




###############################################################################
## AUXILLIARY FUNCTIONS ##
###############################################################################

def aggregate_line_violations(overall_line_violations, total_mc_samples, time_steps, line_indices):
    records = []

    # If violations occurred, process them
    if overall_line_violations:
        for line_idx, times in overall_line_violations.items():
            for t, violation_count in times.items():
                # Calculate the probability of violation
                violation_probability = violation_count / total_mc_samples
                records.append({
                    'line': line_idx,
                    'time_step': t,
                    'violation_probability': violation_probability,
                    'violation_probability_percent': violation_probability * 100,
                })

        # If no violations occurred for some time steps, include them with probability 0.0
        for line_idx in overall_line_violations.keys():
            existing_time_steps = overall_line_violations[line_idx].keys()
            for t in time_steps:
                if t not in existing_time_steps:
                    records.append({
                        'line': line_idx,
                        'time_step': t,
                        'violation_probability': 0.0,
                        'violation_probability_percent': 0.0,
                    })

    # If no violations occurred at all, populate all time steps for all lines with 0
    else:
        for line_idx in line_indices:
            for t in time_steps:
                records.append({
                    'line': line_idx,
                    'time_step': t,
                    'violation_probability': 0.0,
                    'violation_probability_percent': 0.0,
                })

    return pd.DataFrame(records)

def aggregate_trafo_violations(trafo_violations, num_samples, time_steps):
    records = []

    # Process existing violations
    if trafo_violations:
        for time_step, count in trafo_violations.items():
            violation_probability = count / num_samples if num_samples > 0 else 0
            records.append({
                'time_step': time_step,
                'violation_count': count,
                'violation_probability': violation_probability,
                'violation_probability_percent': violation_probability * 100,
            })

    # Populate with 0 for missing time steps and transformers
    for t in time_steps:
        if t not in trafo_violations.keys():
            records.append({
                'time_step': t,
                'violation_count': 0,
                'violation_probability': 0.0,
                'violation_probability_percent': 0.0,
            })

    return pd.DataFrame(records)

###############################################################################
## MONTECARLO ANALYSIS ##
###############################################################################

def run_single_sample_with_violation(
    net, time_steps, sample_profile, opf_results, const_load_household, const_load_heatpump, heatpump_scaling_factors_df
):
    net = net.deepcopy()

    # Extract OPF results
    # Extract flexible loads (heat pump dispatch) from OPF results
    flexible_load_dispatch = {
        t: opf_results['load'][t]['flexible_loads'] for t in time_steps
    }
    ts_in = opf_results['thermal_storage']['ts_in']
    ts_out = opf_results['thermal_storage']['ts_out']

    # Initialize violation counters
    line_violations = {}  # Store line violations: {line_idx: {timestep: count}}
    trafo_violations = {}  # Store transformer violations: {timestep: count}
    total_violations = 0
    # Initialize mc_line_results to store loading results
    mc_line_results = {line_idx: [] for line_idx in net.line.index}

    # Results storage
    sample_results = {'loads': [], 'buses': [], 'lines': [], 'trafos': []}
    flexible_time_synchronized_loads = {t: {} for t in time_steps}

    # Add variables for each time step
    for t in time_steps:
        # Update const_pv and const_load for this time step
        const_load_heatpump.time_step(net, time=t)

        # Initialize dictionaries for time-synchronized loads
        flexible_time_synchronized_loads[t] = {}
        # Iterate over all loads
        for load in net.load.itertuples():
            bus = load.bus
            if load.controllable:
                # Flexible load
                flexible_time_synchronized_loads[t][bus] = (
                    flexible_time_synchronized_loads[t].get(bus, 0.0) + load.p_mw
                )

    for t in time_steps:
        # Update fixed household loads using the ConstControl
        const_load_household.time_step(net, time=t)

        for load_index, scaling_data in heatpump_scaling_factors_df.iterrows():
            scaling_factor = scaling_data['p_mw']
            #print(f"scaling_factor: {scaling_factor}")
            bus = scaling_data['bus']
            #print(f"bus: {bus}")

            try:
                # Map Monte Carlo sample to heat pump load
                sampled_heat_demand = sample_profile.loc[t].at['P_HEATPUMP_NORM'] * scaling_factor * par.hp_scaling 

                #print(f"Time step {t}, Bus {bus}: Sampled heat demand = {sampled_heat_demand}")  # Debug statement

                # Get the flexible load dispatch from OPF results

                nominal_heatpump = flexible_load_dispatch[t].get(bus, 0.0)
                #ts_out_value = ts_out[t][bus]
                #ts_in_value = ts_in[t][bus]
                nominal_heat_demand = flexible_time_synchronized_loads[t][bus] * (1 / par.tsnet_eff) 

                # Ensure adjusted_load is non-negative
                adjusted_load = max(
                    0.0,
                    nominal_heatpump + (sampled_heat_demand - nominal_heat_demand)
                )
                #adjusted_load = nominal_heatpump #debug dispatch solution with pandapower

                # print(
                #     f"Time step {t}, Bus {bus}: "
                #     f"Nominal heatpump = {nominal_heatpump}, "
                #     f"Nominal heat demand = {nominal_heat_demand}, "
                #     f"Sampled heat demand = {sampled_heat_demand}, "
                #     f"Adjusted load = {adjusted_load}"
                # )  # Debug statement

                # Update the load in the network
                net.load.at[load_index, 'p_mw'] = float(adjusted_load)
                # if t == 166:
                #     print(f"Assigned adjusted load {adjusted_load} to load index {load_index}, bus {bus}")
                #     print(f"Load with index {load_index} is now {net.load.at[load_index, 'p_mw']}")

            except Exception as e:
                print(f"Error updating load_index {load_index}, bus {bus} at time {t}: {e}")
                continue

        try:
            pp.rundcpp(net, check_connectivity=False, verbose=False)
            #print(f"[INFO] Pandapower run successful for time step {t}.")
            # if t == 166:
            #     print(net.load)
        except pp.optimality.PandapowerRunError:
            total_violations += 1
            print(f"[ERROR] Pandapower failed to converge for time step {t}.")
            continue

        # Save line loading results into mc_line_results
        for line_idx, loading in net.res_line['loading_percent'].items():
            mc_line_results[line_idx].append({'time_step': t, 'loading_percent': loading})

        # Check for line violations
        for line_idx, loading in net.res_line['loading_percent'].items():
            if loading > 100:
                total_violations += 1
                if line_idx not in line_violations:
                    line_violations[line_idx] = {}
                line_violations[line_idx][t] = line_violations[line_idx].get(t, 0) + 1

        # Check for transformer violations
        for trafo_idx, loading in net.res_trafo['loading_percent'].items():
            if loading > par.max_trafo_loading*100:
                total_violations += 1
                trafo_violations[t] = trafo_violations.get(t, 0) + 1

        # Collect results
        load_results = net.res_load[['p_mw']].copy()
        load_results['time_step'] = t

        bus_results = net.res_bus[['vm_pu', 'va_degree']].copy()
        bus_results['time_step'] = t

        line_results = net.res_line[['loading_percent', 'i_ka']].copy()
        line_results['time_step'] = t

        trafo_results = net.res_trafo[['loading_percent']].copy()
        trafo_results['time_step'] = t
        #print(f"trafo_results: {trafo_results}")

        sample_results['loads'].append(load_results)
        sample_results['buses'].append(bus_results)
        sample_results['lines'].append(line_results)
        sample_results['trafos'].append(trafo_results)

    return (
        pd.concat(sample_results['loads'], ignore_index=True),
        pd.concat(sample_results['buses'], ignore_index=True),
        pd.concat(sample_results['lines'], ignore_index=True),
        pd.concat(sample_results['trafos'], ignore_index=True),
        mc_line_results,
        total_violations,
        line_violations,
        trafo_violations,
    )



def montecarlo_analysis_with_violations(
    net,
    time_steps,
    opf_results,
    const_load_household,
    const_load_heatpump,
    heatpump_scaling_factors_df,
    mc_samples,
    n_jobs,
    log_file,
):

    # Initialize log storage
    overall_line_violations = {}
    overall_trafo_violations = {}
    combined_mc_line_results = []
    # Start timing
    start_time = time.time()

    # Use joblib with tqdm for parallel processing
    results_and_violations = Parallel(n_jobs=n_jobs)(
        delayed(run_single_sample_with_violation)(
            net,
            time_steps,
            sample_profile,
            opf_results,
            const_load_household,
            const_load_heatpump,
            heatpump_scaling_factors_df,
        )
        for sample_profile in tqdm(mc_samples, desc="Processing samples")
    )

    # Combine results from all samples
    for single_sample_result in results_and_violations:
        (
            _loads_df, _buses_df, _lines_df, _trafos_df,  # Other results (optional to use)
            sample_mc_line_results,
            sample_total_violations,
            sample_line_violations,
            sample_trafo_violations
        ) = single_sample_result

    # Add sample line results to combined_mc_line_results
    for line_idx, results in sample_mc_line_results.items():
        for result in results:
            combined_mc_line_results.append({
                'line': line_idx,
                'time_step': result['time_step'],
                'loading_percent': result['loading_percent']
            })
    
    # Convert combined_mc_line_results into a DataFrame
    mc_line_results_df = pd.DataFrame(combined_mc_line_results)

    # Process results
    all_results = [res[:-3] for res in results_and_violations]
    violation_counts = [res[-3] for res in results_and_violations]
    line_violations_list = [res[-2] for res in results_and_violations]
    trafo_violations_list = [res[-1] for res in results_and_violations]
    #print(f"trafo_violations_list: {trafo_violations_list}")

    # Aggregate line and transformer violations
    for line_violations in line_violations_list:
        for line_idx, times in line_violations.items():
            if line_idx not in overall_line_violations:
                overall_line_violations[line_idx] = {}
            for t, count in times.items():
                overall_line_violations[line_idx][t] = overall_line_violations[line_idx].get(t, 0) + count


    for trafo_violations in trafo_violations_list:
        for t, count in trafo_violations.items():
            overall_trafo_violations[t] = overall_trafo_violations.get(t, 0) + count

    #print(f"overall_trafo_violations: {overall_trafo_violations}")

    # Find maximum violations
    max_violations_line = max(
        ((line, t, count) for line, times in overall_line_violations.items() for t, count in times.items()),
        key=lambda x: x[2],
        default=(None, None, 0),
    )


    # Log violations to a file
    with open(log_file, "w") as f:
        f.write("Line Constraint Violations:\n")
        for line_idx, times in overall_line_violations.items():
            for t, count in times.items():
                f.write(f"Line {line_idx}: Time Step {t}, {count} violations\n")

        f.write("\nTransformer Constraint Violations:\n")
        for t, count in overall_trafo_violations.items():
            f.write(f"Time Step {t}, {count} violations\n")

        f.write("\nMaximum Violations:\n")
        f.write(f"Line {max_violations_line[0]}, Time Step {max_violations_line[1]}: {max_violations_line[2]} violations\n")


    # Calculate the number of simulations with at least one violation
    num_simulations_with_violations = sum(1 for count in violation_counts if count > 0)

    # Get total constraints checked
    num_line_constraints = len(net.line)  # Total number of lines
    num_trafo_constraints = len(net.trafo)  # Total number of transformers
    number_of_constraints = num_line_constraints + num_trafo_constraints
    # Calculate probability of constraint violation
    total_violations = sum(violation_counts)
    # Calculate total number of constraints checked
    total_constraints = len(mc_samples) * len(time_steps) * number_of_constraints
    # Calculate the probability of a single constraint being violated
    violation_probability = total_violations / total_constraints

    violation_probability_samples = num_simulations_with_violations / len(mc_samples)

    total_time = time.time() - start_time

    # Calculate violation probabilities
    # print("Overall Line Violations:")
    # Extract line_indices from mc_line_results_df
    line_indices = mc_line_results_df['line'].unique()
    for line_idx, times in overall_line_violations.items():
        print(f"Line {line_idx}: {times}")
    # Aggregate violation probabilities
    violations_df = aggregate_line_violations(overall_line_violations, len(mc_samples), time_steps, line_indices)
    print("Aggregated Violations DataFrame:")
    print(violations_df.head(20))
    violations_df['violation_probability_percent'] = violations_df['violation_probability'] * 100

    # Aggregate transformer violations into a DataFrame
    trafo_violations_df = aggregate_trafo_violations(overall_trafo_violations, len(mc_samples), time_steps)
    # Add a probability column to the transformer violations DataFrame
    trafo_violations_df['violation_probability_percent'] = trafo_violations_df['violation_probability'] * 100
    # Print or log transformer violations
    print("Transformer Violations DataFrame:")
    print(trafo_violations_df.head(20))
                                                     

    print(f"Monte Carlo analysis completed for {len(mc_samples)} samples in parallel.")
    print(f"Total time taken: {total_time:.2f} seconds.")
    # print(f"Probability of constraint violation: {violation_probability:.4f}")
    print(f"Violation log saved to {log_file}")
    

    return all_results, violation_probability, violations_df, trafo_violations_df, overall_line_violations, mc_line_results_df

