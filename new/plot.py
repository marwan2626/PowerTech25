"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
Optimization by Marwan Mostafa 

Plot File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### PACKAGES ####

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pandapower.plotting.plotly as pp_plotly
import seaborn as sns
import matplotlib.ticker as ticker

###############################################################################
def plot_load_p_mw(load_p_mw):
    plt.figure(figsize=(10, 5))
    for column in load_p_mw.columns:
        plt.plot(load_p_mw.index, load_p_mw[column], label=f'Load {column}')
    plt.xlabel('Time Step')
    plt.ylabel('Load Power (MW)')
    plt.title('Load Power over Time')
    plt.legend()
    plt.show()

def plot_sgen_p_mw(sgen_p_mw):
    plt.figure(figsize=(10, 5))
    for column in sgen_p_mw.columns:
        plt.plot(sgen_p_mw.index, sgen_p_mw[column], label=f'Load {column}')
    plt.xlabel('Time Step')
    plt.ylabel('PV Power (MW)')
    plt.title('PV generation over Time')
    plt.legend()
    plt.show()

def plot_line_loading_percent(line_loading_percent):
    plt.figure(figsize=(10, 5))
    for column in line_loading_percent.columns:
        plt.plot(line_loading_percent.index, line_loading_percent[column], label=f'Load {column}')
    plt.xlabel('Time Step')
    plt.ylabel('Loading Percentage %')
    plt.title('Line Loading over Time')
    plt.legend()
    plt.show()


def plot_line_loading_percent2(loading_percent):
    plt.figure(figsize=(10, 5))
    time_steps = loading_percent.index  # Use the index of loading_percent as the time steps
    for column in loading_percent.columns:
        plt.plot(time_steps, loading_percent[column], label=f'Line {column}')
    plt.xlabel('Time Step')
    plt.ylabel('Loading Percentage %')
    plt.title('Line Loading over Time')
    plt.legend()
    plt.show()

# Function to plot line current magnitude
def plot_line_current_magnitude(i_ka):
    plt.figure(figsize=(10, 5))
    time_steps = i_ka.index  # Use the index of i_ka as the time steps
    for column in i_ka.columns:
        plt.plot(time_steps, i_ka[column], label=f'Line {column}')
    plt.xlabel('Time Step')
    plt.ylabel('Current Magnitude (kA)')
    plt.title('Line Current Magnitude over Time')
    plt.legend()
    plt.show()

# Function to plot line power flow
def plot_line_power_flow(pl_mw):
    plt.figure(figsize=(10, 5))
    time_steps = pl_mw.index  # Use the index of pl_mw as the time steps
    for column in pl_mw.columns:
        plt.plot(time_steps, pl_mw[column], label=f'Line {column}')
    plt.xlabel('Time Step')
    plt.ylabel('Power Flow (MW)')
    plt.title('Line Power Flow over Time')
    plt.legend()
    plt.show()

def plot_opf_results(results):
    # Extract the results from the dictionary
    pv_gen = results['pv_gen']
    load = results['load']
    ext_grid_import = results['ext_grid_import']
    ext_grid_export = results['ext_grid_export']
    theta = results['theta']
    line_results = results['line_results']  # Assuming you have 'line_results' in the results
    thermal_storage = results['thermal_storage']  # Thermal storage results
    transformer_loading = results.get('transformer_loading', {})  # Transformer loading results 
    
    # Get the list of time steps
    time_steps = list(pv_gen.keys())

    # Plot PV Generation for each bus
    plt.figure(figsize=(10, 6))
    for bus in pv_gen[time_steps[0]].keys():
        pv_values = [pv_gen[t][bus] for t in time_steps]
        plt.plot(time_steps, pv_values, label=f'Bus {bus}')
    plt.xlabel('Time Steps')
    plt.ylabel('PV Generation (MW)')
    plt.title('PV Generation by Bus and Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Load for each bus
    plt.figure(figsize=(10, 6))
    for bus in load[time_steps[0]].keys():
        load_values = [load[t][bus] for t in time_steps]
        plt.plot(time_steps, load_values, label=f'Bus {bus}')
    plt.xlabel('Time Steps')
    plt.ylabel('Load (MW)')
    plt.title('Load by Bus and Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot External Grid Import/Export
    plt.figure(figsize=(10, 6))
    ext_import_values = [ext_grid_import[t] for t in time_steps]
    ext_export_values = [ext_grid_export[t] for t in time_steps]
    plt.plot(time_steps, ext_import_values, label='External Grid Import (MW)', color='green')
    plt.plot(time_steps, ext_export_values, label='External Grid Export (MW)', color='red')
    plt.xlabel('Time Steps')
    plt.ylabel('Power (MW)')
    plt.title('External Grid Import and Export over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Transformer Loading Percentage
    plt.figure(figsize=(10, 6))
    transformer_loading_values = [transformer_loading[t] for t in time_steps]
    plt.plot(time_steps, transformer_loading_values, label='Transformer Loading (%)', color='blue')
    plt.xlabel('Time Steps')
    plt.ylabel('Loading (%)')
    plt.title('Transformer Loading Percentage over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Theta (Voltage Angles) for each bus
    plt.figure(figsize=(10, 6))
    for bus in theta[time_steps[0]].keys():
        theta_values = [theta[t][bus] for t in time_steps]
        plt.plot(time_steps, theta_values, label=f'Bus {bus}')
    plt.xlabel('Time Steps')
    plt.ylabel('Voltage Angle Theta (Radians)')
    plt.title('Voltage Angle Theta by Bus and Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Line Power Flow (MW) for each line
    plt.figure(figsize=(10, 6))
    for line in line_results[time_steps[0]]['line_pl_mw'].keys():
        line_pl_mw_values = [line_results[t]['line_pl_mw'][line] for t in time_steps]
        plt.plot(time_steps, line_pl_mw_values, label=f'Line {line}')
    plt.xlabel('Time Steps')
    plt.ylabel('Line Power Flow (MW)')
    plt.title('Line Power Flow (MW) by Line and Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Line Loading Percentage for each line
    plt.figure(figsize=(10, 6))
    for line in line_results[time_steps[0]]['line_loading_percent'].keys():
        line_loading_values = [line_results[t]['line_loading_percent'][line] for t in time_steps]
        plt.plot(time_steps, line_loading_values, label=f'Line {line}')
    plt.xlabel('Time Steps')
    plt.ylabel('Line Loading (%)')
    plt.title('Line Loading Percentage by Line and Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Line Current Magnitude (kA) for each line
    plt.figure(figsize=(10, 6))
    for line in line_results[time_steps[0]]['line_current_mag'].keys():
        line_current_values = [line_results[t]['line_current_mag'][line] for t in time_steps]
        plt.plot(time_steps, line_current_values, label=f'Line {line}')
    plt.xlabel('Time Steps')
    plt.ylabel('Line Current Magnitude (kA)')
    plt.title('Line Current Magnitude (kA) by Line and Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot SOF (State of Fill) for each bus with thermal storage
    plt.figure(figsize=(10, 6))
    for bus in thermal_storage['ts_sof'][time_steps[0]].keys():
        sof_values = [thermal_storage['ts_sof'][t][bus] for t in time_steps]
        plt.plot(time_steps, sof_values, label=f'Bus {bus}')
    plt.xlabel('Time Steps')
    plt.ylabel('State of Fill (SOF)')
    plt.title('Thermal Storage State of Fill (SOF) by Bus and Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Power In and Power Out for Thermal Storage for each bus
    plt.figure(figsize=(10, 6))
    for bus in thermal_storage['ts_in'][time_steps[0]].keys():
        ts_in_values = [thermal_storage['ts_in'][t][bus] for t in time_steps]
        ts_out_values = [thermal_storage['ts_out'][t][bus] for t in time_steps]
        plt.plot(time_steps, ts_in_values, label=f'Power In (Bus {bus})', linestyle='--')
        plt.plot(time_steps, ts_out_values, label=f'Power Out (Bus {bus})', linestyle='-')
    plt.xlabel('Time Steps')
    plt.ylabel('Power (MW)')
    plt.title('Thermal Storage Power In and Power Out by Bus and Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()





def plot_curtailment(curtailment_factor, time_steps):
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, curtailment_factor)
    plt.xlabel('Time Steps')
    plt.ylabel('Curtailment Factor')
    plt.title('Curtailment Factor Over Time')
    plt.ylim(0, 1)  # Limits to keep the plot between 0 and 1
    plt.grid(True)
    plt.show()

def plot_line_current_histogram(all_results, net, line_index, time_step):
    # Collect line current data for the specified line and time step across all samples
    line_current_magnitudes = []
    for sample_results in all_results:
        line_results = sample_results[2]  # Access the line DataFrame from the tuple (loads, buses, lines, trafos)
        # Filter for the specific time step and retrieve the current for the specified line index
        time_step_data = line_results[line_results['time_step'] == time_step]
        
        if not time_step_data.empty:
            line_current_magnitudes.append(time_step_data['i_ka'].values[line_index])

    if not line_current_magnitudes:
        print(f"No data found for line index {line_index} at time step {time_step} across samples.")
        return

    # Convert to Series for easier statistical calculations
    line_current_magnitudes = pd.Series(line_current_magnitudes)
    
    # Calculate the 95th percentile
    percentile_95 = line_current_magnitudes.quantile(0.95)
    
    # Get the max allowable current from the network data
    max_i_ka = net.line.loc[line_index, 'max_i_ka']
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(line_current_magnitudes, bins=100, color='lightblue', edgecolor='black', alpha=0.7, label='Line Current Magnitude')
    
    # Plot the 95th percentile and max_i_ka as vertical lines
    plt.axvline(percentile_95, color='orange', linestyle='--', linewidth=2, label=f'95th Percentile: {percentile_95:.3f} kA')
    plt.axvline(max_i_ka, color='red', linestyle='-', linewidth=2, label=f'Max Allowable: {max_i_ka:.3f} kA')
    
    # Add labels and legend
    plt.xlabel('Line Current Magnitude (kA)')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Line Current Magnitude for Line {line_index} at Time Step {time_step}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_opf_results_plotly(results):
    # Extract the results from the dictionary
    pv_gen = results['pv_gen']
    load = results['load']
    ext_grid_import = results['ext_grid_import']
    ext_grid_export = results['ext_grid_export']
    theta = results['theta']
    line_results = results['line_results']
    thermal_storage = results['thermal_storage']
    transformer_loading = results.get('transformer_loading', {})
    
    # Get the list of time steps
    time_steps = list(pv_gen.keys())

    # # Plot pandapower grid using simple_plotly
    # fig = pp_plotly.simple_plotly(net)
    # fig.show()

    # Plot PV Generation
    fig = go.Figure()
    for bus in pv_gen[time_steps[0]].keys():
        pv_values = [pv_gen[t][bus] for t in time_steps]
        fig.add_trace(go.Scatter(x=time_steps, y=pv_values, mode='lines', name=f'Bus {bus}'))
    fig.update_layout(title='PV Generation by Bus and Time Step',
                    xaxis_title='Time Steps',
                    yaxis_title='PV Generation (MW)',
                    hovermode="x unified")
    fig.show()

    # Plot Heat Pump Dispatch (Flexible Loads)
    fig = go.Figure()
    for bus in load[time_steps[0]]['flexible_loads'].keys():
        flexible_load_values = [load[t]['flexible_loads'][bus] for t in time_steps]
        fig.add_trace(go.Scatter(x=time_steps, y=flexible_load_values, mode='lines', name=f'Bus {bus}'))
    fig.update_layout(title='Heat Pump Dispatch (Flexible Loads) by Bus and Time Step',
                      xaxis_title='Time Steps',
                      yaxis_title='Heat Pump Dispatch (MW)',
                      hovermode="x unified")
    fig.show()

    # Plot External Grid Import/Export
    fig = go.Figure()
    ext_import_values = [ext_grid_import[t] for t in time_steps]
    ext_export_values = [ext_grid_export[t] for t in time_steps]
    fig.add_trace(go.Scatter(x=time_steps, y=ext_import_values, mode='lines', name='Import (MW)', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=time_steps, y=ext_export_values, mode='lines', name='Export (MW)', line=dict(color='red')))
    fig.update_layout(title='External Grid Import and Export over Time',
                    xaxis_title='Time Steps',
                    yaxis_title='Power (MW)',
                    hovermode="x unified")
    fig.show()

    # Plot Household Load (Non-Flexible Loads)
    fig = go.Figure()
    for bus in load[time_steps[0]]['non_flexible_loads'].keys():
        non_flexible_load_values = [load[t]['non_flexible_loads'][bus] for t in time_steps]
        fig.add_trace(go.Scatter(x=time_steps, y=non_flexible_load_values, mode='lines', name=f'Bus {bus}'))
    fig.update_layout(title='Household Load (Non-Flexible Loads) by Bus and Time Step',
                      xaxis_title='Time Steps',
                      yaxis_title='Household Load (MW)',
                      hovermode="x unified")
    fig.show()

    # Plot Transformer Loading
    fig = go.Figure()
    transformer_loading_values = [transformer_loading[t] for t in time_steps]
    fig.add_trace(go.Scatter(x=time_steps, y=transformer_loading_values, mode='lines', name='Transformer Loading (%)'))
    fig.update_layout(title='Transformer Loading Percentage over Time',
                    xaxis_title='Time Steps',
                    yaxis_title='Loading (%)',
                    hovermode="x unified")
    fig.show()

    # Plot Theta (Voltage Angles)
    fig = go.Figure()
    for bus in theta[time_steps[0]].keys():
        theta_values = [theta[t][bus] for t in time_steps]
        fig.add_trace(go.Scatter(x=time_steps, y=theta_values, mode='lines', name=f'Bus {bus}'))
    fig.update_layout(title='Voltage Angle Theta by Bus and Time Step',
                    xaxis_title='Time Steps',
                    yaxis_title='Voltage Angle (Radians)',
                    hovermode="x unified")
    fig.show()

    # Plot Line Power Flow
    fig = go.Figure()
    for line in line_results[time_steps[0]]['line_pl_mw'].keys():
        line_pl_mw_values = [line_results[t]['line_pl_mw'][line] for t in time_steps]
        fig.add_trace(go.Scatter(x=time_steps, y=line_pl_mw_values, mode='lines', name=f'Line {line}'))
    fig.update_layout(title='Line Power Flow (MW) by Line and Time Step',
                    xaxis_title='Time Steps',
                    yaxis_title='Line Power Flow (MW)',
                    hovermode="x unified")
    fig.show()

    # Plot Line Current
    fig = go.Figure()
    for line in line_results[time_steps[0]]['line_current_mag'].keys():
        line_current_values = [line_results[t]['line_current_mag'][line] for t in time_steps]
        fig.add_trace(go.Scatter(x=time_steps, y=line_current_values, mode='lines', name=f'Line {line}'))
    fig.update_layout(title='Line Current Magnitude (kA) by Line and Time Step',
                    xaxis_title='Time Steps',
                    yaxis_title='Line Current Magnitude (kA)',
                    hovermode="x unified")
    fig.show()

    # Plot Line Current Magnitude as Bar Plot
    line_min_values = {}
    line_max_values = {}
    line_mean_values = {}
    # Compute min, max, and mean for each line
    for line in line_results[time_steps[0]]['line_current_mag'].keys():
        line_current_values = [line_results[t]['line_current_mag'][line] for t in time_steps]
        line_min_values[line] = min(line_current_values)
        line_max_values[line] = max(line_current_values)
        line_mean_values[line] = sum(line_current_values) / len(line_current_values)
    # Create the bar plot
    fig = go.Figure()   
    # Add bars for min-max range
    for line in line_min_values.keys():
        fig.add_trace(go.Bar(
            x=[line],  # Line index
            y=[line_max_values[line] - line_min_values[line]],  # Height of the bar is the range
            base=[line_min_values[line]],  # Base starts at the min value
            name=f'Line {line}',
            marker=dict(color='blue', opacity=0.7)
        ))
    # Add a line for the mean value
    fig.add_trace(go.Scatter(
        x=list(line_mean_values.keys()),  # Line indices
        y=list(line_mean_values.values()),  # Mean current magnitudes
        mode='lines+markers',
        name='Mean Current Magnitude',
        line=dict(color='red', width=2),
        marker=dict(size=8)
    ))
    # Update layout for the plot
    fig.update_layout(
        title='Line Current Magnitude (Min, Max, and Mean) by Line',
        xaxis_title='Line Index',
        yaxis_title='Current Magnitude (kA)',
        barmode='overlay',
        hovermode="x unified"
    )

    fig.show()


    # Plot Line Loading
    fig = go.Figure()
    for line in line_results[time_steps[0]]['line_loading_percent'].keys():
        line_loading_values = [line_results[t]['line_loading_percent'][line] for t in time_steps]
        fig.add_trace(go.Scatter(x=time_steps, y=line_loading_values, mode='lines', name=f'Line {line}'))
    fig.update_layout(title='Line Loading Percentage by Line and Time Step',
                    xaxis_title='Time Steps',
                    yaxis_title='Line Loading (%)',
                    hovermode="x unified")
    fig.show()

    # Plot SOF (State of Fill)
    fig = go.Figure()
    for bus in thermal_storage['ts_sof'][time_steps[0]].keys():
        sof_values = [thermal_storage['ts_sof'][t][bus] for t in time_steps]
        fig.add_trace(go.Scatter(x=time_steps, y=sof_values, mode='lines', name=f'Bus {bus}'))
    fig.update_layout(title='Thermal Storage State of Fill (SOF) by Bus and Time Step',
                    xaxis_title='Time Steps',
                    yaxis_title='State of Fill (SOF)',
                    hovermode="x unified")
    fig.show()

    # Plot Power In and Out for Thermal Storage
    fig = go.Figure()
    for bus in thermal_storage['ts_in'][time_steps[0]].keys():
        ts_in_values = [thermal_storage['ts_in'][t][bus] for t in time_steps]
        ts_out_values = [thermal_storage['ts_out'][t][bus] for t in time_steps]
        fig.add_trace(go.Scatter(x=time_steps, y=ts_in_values, mode='lines', name=f'Power In (Bus {bus})', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=time_steps, y=ts_out_values, mode='lines', name=f'Power Out (Bus {bus})'))
    fig.update_layout(title='Thermal Storage Power In and Out by Bus and Time Step',
                    xaxis_title='Time Steps',
                    yaxis_title='Power (MW)',
                    hovermode="x unified")
    fig.show()

def plot_line_violation_boxplot(violations_df):

    # Create the boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="line",
        y="violation_probability_percent",
        data=violations_df,
        color="skyblue",
        showfliers=True,
        boxprops=dict(edgecolor="black"),
    )

    # Customize the plot
    plt.xlabel("Line Index", fontsize=12)
    plt.ylabel("Violation Probability (%)", fontsize=12)
    plt.title("Violation Probability Boxplot for Each Line", fontsize=14)
    plt.xticks(rotation=45)  # Rotate x-axis labels if necessary
    plt.tight_layout()
    plt.show()

def plot_specific_line_violation_boxplot(violations_df, line_index):

    plt.figure(figsize=(12, 6))
    
    # Create a boxplot: group probabilities by 'line'
    sns.boxplot(
        x='line', 
        y='violation_probability', 
        data=violations_df, 
        color='skyblue'
    )
    
    # Add labels and title
    plt.xlabel("Line Index")
    plt.ylabel("Violation Probability")
    plt.title("Boxplot of Violation Probabilities Per Line")
    
    # Ensure x-axis ticks are readable
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_violation_heatmap(violations_df, threshold=0.05):
    """
    Generates a heatmap of violation probabilities with lines on the y-axis and timesteps on the x-axis.

    Args:
        violations_df (pd.DataFrame): DataFrame containing columns 'line', 'time_step', and 'violation_probability'.
        threshold (float): Minimum value for visualization. Values below this will be colored lightly.
    """
    # Pivot the DataFrame to get lines as rows and time_steps as columns
    heatmap_data = violations_df.pivot(index='line', columns='time_step', values='violation_probability')

    # Replace NaN with 0.0 for missing values
    heatmap_data = heatmap_data.fillna(0)

    # Apply a mask for values below the threshold (optional for visual emphasis)
    mask = heatmap_data < threshold

    # Create figure and axis with adjusted size
    fig, ax = plt.subplots(figsize=(40, 10))  # Explicitly set figure size

    ax = sns.heatmap(
        heatmap_data,
        cmap="Blues",  # Grayscale colormap for better printing
        cbar_kws={'label': 'Violation Probability'},
        annot=False,  # Set to True if you want to display numbers in the cells
        mask=None,    # Do not mask low values; show all values for clarity
        linewidths=0,  # Remove borders around squares
        vmin=0,       # Ensure the color scale starts at 0
        #vmax=1,        # Ensure the color scale covers the full range (0 to 1)
        xticklabels=10,  # Show every 10th x-tick
        yticklabels=1    # Show all y-ticks
    )

    # Add axis labels and title
    plt.xlabel("Time Step")
    plt.ylabel("Line Index")
    plt.title("Violation Probability Heatmap (Lines vs. Time Steps)")

    # Set x-ticks to step by 10
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    # Add a frame around the plot
    ax.figure.gca().spines['top'].set_visible(True)
    ax.figure.gca().spines['right'].set_visible(True)
    ax.figure.gca().spines['left'].set_visible(True)
    ax.figure.gca().spines['bottom'].set_visible(True)

    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

def compare_heatmap(violations_df_opf, violations_df_drcc, threshold=0.05):

    # Pivot the DataFrame to get lines as rows and time_steps as columns
    heatmap_data_opf = violations_df_opf.pivot(index='line', columns='time_step', values='violation_probability')
    heatmap_data_drcc = violations_df_drcc.pivot(index='line', columns='time_step', values='violation_probability')

    # Replace NaN with 0.0 for missing values
    heatmap_data_opf = heatmap_data_opf.fillna(0)
    heatmap_data_drcc = heatmap_data_drcc.fillna(0)

    # Apply a mask for values below the threshold (optional for visual emphasis)
    #mask = heatmap_data < threshold

    # Create figure and axis with adjusted size
    fig, axes = plt.subplots(2, 1, figsize=(40, 20), sharex=False)  # Explicitly set figure size

    sns.heatmap(
        heatmap_data_opf,
        cmap="Blues",  # Grayscale colormap for better printing
        cbar_kws={'label': 'Violation Probability'},
        annot=False,  # Set to True if you want to display numbers in the cells
        mask=None,    # Do not mask low values; show all values for clarity
        linewidths=0,  # Remove borders around squares
        vmin=0,       # Ensure the color scale starts at 0
        #vmax=1,        # Ensure the color scale covers the full range (0 to 1)
        xticklabels=10,  # Show every 10th x-tick
        yticklabels=1,    # Show all y-ticks
        ax=axes[0]
    )

    # Add axis labels and title
    axes[0].set_title("Heatmap for deterministic optimization")
    axes[0].set_ylabel("Line Index")
    axes[0].set_xlabel("Time Step")

    sns.heatmap(
        heatmap_data_drcc,
        cmap="Blues",  # Grayscale colormap for better printing
        cbar_kws={'label': 'Violation Probability'},
        annot=False,  # Set to True if you want to display numbers in the cells
        mask=None,    # Do not mask low values; show all values for clarity
        linewidths=0,  # Remove borders around squares
        vmin=0,       # Ensure the color scale starts at 0
        #vmax=1,        # Ensure the color scale covers the full range (0 to 1)
        xticklabels=10,  # Show every 10th x-tick
        yticklabels=1,    # Show all y-ticks
        ax=axes[1]
    )

    # Add axis labels and title
    axes[1].set_title("Heatmap for DRCC optimization")
    axes[1].set_ylabel("Line Index")
    axes[1].set_xlabel("Time Step")

    # Set x-ticks to step by 10
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    # Add a frame around each subplot
    for ax in axes:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

    plt.tight_layout()
    plt.show()


def box_line_loading_two_subplots(mc_line_results_df1, mc_line_results_df2):
    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(20, 15), sharex=True)

    # Plot the first DataFrame
    sns.boxplot(
        data=mc_line_results_df1,
        x="line",
        y="loading_percent",
        color="lightgray",
        showfliers=True,
        ax=axes[0]
    )
    axes[0].set_ylim(0, 120)  # Set y-axis limits
    axes[0].set_ylabel("Loading Percentage (%)")
    axes[0].set_title("Line Loading Percentages Across Monte Carlo Samples (OPF)")
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Hide x-axis labels for the top plot

    # Plot the second DataFrame
    sns.boxplot(
        data=mc_line_results_df2,
        x="line",
        y="loading_percent",
        color="lightgray",
        showfliers=True,
        ax=axes[1]
    )
    axes[1].set_ylim(0, 120)  # Set y-axis limits
    axes[1].set_xlabel("Line Index")
    axes[1].set_ylabel("Loading Percentage (%)")
    axes[1].set_title("Line Loading Percentages Across Monte Carlo Samples (DRCC)")
    axes[1].tick_params(axis='x', rotation=45, labelsize=10)  # Rotate x-axis labels for better readability

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()
