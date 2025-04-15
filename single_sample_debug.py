import pickle
import plotly.express as px
import pandas as pd

import montecarlo_validation as mc
import results as rs 
import parameters as par
import griddata as gd

net, time_steps, const_pv, const_load_household_P, const_load_household_Q, const_load_heatpump, const_load_heatpump_Q ,df_household_prognosis, df_season_heatpump_prognosis, df_season_pv_prognosis, heatpump_scaling_factors_df, T_amb, electricity_price, original_sgen_p_mw = gd.setup_grid_IAS(season='winter')

mc_samples = mc.generate_samples_ldf(df_season_heatpump_prognosis,df_season_pv_prognosis)

with open("mc_samples.pkl", "wb") as f:
    pickle.dump(mc_samples, f)

# with open("mc_samples.pkl", "rb") as f:
#     mc_samples = pickle.load(f)

sample_profile = mc_samples[0]

filename = f"drcc_results_drcc_{par.DRCC_FLG}_{par.epsilon}.pkl"
drcc_results = rs.load_optim_results(filename)

# === Run the function for a single sample ===
res_loads, res_buses, res_lines, res_trafos, res_sgen, *_ = mc.run_single_ldf_sample_with_violation(
    net,
    original_sgen_p_mw,
    time_steps,
    sample_profile,
    drcc_results,
    heatpump_scaling_factors_df,
    const_load_household_P,
    const_load_household_Q
)

# === Add index column for grouping in plotly ===
for df in [res_loads, res_buses, res_lines, res_trafos, res_sgen]:
    df['index'] = df.index % len(df[df['time_step'] == df['time_step'].iloc[0]])

# === Plotting function ===
def plot_result(df, value_col, title, yaxis_label):
    fig = px.line(df, x="time_step", y=value_col, color="index", markers=True,
                  title=title,
                  labels={"time_step": "Time Step", value_col: yaxis_label})
    fig.update_layout(template="plotly_white", legend_title_text="Index")
    fig.show()

# === Create plots ===
plot_result(res_loads, "p_mw", "Load  Power Over Time", "Load [MW]")
plot_result(res_buses, "vm_pu", "Bus Voltage Magnitude Over Time", "Voltage [p.u.]")
plot_result(res_lines, "loading_percent", "Line Loading Over Time", "Loading [%]")
plot_result(res_trafos, "loading_percent", "Transformer Loading Over Time", "Loading [%]")
plot_result(res_sgen, "p_mw", "Generator Active Power Over Time", "Generator [MW]")