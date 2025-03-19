"""
Code for the publication Reliability-Constrained Thermal Storage Sizing in
Multi-Energy Systems under Uncertainty
Optimization by Marwan Mostafa 

Parameters File
"""

###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### PACKAGES ####
import gurobipy as gp
import numpy as np

###############################################################################
## GUROBI PARAMETERS ## 
###############################################################################
# gp.setParam("NonConvex",-1) # enable non convex constraints, enable = 2
#gp.setParam("OutputFlag",0) # solver output, enable = 1
# gp.setParam("DualReductions", 0) # check if feasible or unbounded: enable = 0
# gp.setParam("MIPGap",2e-4) # MIP gap, default = 1e-4


###############################################################################
## GENERAL ## 
###############################################################################
### NETWORK ###
hp_scaling = 30 # heat pump scaling factor
hh_scaling = 1 # household scaling factor

max_trafo_loading = 0.7 # max transformer loading

### THERMAL SYSTEM ###
hp_max_power = 0.5 # heat pump max power in MW
hp_pf = 0.98
Q_scaling = 0.203
ts_sof_init = 0.5 # initial state of fill of thermal storage
ts_eff = 0.90 # thermal storage efficiency
ts_alpha = 0.0001 # thermal storage heat loss factor
ts_out_max = 0.45 # thermal storage max output in MWth
ts_in_max = 0.45 # thermal storage max input in MWth
tsnet_eff = 0.95 # thermal network efficiency
#COP = 3 # heat pump COP
eta_c0 = 0.6 # carnot efficiency
psi = 0.13
DeltaT = 2.1966# temperature difference in K
T_S = 75+273.15 # temperature of the heat source in K

### Optimization Costs ###
curtailment_cost = 150  # €/MW for curtailing PV (set higher than import/export costs)
HNS_cost = 5000  # €/MW for HNS
c_cost = 8000 #€/MWh cost of ts energy storage in €/MWh

## DRCC ##
DRCC_FLG = 1 # DRCC flag, 1 = enable, 0 = disable
epsilon = 0.05 # CC violation probability
### FORECAST ###
N_MC = 1000 # number of samples for monte-carlo simulation


# Define reliability parameters
failure_rate_trafo = 0.0054/168  # failures per hour for transformer
repair_time_trafo = 10      # hours to repair transformer
failure_rate_hp = 0.1/168      # failures per hour for heat pump
repair_time_hp = 24         # hours to repair heat pump
failure_rate_ts = 0.001     # failures per hour for thermal storage
repair_time_ts = 0         # hours to repair thermal storage

N_scenarios = 200000  # Number of Monte Carlo scenarios
TS_capacity = 0.755075552


#failure_timestep_trafo = 120
#failure_timestep_hp = 48
#failure_timestep_ts = 10
