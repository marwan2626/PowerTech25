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
ts_size_mwh = 48 # thermal storage size in MWh
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
c_cost = 5200 #€/MWh cost of ts energy storage in €/MWh

### TIME HORIZON ###

    
### CONVERGENCE CRITERIA ###
ETA_LF = 1e-4 # bfs standalone #ETA_BFS RENAMED
ETA_OPF = 5e-4 # bfs-opf voltage mismatch #ETA_BFSOPF RENAMED
#ETA_MARG_V = 1e-1 # bus voltage uncertainty margin


## DRCC ##
DRCC_FLG = 1 # DRCC flag, 1 = enable, 0 = disable
epsilon = 0.1 # CC violation probability
### FORECAST ###
N_MC = 10 # number of samples for monte-carlo simulation


