"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
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
hp_scaling = 20 # heat pump scaling factor
hh_scaling = 0.99 # household scaling factor

### THERMAL SYSTEM ###
hp_max_power = 10 # heat pump max power in MW
ts_size_mwh = 48 # thermal storage size in MWh
ts_sof_init = 0.5 # initial state of fill of thermal storage
ts_eff = 0.95 # thermal storage efficiency
ts_out_max = 0.45 # thermal storage max output in MWth
ts_in_max = 0.45 # thermal storage max input in MWth
tsnet_eff = 0.90 # thermal network efficiency
#COP = 3 # heat pump COP
eta_c0 = 0.6 # carnot efficiency

DeltaT = 2.1966# temperature difference in K
T_S = 75+273.15 # temperature of the heat source in K

### Optimization Costs ###
import_cost = 80  # €/MW for importing power from the external grid
export_cost = 80  # €/MW for exporting power to the external grid
curtailment_cost = 150  # €/MW for curtailing PV (set higher than import/export costs)
flexibility_cost = 200  # €/MW for using flexibility 
c_cost = 1 #€/MWh cost of ts energy storage in €/MWh

### TIME HORIZON ###

    
### CONVERGENCE CRITERIA ###
ETA_LF = 1e-4 # bfs standalone #ETA_BFS RENAMED
ETA_OPF = 5e-4 # bfs-opf voltage mismatch #ETA_BFSOPF RENAMED
#ETA_MARG_V = 1e-1 # bus voltage uncertainty margin


### ITERATION COUNTERS ###
M_MAX = 1 # maximum iterations outer CC loop
M_MIN = 1 # minimum iterations for outer CC loop
B_MAX = 5 # maximum iterations opf
K_MAX = 5 # maximum inner lf iterations


## DRCC ##
DRCC_FLG = 1 # DRCC flag, 1 = enable, 0 = disable
epsilon = 0.05 # CC violation probability
### FORECAST ###
N_MC = 10 # number of samples for monte-carlo simulation


