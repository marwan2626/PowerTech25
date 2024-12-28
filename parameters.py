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
hp_scaling = 92 # heat pump scaling factor
hh_scaling = 0.75 # household scaling factor

hp_max_power = 10 # heat pump max power in MW
ts_size_mwh = 48 # thermal storage size in MWh
ts_sof_init = 0.5 # initial state of fill of thermal storage
ts_eff = 0.95 # thermal storage efficiency
ts_out_max = 0.45 # thermal storage max output in MWth
ts_in_max = 0.45 # thermal storage max input in MWth
tsnet_eff = 0.90 # thermal network efficiency
COP = 3 # heat pump COP

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


### FORECAST ###
N_MC = 10 # number of samples for monte-carlo simulation


###############################################################################
## FLAGS: DISABLE = 0 , ENABLE = 1 ## 
###############################################################################
### UNITS ###
#FLGBAT = 1 # BESS
#FLGSHED = 0 # load shedding
#FLGSHIFT = 0 # load shifting
#FLGCURT = 1 # active power curtailment
#FLGOLTC = 0 # OLTC trafo
#FLGLOAD = 1 # load profile: 0 = constant, 1 = time varying
#FLGPF = 0 # power factor limit PV inverters
#FLGPV = 0 # installed capacity PV from input file: 0 = input data, 1 = load dependent
#FLGCC = 0 # chance constraints
#FLGDRCC = 0 # distributionally robust or gaussian: 1 = DR, 0 = Gaussian



###############################################################################
## CHANCE-CONSTRAINTS ## 
###############################################################################
### UNCERTAINTY MARGIN ###
# power ratio gamma
#FLGCC_GMA = 0 # pre-defined gamma or from OPF: pre-defined = 0 - from OPF = 1
#power_factor = 0.95 # pre-defined power factor
#CC_GMA = np.sqrt((1-power_factor**2)/power_factor**2) # pre-defined power ratio


