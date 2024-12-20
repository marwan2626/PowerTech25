# -*- coding: utf-8 -*-
"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
Optimization by Marwan Mostafa 

based on the original code in Master Thesis by Dominic Scotoni

Parameter File
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
gp.setParam("OutputFlag",0) # solver output, enable = 1
# gp.setParam("DualReductions", 0) # check if feasible or unbounded: enable = 0
# gp.setParam("MIPGap",2e-4) # MIP gap, default = 1e-4


###############################################################################
## GENERAL ## 
###############################################################################
### NETWORK ###
N_BUS = 18 # number of buses
N_PH = 1 # number of phases - {1,3}
S_BASE = 0.1 # base power [MVA] 


### TIME HORIZON ###
TIME_HORZ = 24 # time horizon [h]
TIMESTEP = 0.25 # timestep [h]
T = int(TIME_HORZ/TIMESTEP) # number of timesteps [#]

    
### CONVERGENCE CRITERIA ###
ETA_BFS = 1e-4 # bfs standalone
ETA_BFSOPF = 5e-4 # bfs-opf voltage mismatch
ETA_MARG_V = 1e-1 # bus voltage uncertainty margin

# if jumping solutions for BFS-OPF: weighted average solution update BFS-OPF
BFSUPD = 0 # smooth solution update: enable = 1, disable = 0
A_BFS = 0.90 # factor


### ITERATION COUNTERS ###
M_MAX = 1 # maximum iterations outer CC loop
M_MIN = 1 # minimum iterations for outer CC loop
B_MAX = 5 # maximum iterations bfs-opf
K_MAX = 5 # maximum inner bfs iterations


### FORECAST ###
V_FCST = 1 # forecast version, for definition see forecast script header
PV_MAX = 8 # 8 kWp installations for data set to normalize
N_DAY = 2 # number of days for monte-carlo simulation


###############################################################################
## FLAGS: DISABLE = 0 , ENABLE = 1 ## 
###############################################################################
### UNITS ###
FLGBAT = 1 # BESS
FLGSHED = 0 # load shedding
FLGSHIFT = 0 # load shifting
FLGCURT = 1 # active power curtailment
FLGOLTC = 0 # OLTC trafo
FLGLOAD = 1 # load profile: 0 = constant, 1 = time varying
FLGPF = 0 # power factor limit PV inverters
FLGPV = 0 # installed capacity PV from input file: 0 = input data, 1 = load dependent
FLGCC = 0 # chance constraints
FLGDRCC = 0 # distributionally robust or gaussian: 1 = DR, 0 = Gaussian


###############################################################################
## PARAMETER VARIATION: DISABLE = 0 , ENABLE = 1 ## 
###############################################################################
FLGVAR_LOAD = 0 # load variation
FLGVAR_PV = 0 # PV variation
FCSTCASE = ['winter'] # seasonal forecast
if FLGVAR_LOAD == 0 and FLGVAR_PV == 0:
    LOADCASE = [1] # single case with nominal load
    PVCASE = [0.5] # single case with nominal PV
elif FLGVAR_LOAD == 1 and FLGVAR_PV == 0:
    LOADCASE = [0.75,1,1.25] # load variation
    PVCASE = [0.5] # single case with nominal PV
elif FLGVAR_LOAD == 1 and FLGVAR_PV == 1:
    LOADCASE = [0.5,1,1.5] # load variation
    PVCASE = [0.5,1,1.5] # single case with nominal PV
elif FLGVAR_LOAD == 0 and FLGVAR_PV == 1:
    LOADCASE = [1] # load variation
    PVCASE = [0.5,1,1.5] # single case with nominal PV
    
### UNBALANCED LOADING ###
# share of total load/PV to phase a,b,c
UNBALANCE = 'symmetric' # degree of unbalance (symmetric,ligthly,heavily)
if N_PH == 3:
    if UNBALANCE == 'symmetric':
        LOADSHARE = [1/3,1/3,1/3]
        PVSHARE = LOADSHARE
    elif UNBALANCE == 'lightly':
        LOADSHARE = [0.35,0.25,0.4]
        PVSHARE = LOADSHARE
    elif UNBALANCE == 'heavily':
        LOADSHARE = [0.2,0.15,0.65]
        PVSHARE = LOADSHARE
else:
    LOADSHARE = [1]
    PVSHARE = LOADSHARE
    
    

###############################################################################
## CHANCE-CONSTRAINTS ## 
###############################################################################
# if jumping solutions: weighted average solution for uncertainty margin
MARGUPD = 1 # enable = 1, disable = 0
A_MARG = 0.95 # factor


### UNCERTAINTY MARGIN ###
# power ratio gamma
FLGCC_GMA = 0 # pre-defined gamma or from OPF: pre-defined = 0 - from OPF = 1
power_factor = 0.95 # pre-defined power factor
CC_GMA = np.sqrt((1-power_factor**2)/power_factor**2) # pre-defined power ratio






