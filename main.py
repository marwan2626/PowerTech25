"""
Code for the publication Exploiting Flexibility in Multi-Energy Systems
through Distributionally Robust Chance-Constrained
Optimization by Marwan Mostafa 

Main File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### SCRIPTS ####
#import data as dt
import griddata as gd 
#import results as rs
#import plot as pl
#import opf as opf
#import drcc as drcc
#import montecarlo as mc

net = gd.setup_grid_powertech25(season='winter')
