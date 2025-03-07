#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:44:23 2025

@author: ehansen
"""

from deterministic1dsolver import Ordinate1DSolver,Spectral1DSolver
from montecarlo1dsolver import MonteCarlo1DSolver
import time  
import matplotlib.pyplot as plt
import numpy as np
from numpy import format_float_positional as ff

Nx = 160

def simpledistanceabsorb(N,x,mu,name):
    montecarlo = MonteCarlo1DSolver([100], [1], [0], [0], Nx, N*10**2, 
                                    (1,1), (1,0), (0,0), name)
    
    montecarlo.simple_distanceabsorb(x, mu)
    
    return(montecarlo)
    
Ns = [1,4,16]
mus = [1,-1,1/np.sqrt(3),-1/np.sqrt(3)]
xs = [0,100,0,100]
casename = ["mu1","mun1","mu13","mun13"]

for i in range(0,4):
    
    casesims = []
    for n in Ns:
        montecarlo = simpledistanceabsorb(n,xs[i],mus[i],"simpleleft"+casename[i])
        casesims.append(casename)
        
    
        
        

# Consider a couple different cases of sources and transport cross sections
qs = [1,2]
sigmats = [1,2]

for q in qs:
    for t in sigmats:
        reflectcasesims = []
        for n in Ns:
            montecarlo =  MonteCarlo1DSolver([100], [t], [0], [q], Nx, n*10**4, 
                                            (0,0), (0,0), (0,0), "simplereflect"+"q"+str(q)+"st"+str(t))
            montecarlo.simple_locationdirectionreflect(travellength=1/t)
    
            reflectcasesims.append(montecarlo)
            
# Consider arbitrary angle flux coming 
leftcasesims = []
for n in Ns:
    montecarlo = MonteCarlo1DSolver([100], [1], [0], [0], Nx, n*10**2, 
                                    (1,1), (1,0), (0,0), "simplefluxmu")
    montecarlo.simple_angledistanceabsorb(0)
    leftcasesims.append(montecarlo)
        
        


    
