#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 11:12:52 2025

@author: ehansen
"""

import numpy as np
import matplotlib.pyplot as plt

from deterministic1dsolver import Ordinate1DSolver,Spectral1DSolver
from montecarlo1dsolver import MonteCarlo1DSolver



Nx = [40,160,640]

bounds = (1,1)

scatterab = [[1],[0.99],[0.01],[0.0]]

# There are 9 runs to perform for each Nx, so we generate lists to loop through

lengths = []
sources = []
scatter = []
transport = []
forwards = []
backwards = []
probname = []

for i in range(4):
    lengths.append([100])
    sources.append([0])
    scatter.append(scatterab[i])
    transport.append([1])
    forwards.append((1,0))
    backwards.append((0,0))
    probname.append("leftsource/")
    
for i in range(4):
    lengths.append(100)
    sources.append([1])
    scatter.append(scatterab[i])
    transport.append([1])
    forwards.append((0,0))
    backwards.append((0,0))
    probname.append("midsource/")
    
lengths.append([20,80])
sources.append([0,0])
scatter.append([2,0.006])
transport.append([10,0.01])
forwards.append((0,0))
backwards.append((0,1))
probname.append("absorbair/")


def simulationcomparison(prob,Nx):    
    
    ordinate = Ordinate1DSolver(lengths[prob], transport[prob],scatter[prob],
                                sources[prob], Nx,64,bounds,forwards[prob],
                                backwards[prob],probname[prob])
    ordinate.solve()

    spectral = Spectral1DSolver(lengths[prob], transport[prob],scatter[prob],
                                sources[prob], Nx,64,bounds,forwards[prob],
                                backwards[prob],probname[prob])
    spectral.solve()  
    
    return([ordinate.scalarflux,spectral.scalarflux])

montecarlo = MonteCarlo1DSolver([100], [1], [0.99], [0], 160, 10**4, 
                                (1,1), (1,0), (0,0), "highscattersourcefree")
montecarlo.simulation()

deterministicconvergence=False
if deterministicconvergence:
    for prob in range(0,9):
        
        bestord,bestspec,bestmc = simulationcomparison(prob, 2560)
        
        fig1,ax1 = plt.subplots(1)
        fig2,ax2 = plt.subplots(2)
        
        for n in Nx[::-1]:
            
            ordinate,spectral,mc= simulationcomparison(prob,n)
            
            meshdiff = np.size(bestord)//np.size(ordinate)
            
            # Compute L1 norms of errors based on leftmost points in best grid
            
            err1 = np.sum(np.abs(bestord[::meshdiff]-ordinate))/np.size(ordinate)
            err2 = np.sum(np.abs(bestspec[::meshdiff]-spectral))/np.size(spectral)
            
            ax1.plot(n,err1,"ks")
            ax2.plot(n,err2,"ks")
        
        ax1.set_xlabel("$N_x$")
        ax1.set_ylabel("Average Error")
        ax2.set_xlabel("$N_x$")
        ax2.set_ylabel("Average Error")
        
        ax1.set_xscale("log")
        ax1.set_xscale("log")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        
        fig1.suptitle("Convergence of S_N ")
        fig2.suptitle("Diffusion Convergence")
        
        fig1.savefig("ordinateconvergence"+str(prob))
        fig2.savefig("diffusionconvergence"+str(prob))
        
        fig1.close()
        fig2.close()
        