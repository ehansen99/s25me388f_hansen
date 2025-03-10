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



Nx = [40,160,640,2560]
NP = [10**4,2*10**4,4*10**4,10**5]

bounds = []

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
    bounds.append((1,1))
    forwards.append((1,0))
    backwards.append((0,0))
    probname.append("leftsource/")
    
for i in range(4):
    lengths.append(100)
    sources.append([1])
    scatter.append(scatterab[i])
    transport.append([1])
    bounds.append((1,1))
    forwards.append((0,0))
    backwards.append((0,0))
    probname.append("midsource/")
    
lengths.append([20,80])
sources.append([0,0])
scatter.append([2,0.006])
transport.append([10,0.01])
bounds.append((1,1))
forwards.append((0,0))
backwards.append((0,1))
probname.append("absorbair/")

lengths.append([50,10])
sources.append([1,0])
scatter.append([0,1.8])
transport.append([0.1,2])
bounds.append((1,0))
forwards.append((0,0))
backwards.append((0,0))
probname.append("sourcereflector/")

def simulationcomparisondeterministic(prob,Nx):    
    
    ordinate = Ordinate1DSolver(lengths[prob], transport[prob],scatter[prob],
                                sources[prob], Nx,64,bounds[prob],forwards[prob],
                                backwards[prob],probname[prob],accelerator=2)
    ordinate.solve()

    spectral = Spectral1DSolver(lengths[prob], transport[prob],scatter[prob],
                                sources[prob], Nx,2,bounds[prob],forwards[prob],
                                backwards[prob],probname[prob],accelerator=2)
    spectral.solve()
    
    return([ordinate.scalarflux,spectral.scalarflux])

def simulationcomparisonmc(prob,NP):
    
    montecarlo = MonteCarlo1DSolver(lengths[prob], transport[prob],scatter[prob],
                                sources[prob], Nx,NP,bounds[prob],forwards[prob],
                                backwards[prob],probname[prob])
    montecarlo.simulation()
    
    return([montecarlo.scalarflux_rates])

deterministicconvergence=True
if deterministicconvergence:
    for prob in range(0,10):
        
        bestord,bestspec = simulationcomparisondeterministic(prob, 2560)
        
        fig1,ax1 = plt.subplots(1)
        fig2,ax2 = plt.subplots(1)
        
        err1s = []
        err2s = []
        for n in Nx:
            
            ordinate,spectral = simulationcomparisondeterministic(prob,n)
            
            meshdiff = np.size(bestord)//np.size(ordinate)
            
            # Compute L1 norms of errors based on leftmost points in best grid
            
            err1 = np.sum(np.abs(bestord[::meshdiff]-ordinate))/np.size(ordinate)
            err2 = np.sum(np.abs(bestspec[::meshdiff]-spectral))/np.size(spectral)
            
            err1s.append(err1) 
            err2s.append(err2)
        
        ax1.plot(Nx,err1s,"ks")
        ax2.plot(Nx,err2s,"ks")
        ax1.set_xlabel("$N_x$")
        ax1.set_ylabel("$L_1$ Error")
        ax2.set_xlabel("$N_x$")
        ax2.set_ylabel("$L_1$ Error")
        
        ax1.set_xscale("log")
        ax1.set_xscale("log")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        
        fig1.suptitle("Convergence of S_N ")
        fig2.suptitle("Diffusion Convergence")
        
        fig1.savefig("ordinateconvergence"+str(prob)+"L1")
        fig2.savefig("diffusionconvergence"+str(prob)+"L1")
        
        plt.close(fig1)
        plt.close(fig2)

montecarloconvergence = True
if montecarloconvergence:
    for prob in range(0,10):
        
        bestmc = simulationcomparisonmc(prob, 4*10**5)
        
        plt.figure()
        
        err1s = []
        for n in NP:
            
            mc = simulationcomparisonmc(prob, n)
            err1 = np.sum(np.abs(bestmc-mc))/np.size(bestmc)
            err1s.append(err1)
        
        plt.plot(NP,err1s,"ks")
        plt.xlabel("$Number of Particles$")
        plt.ylabel("$L_1$ Error")
        
        plt.xscale("log")
        plt.yscale("log")
        
        plt.title("Convergence of Monte Carlo")
        plt.savefig("montecarloconvergence"+str(prob)+"L1")
        plt.close()
        
            
            
        
        
        
        
        