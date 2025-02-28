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

sigmass = [0,0.0001,0.001,0.01,0.1,0.5,0.9,0.99,0.999,0.9999,1]
Nxs = [10,40,160,640,2560,10240]
mus = [2,4,8,16,32,64]
q0s = [0,1]

# Infinite Homogeneous Medium

part1a = False

if part1a:
    ordinate = Ordinate1DSolver(100,1,0.9999,1,Nx=160,
                 boundary=(0,0),psif=(0,0),psib=(0,0),Nmu=64,accelerator=1)
    ordinate.solve()
    fmts = ["k1","k2","k3","k4","k^","kv","k>","k<"]
    for i in range(1,8):
        plt.plot(ordinate.xleft,np.abs(ordinate.legmoments[i,:]),fmts[i-1],
                 markersize=5,label=i)
    plt.xlabel("x (cm)")
    plt.ylabel("Moment")
    plt.ylim(10**(-20),10**2)
    plt.yscale("log")
    plt.title("Infinite Homogeneous Medium Angular Flux Moments")
    plt.legend(loc="upper right")
    plt.savefig("infinite/moments")
    plt.close()
    
    spectral = Spectral1DSolver(100,1,0.9999,1,Nx=160,
             boundary=(0,0),psif=(0,0),psib=(0,0),Nmu=2,accelerator=1)
    spectral.solve()
    
    plt.plot(ordinate.xleft[:-1],ordinate.scalarflux,"b-",
             label="Discrete Ordinate")
    plt.plot(spectral.xleft[:-1],spectral.scalarflux,"r:",
             label="Diffusion")
    plt.legend(loc="upper right")
    plt.xlabel("x (cm)")
    plt.ylabel("Scalar Flux")
    plt.title("Infinite Homogeneous Medium Scattering 0.9999")
    plt.savefig("infinite/fluxcomp")
    plt.close()
    

# Source Free Transport

# Get ordinate solution

def plotmoment(ordinate,spectral,name,i):
    plt.plot(ordinate.xleft,ordinate.legmoments[0,:],label="Discrete Ordinate",
             color="mediumturquoise",marker="s",markersize=1)
    plt.plot(spectral.xleft[:-1],spectral.scalarflux,label="Diffusion",
             color="tomato",marker="s",markersize=1)
    plt.xlabel("x (cm)")
    plt.ylabel("Scalar Flux")
    plt.title( "Scattering  = "+ff(s,5))
    plt.legend()
    plt.savefig("fluxcomp/"+name+str(i))
    plt.close()
    
def plotmomentratio(ordinate,name,i):
    plt.plot(ordinate.xleft,ordinate.legmoments[1,:]/ordinate.legmoments[0,:],
             color="mediumturquoise",marker="s",label="Current/Flux",markersize=1)
    plt.plot(ordinate.xleft,ordinate.legmoments[2,:]/ordinate.legmoments[0,:],
             color="tomato",marker="s",label="2nd Moment/Flux",markersize=1)
    plt.xlabel("x (cm)")
    plt.ylabel("Moment Ratio")
    plt.title(" Scattering = "+ff(s,5))
    plt.legend()
    plt.savefig("ordmoment/"+name+str(i))
    plt.close()
    
def plotreactionrate(ordinate,spectral,name,i):
    plt.plot(ordinate.xleft[:-1],ordinate.absorb,"mediumturquoise",
             marker="s",label="Discrete Ordinate",markersize=1)
    plt.plot(ordinate.xleft[:-1],spectral.absorb,"tomato",
             marker="s",label="Diffusion",markersize=1)
    plt.xlabel("x (cm)")
    plt.ylabel("Absorption Rate")
    plt.title("Scattering = "+ff(s,5))
    plt.legend()
    plt.savefig("absorb/"+name+str(i))
    plt.close()
    
part1bc = False
if part1bc:
    for q in range(0,2):
        
        if q == 0:
            name = "sourcefree"
        else:
            name = "vacuumboundary"
                
        for i,s in enumerate(sigmass[0:10]):
            print("Scatter",s)
        # Get current boundaries
        
            ordinate = Ordinate1DSolver(100,1,s,q,Nx=160,
                                        boundary=(1,1),psif=(1-q,0),psib=(0,0),Nmu=64,accelerator=1)
            ordinate.solve()

            fluxf = ordinate.legmoments[1,0]
            fluxb = ordinate.legmoments[1,-1]

            spectral = Spectral1DSolver(100, 1, s, q,Nx=160,
                                        boundary=(2,2),psif=(0,fluxf),psib=(0,fluxb),Nmu=2,accelerator=1)
            spectral.solve()
            
            plotmoment(ordinate,spectral,name,i)
        
            plotmomentratio(ordinate,name,i)
    
            plotreactionrate(ordinate, spectral, name, i)



