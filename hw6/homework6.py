#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 22:50:23 2025

@author: ehansen
"""

from deterministic1dsolver6 import Ordinate1DSolver
from montecarlo1dsolver6 import MonteCarlo1DSolver
import numpy as np
import matplotlib.pyplot as plt


discreteord1st = True
Nx = 10000
Nmu = 256
NP = 10000

Nx2 = 4000
NP2 = 40000

if discreteord1st:
    #ordinate = Ordinate1DSolver([1],[1],[0.999],[1],100,32,(0,0),(0,0),(0,0),fname="test",accelerator=2)
    #ordinate.solve()
    
    
    # ordinatecoll = Ordinate1DSolver([35,5,10],[2,2,10],[1.99,1.8,2],[0,0,0], Nx,Nmu, (1,1), (0,0), (0,0),
    #                    fname="ord1stcoll"+str(Nx),accelerator=2)
    # ordinatecoll.homework6()
    
    ordinate = Ordinate1DSolver([35,5,10],[2,2,10],[1.8,1.8,2],[0,0,0], Nx,Nmu, (1,1), (1,0), (0,0),
                       fname="ordguessway"+str(Nx),accelerator=0)
    ordinate.solve()
    
    # ordinatecoll2 = Ordinate1DSolver([35,5,10],[2,2,10],[1.99,1.8,2],[0,0,0], Nx2,Nmu, (1,1), (0,0), (0,0),
    #                    fname="ord1stcoll"+str(Nx),accelerator=2)
    # ordinatecoll2.homework6()
    
montecarlo1st = False
if montecarlo1st:
    mccoll = MonteCarlo1DSolver([35,5,10],[2,2,10],[1.99,1.8,2],[0,0,0],Nx,NP,(1,1),(0,0),(0,0),
                            "mc1stcoll"+str(Nx))
    mccoll.homework6()
    
    mc = MonteCarlo1DSolver([35,5,10],[2,2,10],[1.99,1.8,2],[0,0,0],Nx,NP,(1,1),(1,0),(0,0),
                            "mcguessway"+str(Nx))
    mc.simulation(mu=1)
    
    mccoll2 = MonteCarlo1DSolver([35,5,10],[2,2,10],[1.99,1.8,2],[0,0,0],Nx2,NP2,(1,1),(0,0),(0,0),
                            "mc1stcoll"+str(Nx))
    mccoll2.homework6()
    

# Get impact of each cell on the detector measurement
# Average distance to each cell



def analyticflux(x):
    if x < 40:
        return(np.exp(-2*x))
    else:
        return(np.exp(-80-10*(x-40)))
    
def reverseanalyticflux(x):
    return(analyticflux(50)/analyticflux(x))

# Code from this based on W3Schools "Create your own ufunc"
reverseanalyticflux = np.frompyfunc(reverseanalyticflux,1,1)

    
fig1,ax1 = plt.subplots(1)
fig2,ax2 = plt.subplots(1)
fig3,ax3 = plt.subplots(1)

colors = ["mediumturquoise","tomato","dodgerblue","fuchsia","navy","firebrick"]
linestyle = ["-","-",":",":","--","--"]
label = ["Collision $S_{64}$ Nx 1000","Collision MC Nx 1000 NP 10000",
         "No Collision $S_{64}$ Nx 1000",
         "No Collision MC Nx 1000 NP 10000",
         "Collision $S_{64}$ Nx 4000",
         "Collision MC Nx 4000 NP 40000"]

detectorfluxes = []

for i,sim in enumerate([ordinatecoll,mccoll,ordinate,mc,ordinatecoll2,mccoll2]):
    
    
    weight = reverseanalyticflux((sim.cellmesh+sim.surfacemesh[1:])/2)
    ax2.plot(sim.cellmesh,weight,color=colors[i],linestyle=linestyle[i],label=label[i])
    
    ax1.plot((sim.cellmesh+sim.surfacemesh[1:])/2,sim.scalarflux,color=colors[i],
             linestyle=linestyle[i],label=label[i])
    ax3.plot((sim.cellmesh+sim.surfacemesh[1:])/2,sim.scalarflux,color=colors[i],
             linestyle=linestyle[i],label=label[i])
    
    # Get Uncollided source
    uncollided = np.zeros_like(sim.cellmesh)
    a = np.nonzero(sim.cellmesh < 40)
    uncollided[a] = 0.5*(np.exp(-2*sim.cellmesh[a])+np.exp(-2*(sim.dx+sim.cellmesh[a])))
    a = np.nonzero(sim.cellmesh >= 40)
    uncollided[a] = 0.5*(np.exp(-80)*np.exp(-10*(sim.cellmesh[a]-40)) + np.exp(-80)*np.exp(-10*(sim.cellmesh[a]+sim.dx-40)))
    
    uncollided *= sim.sigmas0/2
    
    collided = sim.scalarflux * sim.sigmas0/2
    
    if i >= 4:
        detectorflux = np.sum(weight*(collided+uncollided)*sim.dx)
        if i == 4:
           print("Ordinate 2 w/ First Last Collision Source ",detectorflux)
        if i == 5:
            print("MC 2 w/ First Last Collision Source ",detectorflux)
    
    if i >= 2: 
        detectorflux = np.sum(weight*(collided)*sim.dx)
        if i == 2:
           print("Ordinate w/o First Last Collision Source ",detectorflux)
        if i == 3:
            print("MC w/o First Last Collision Source ",detectorflux)
    else:
        detectorflux = np.sum(weight*(collided+uncollided)*sim.dx)
        if i == 0:
           print("Ordinate w/ First Last Collision Source ",detectorflux)
        if i == 1:
            print("MC w/ First Last Collision Source ",detectorflux)
            
    detectorfluxes.append(detectorflux)
    
ax1.set_ylabel("Scalar Flux (1/($cm^2$ $s$ MeV) )")
ax1.set_xlabel("Length (cm)")
ax1.legend(loc="upper right")
ax1.set_yscale("log")
ax2.set_ylabel("Weight")
ax2.set_xlabel("Length (cm)")
ax2.legend(loc="upper right")
ax3.set_ylabel("Scalar Flux (1/($cm^2$ $s$ MeV) )")
ax3.set_xlabel("Length (cm)")
ax3.legend(loc="upper right")


fig1.savefig("fullhomework6_log")
fig2.savefig("weights")
fig3.savefig("fullhomework6_norm")

plt.show()
plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    