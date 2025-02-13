#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:44:23 2025

@author: ehansen
"""

from ordinate1dsolver import Ordinate1DSolver
from spectral1dsolver import Spectral1DSolver
import time  
import matplotlib.pyplot as plt
import numpy as np
from numpy import format_float_positional as ff

sigmass = [0,0.0001,0.001,0.01,0.1,0.5,0.9,0.99,0.999,0.9999,1]
Nxs = [10,40,160,640,2560,10240]
mus = [2,4,8,16,32,64]
q0s = [0,1]

spectral3 = Spectral1DSolver(0.5,boundary=1,q0=1,Nx=40,Nmu=2,timer=False,sname=5,fname="3p")
spectral3.solve()

bestcase = True
if bestcase:
    N = 2560
    M = 16
    for i,s in enumerate(sigmass):
        ordinate3 = Ordinate1DSolver(s,boundary=1,q0=1,Nx=N,Nmu=M,timer=False,sname=i,fname="3p")
        ordinate3.solve()
        np.savez("bestord3"+str(i),moment=ordinate3.moment,flux=ordinate3.scalarflux,mu=ordinate3.mus)
        
        print("\n\n Best Ordinate 3 Done \n \n ")
        ordinate2 = Ordinate1DSolver(s,boundary=1,q0=0,Nx=N,Nmu=M,timer=False,sname=i,fname="2p")
        ordinate2.solve()
        np.savez("bestord2"+str(i),moment=ordinate2.moment,flux=ordinate2.scalarflux,mu=ordinate2.mus)
        print("\n \n Best Ordinate 2 Done\n \n ")
        spectral3 = Spectral1DSolver(s,boundary=1,q0=1,Nx=N,Nmu=2,timer=False,sname=i,fname="3p")
        spectral3.solve()
        np.savez("bestspec3"+str(i),moment=ordinate3.moment,flux=ordinate3.scalarflux)
        print("\n \n Best Spectral 3 Done\n \n ")
        spectral2 = Spectral1DSolver(s,boundary=1,q0=0,Nx=N,Nmu=2,timer=False,sname=i,fname="2p")
        spectral2.solve()
        np.savez("bestspec2"+str(i),moment=ordinate2.moment,flux=ordinate2.scalarflux)
        print("\n \n Best Spectral 2 Done\n \n ")

# Timing of the Problem
timenmu = True
if timenmu:
    time2 = []
    time3 = []
    for mu in mus:
        a = time.perf_counter() #Usage reference from Google AI
        solver = Ordinate1DSolver(0.9,boundary=1,q0=0,Nx=40,Nmu=mu,timer=True)
        solver.solve()
        b = time.perf_counter()  
        time2.append(b-a)
        a = time.perf_counter()
        solver = Ordinate1DSolver(0.9,boundary=1,q0=1,Nx=40,Nmu=mu,timer=True)
        solver.solve()
        b = time.perf_counter()  
        time3.append(b-a)
    plt.plot(mus,time2,"bs",label="Problem 2")
    plt.plot(mus,time3,"rs",label="Problem 3")
    plt.xlabel("Number of Angular Ordinates")
    plt.ylabel("Time (s)")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Time for Problem with Angular Ordinates")
    plt.legend()
    plt.savefig("timemuproblems")
    plt.close()
    
# Timing of the 1D homogeneous
timeord = True
if timeord:
    timeo = []
    times = []
    for s in sigmass:
        a = time.perf_counter() #Usage reference from Google AI
        solver = Ordinate1DSolver(s,boundary=0,q0=1,Nx=10,Nmu=2,timer=True)
        solver.solve()
        b = time.perf_counter() 
        timeo.append(b-a)
        a = time.perf_counter() #Usage reference from Google AI
        solver = Spectral1DSolver(s,boundary=0,q0=1,Nx=10,Nmu=2,timer=True)
        solver.solve()
        b = time.perf_counter()
        times.append(b-a)
    plt.plot(sigmass,times,"bs",label="Spherical Harmonic")
    plt.plot(sigmass,timeo,"rs",label="Ordinates")
    plt.xlabel("Scattering Cross Section")
    plt.ylabel("Time (s)")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Time for Ordinate Iterative Scattering Solution")
    plt.legend()
    plt.savefig("timesinfinite")
    plt.close()

crosssections = True
if crosssections:
    
    for i,s in enumerate(sigmass):
        
        
        file1 = np.load("bestord3"+str(i)+".npz")    
        file2 = np.load("bestord2"+str(i)+".npz")
        
        mom1 = file1["moment"]
        mom2 = file2["moment"]
        mu = file1["mu"]
        Nmu = 16
        Nx = 2560
        
        left1 = mom1[:Nmu]
        center1 = mom1[Nx//2*Nmu:(Nx//2+1)*Nmu]
        right1 = mom1[-Nmu:]
        
        left2 = mom2[:Nmu]
        center2 = mom2[Nx//2*Nmu:(Nx//2+1)*Nmu]
        right2 = mom2[-Nmu:]
        
        plt.plot(mu,left1,"rs",markersize=0.5,label="Left")
        plt.plot(mu,right1,"bs",markersize=0.5,label="Right")
        plt.plot(mu,center1,"ks",markersize=0.5,label="Center")
        
        plt.xlabel("$\mu$")
        plt.ylabel("Angular Flux")
        plt.legend()
        plt.title("Angular Flux Finite Source "+ff(s,5))
        plt.savefig("crosssectionordp3"+str(i))
        plt.close()
        
        plt.plot(mu,left2,"rs",markersize=0.5,label="Left")
        plt.plot(mu,right2,"bs",markersize=0.5,label="Right")
        plt.plot(mu,center2,"ks",markersize=0.5,label="Center")
        
        plt.xlabel("$\mu$")
        plt.ylabel("Angular Flux")
        plt.legend()
        plt.title("Angular Flux Source Free "+ff(s,5))
        plt.savefig("crosssectionordp2"+str(i))
        plt.close()
        
        file1 = np.load("bestspec3"+str(i)+".npz")    
        file2 = np.load("bestspec2"+str(i)+".npz")
        
        mom1 = file1["moment"]
        mom2 = file2["moment"]
        
        left1 = mom1[:Nmu]
        center1 = mom1[Nx//2*Nmu:(Nx//2+1)*Nmu]
        right1 = mom1[-Nmu:]
        
        left2 = mom2[:Nmu]
        center2 = mom2[Nx//2*Nmu:(Nx//2+1)*Nmu]
        right2 = mom2[-Nmu:]
        
        plt.plot(mu,left1[0]+mu*left1[1],"rs",markersize=0.5,label="Left")
        plt.plot(mu,right1[0]+mu*right1[1],"bs",markersize=0.5,label="Right")
        plt.plot(mu,center1[0]+mu*center1[1],"ks",markersize=0.5,label="Center")
        
        plt.xlabel("$\mu$")
        plt.ylabel("Angular Flux")
        plt.legend()
        plt.title("Angular Flux Finite Source "+ff(s,5))
        plt.savefig("crosssectionspecp3"+str(i))
        plt.close()
        
        plt.plot(mu,left2[0]+mu*left2[1],"rs",markersize=0.5,label="Left")
        plt.plot(mu,right2[0]+mu*right2[1],"bs",markersize=0.5,label="Right")
        plt.plot(mu,center2[0]+mu*center2[1],"ks",markersize=0.5,label="Center")
        
        plt.xlabel("$\mu$")
        plt.ylabel("Angular Flux")
        plt.legend()
        plt.title("Angular Flux Source Free "+ff(s,5))
        plt.savefig("crosssectionspecp2"+str(i))
        plt.close()
        
convergence = True
if convergence:
    
    file1 = np.load("bestord35.npz")    
    file2 = np.load("bestord25.npz")
    
    errs3 = [] 
    errs2 = []
    
    for M in mus[:3]:
        ordinate3 = Ordinate1DSolver(0.5,boundary=1,q0=1,Nx=2560,Nmu=M,timer=True,sname=5,fname="3p")
        ordinate3.solve()
        errs3.append(np.sum(np.abs(ordinate3.scalarflux-file1["flux"])))
                
        ordinate2 = Ordinate1DSolver(0.5,boundary=1,q0=0,Nx=2560,Nmu=M,timer=True,sname=5,fname="2p")
        ordinate2.solve()
        errs2.append(np.sum(np.abs(ordinate2.scalarflux-file2["flux"])))
        
    plt.plot(mus[:3],errs3,"r:",label="Problem 3")
    plt.plot(mus[:3],errs2,"b:",label="Problem 2")
    plt.xlabel("$\mu$")
    plt.ylabel("Error in Scalar Flux")
    plt.title("Convergence of Ordinate Solution with Angular Resolution")
    plt.savefig("convergence")
    plt.close()
    
    
    
    
    

