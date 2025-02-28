#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:59:40 2025

@author: ehansen
"""

import numpy as np
from scipy.linalg import solve_toeplitz as levinson
from scipy.linalg import solve_banded as banded
import matplotlib.pyplot as plt
import os

class SourceFree1D():
    
    def __init__(self,rmu,lmu,rphi0,lphi0,sigmat,sigmas=0,X=1,Nx=100):
        
        # Parameters
        self.rmu = rmu
        self.lmu = -lmu
        self.rphi0 = rphi0
        self.lphi0 = lphi0
        self.sigmat = sigmat
        # Take Nx to be the number of cells, not the number of cells
        self.Nx = Nx
        self.X = X
        self.rtau = self.sigmat * self.X/(self.Nx*self.rmu)
        self.ltau = self.sigmat * self.X/(self.Nx*self.lmu)
        
        # Computed Values
        self.xright = np.zeros(self.Nx+1,dtype="float64")
        self.xrightavg = np.zeros(self.Nx,dtype="float64")
        self.xleft = np.zeros(self.Nx+1,dtype="float64")
        self.xleftavg = np.zeros(self.Nx,dtype="float64")
        self.scalaravg = self.xrightavg + self.xleftavg
        
        # Optional Things for Problem 4
        self.sigmas = sigmas
        self.prev = np.ones(self.Nx,dtype="float64")
        self.q = self.scalaravg * 1.0/2.0 * self.sigmas

    def problem23_forwardflux_banded(self):
      
        ab = np.zeros([3,self.Nx+1],dtype="float64")
        b = np.zeros(self.Nx+1,dtype="float64")
        
        # Use an implicit difference formula - idea from Sree to test
        
        ab[0,1:] = 0.0
        ab[1,:] = 1.0+ self.rtau
        ab[1,0] = 1.0
        ab[2,:-1] = -1.0
        
        b[1:self.Nx+1] = self.q * self.X/(self.Nx * self.rmu)
        b[0] = self.rphi0
        
        
        self.xright = banded((1,1),ab,b)

        if self.sigmat < 10**(-10): # Do a naive average in cases of small tau
            # Typo with xleft
            self.xrightavg = (self.xright[1:]+self.xright[:-1])/2
        else:
            self.xrightavg = 1.0/2.0 * self.sigmas*self.scalaravg/self.sigmat-1.0/self.rtau * (self.xright[1:]-self.xright[:-1])
        
        return(None)
    
    def problem25_backwardflux(self):
        
        ab = np.zeros([3,self.Nx+1],dtype="float64")
        b = np.zeros(self.Nx+1,dtype="float64")
        
        ab[0,1:] = -(1.0+self.ltau)
        ab[1,:] = 1.0
        ab[2,:-1] = 0.0
        
        b[0:self.Nx] = - self.q * self.X/(self.Nx * self.lmu)
        b[self.Nx] = self.lphi0
        
        self.xleft = banded((1,1),ab,b)
        if self.sigmat < 10**(-10): # Do a naive average in cases of small tau
            self.xleftavg = (self.xleft[1:]+self.xleft[:-1])/2
        else:
            self.xleftavg = 1.0/2.0 * self.sigmas*self.scalaravg/self.sigmat 
            self.xleftavg += - 1.0/self.ltau * (-self.xleft[:-1]+self.xleft[1:])
            # why not switch left-right for this average? 
            # tau is negative, so we already accounted for the exchanged order
        
        return(None)

    
    def problem26_scalarflux(self):
        
        self.scalaravg = self.xrightavg + self.xleftavg
        
        return(None)
    
    def problem26_plotter(self,problemname):
        
        # Angular Flux Plots
        plt.figure()
        xplot = self.X/self.Nx * np.arange(self.Nx+1)
        rpsianalytic = self.rphi0 * np.exp(-self.sigmat/self.rmu * (xplot))
        lpsianalytic = self.lphi0 * np.exp(-self.sigmat/self.lmu * (xplot-self.X))
        plt.plot(xplot,self.xright,"s",color="mediumslateblue",
                 label="Positive mu",markersize=1)
        plt.plot(xplot,self.xleft,"s",color="tomato",
                 label="Negative mu",markersize=1)
        plt.plot(xplot,rpsianalytic,"k--",label="Source Free + Solution")
        plt.plot(xplot,lpsianalytic,"k--",label="Source Free - Solution")
        plt.xlabel("X (cm)")
        plt.ylabel(" Flux (neutron/(cm^2 s MeV mu))")
        plt.ylim(-0.2,1.5)
        if self.sigmas > 10**(-10):
            plt.title("Angular Flux in Scattering Transport")
        else:
            plt.title("Angular Flux in Source Free Transport")
        plt.legend(loc="upper right")
        if not os.path.exists("plots/"):
            os.mkdir("plots")
            
        plt.savefig("plots/"+problemname+"angular")
        plt.show()
        plt.close()
        
        # Scalar Flux Plots
        if self.sigmat < 10**(-10):
            scalaravganalytic = (rpsianalytic[1:]+rpsianalytic[:-1]
                                 +lpsianalytic[1:]+lpsianalytic[:-1])/2.0
        else:
            scalaravganalytic = -1.0/self.rtau * (rpsianalytic[1:]-rpsianalytic[:-1])
            scalaravganalytic += -1.0/self.ltau * (lpsianalytic[1:]-lpsianalytic[:-1])
        plt.plot(xplot[:-1],self.xleftavg+self.xrightavg,"s",color="olivedrab",
                 label="Computed",markersize=1)
        plt.plot(xplot[:-1],scalaravganalytic,"k:",
                 label="Source Free Solution")
        
        plt.xlabel("X (cm)")
        plt.ylabel("Scalar Flux (neutron/(cm^2 s MeV))")
        # Modified to show
        plt.ylim(-0.2,1.5)
        if self.sigmas > 10**(-10):
            plt.title("Flux in Scattering Transport")
        else:
            plt.title(" Flux in Source Free Transport")
        plt.legend(loc="upper right")
        if not os.path.exists("plots/"):
            os.mkdir("plots")
            
        plt.savefig("plots/"+problemname+"scalar")
        plt.show()
        plt.close()
        
        return(None)
    
    def problem3i(self,problemname):
        
        self.problem23_forwardflux_banded()
        self.problem25_backwardflux()
        self.problem26_scalarflux()
        self.problem26_plotter(problemname)
        
        return(None)
    
    # Didn't use so commenting out
    # def problem4i(self,problemname):
        
    #     i = 0
    #     print("Prev",np.amax(self.prev))
    #     print("Init Avg",np.amax(self.scalaravg))
    #     d = np.amax(np.abs(self.prev-self.scalaravg))
    #     while d > 10**(-10):
    #         print("Iteration",i," Difference in Scalar Fluxes ",d)
    #         self.prev = (self.scalaravg).copy()
    #         self.problem23_forwardflux_banded()
    #         self.problem25_backwardflux()
    #         plt.plot(np.arange(self.Nx+1),self.xright,"c",self.xleft,"r")
    #         plt.show()
    #         self.problem26_scalarflux()
    #         d = np.amax(np.abs(self.prev-self.scalaravg))
    #         print("Prev",np.amax(self.prev))
    #         print("Init Avg",np.amax(self.scalaravg))
    #         print("Discrepancy After Iteration",d)
    #     self.problem26_plotter(problemname)
        
    #     return(None)
    
    def problem4_given_iteration(self):
        
        self.xright[0] = self.rphi0
        self.xleft[self.Nx] = self.lphi0
        
        for i in range(self.Nx):
            
            # Dissipation
            self.xright[i+1] = self.xright[i] * np.exp(-self.rtau)
            self.xleft[self.Nx-i-1] = self.xleft[self.Nx-i] * np.exp(self.ltau)
            
            # Scattering Terms
            self.xright[i+1] += self.q[i]/self.sigmat * (1-np.exp(-self.rtau))
            self.xleft[self.Nx-i-1] += self.q[self.Nx-i-1]/self.sigmat * (1-np.exp(self.ltau))
        
        self.xleftavg = (self.xleft[:-1]+self.xleft[1:])/2.0
        
        self.xrightavg = (self.xright[1:]+self.xright[:-1])/2.0
        
        self.problem26_scalarflux()
        
        #plt.figure()
        #plt.plot(np.arange(self.Nx+1),self.xright,"c",self.xleft,"r")
        #plt.show()
        
        #plt.figure()
        #plt.plot(self.scalaravg)
        
        self.q = self.scalaravg /2.0 * self.sigmas
        
        return(None)
    
    def problem4_given(self,problemname):
        
        # Modified so self.prev remains same size
        self.prev = np.ones_like(self.scalaravg)
        i = 0
        
        d = np.amax(np.abs(self.prev-self.scalaravg))
        while d > 10**(-10) and i < 100:
            
            self.prev = self.scalaravg
            print("Iteration ",i," Discrepancy ",d)
            self.problem4_given_iteration()
            print(np.amax(self.scalaravg))
            d = np.amax(np.abs(self.prev-self.scalaravg))
            i += 1
        
        self.problem26_plotter(problemname)
            
    def problem4_matrix(self,problemname):
        
        # Modified so self.prev remains same size
        self.prev = np.ones_like(self.scalaravg)
        print(self.prev)
        i = 0
        
        d = np.amax(np.abs(self.prev-self.scalaravg))
        while d > 10**(-10) and i < 100:
            self.prev = self.scalaravg
            print("Iteration ",i," Discrepancy ",d)
            self.problem23_forwardflux_banded()
            self.problem25_backwardflux()
            self.problem26_scalarflux()
            self.q = self.scalaravg / 2.0 * self.sigmas
            i += 1 
            d = np.amax(np.abs(self.prev-self.scalaravg))
        if self.sigmat == 0:
            print(self.xrightavg)
            print(self.xleftavg)
            print(self.scalaravg)
        
        self.problem26_plotter(problemname)
        
        
def problem3():
    problem = SourceFree1D(1,1,1,0,1,Nx=10000)
    problem.problem3i("3a")

    problem = SourceFree1D(1,1,0,1,1)
    problem.problem3i("3b")

    problem = SourceFree1D(1,1,1,1,1)
    problem.problem3i("3c")

    problem = SourceFree1D(0.25,0.25,1,1,1)
    problem.problem3i("3d")

    problem = SourceFree1D(0.25,0.25,1,1,4)
    problem.problem3i("3e")

    problem = SourceFree1D(1,1,1,0,0.1)
    problem.problem3i("3f")

    problem = SourceFree1D(1,1,0,1,0)
    problem.problem3i("3g")
        
def problem4():
    
    problem = SourceFree1D(1,1,1,0,1,sigmas=0.1,Nx=10000)
    problem.problem4_matrix("4a")
    
    problem = SourceFree1D(1,1,0,1,1,sigmas=0.1)
    problem.problem4_matrix("4b")

    problem = SourceFree1D(1,1,1,1,1,sigmas=0.1)
    problem.problem4_matrix("4c")

    problem = SourceFree1D(0.25,0.25,1,1,1,sigmas=0.1)
    problem.problem4_matrix("4d")

    problem = SourceFree1D(0.25,0.25,1,1,4,sigmas=0.1)
    problem.problem4_matrix("4e")

    problem = SourceFree1D(1,1,1,0,0.1,sigmas=0.1)
    problem.problem4_matrix("4f")

    problem = SourceFree1D(1,1,0,1,0,sigmas=0.1)
    problem.problem4_matrix("4g")

problem3()
problem4()


        