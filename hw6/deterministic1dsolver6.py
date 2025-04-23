#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 19:43:09 2025

@author: ehansen
"""

import numpy as np
from numpy import format_float_positional as ff
from scipy.linalg import solve_banded
from scipy.special import roots_legendre,legendre_p
import matplotlib.pyplot as plt
import os

class DeterministicSolver1D:
    
    def __init__(self,length,sigmat,sigmas0,q0,Nx,Nmu,
                 boundary,psif,psib,
                 fname="p2",accelerator=1,source="normal"):
        """
        Parameters
        ----------
        length: length per materials
        sigmas0 : double
            Isotropic Scattering Cross Section per material
        sigmat : double
            Transport Cross Section per material
        q0: external source per material

        Nx : int
            Number of mesh cells per material
        Nmu : int
            Number of angular degrees of freedom (discrete ordinates or spatial cells)
        boundary: integer tuple for sides
            0 - use reflecting boundary conditions
            1 - use incident/vacuum boundary conditions prescribed by psif,psib
            2 - Marshak condition for diffusion
        psif,psib: double tuple for forward and backward fluxes on interface 
        timer: if true, don't plot the solution
        sname: plot name for sigmas0
        fname: prefix to plot name for which problem (for sorting)
        accelerator: integer options for source iteration acceleration
            0  - no acceleration
            1 - Anderson acceleration two variable
            2 - S2 synthetic acceleration

        Returns
        -------
        None
        """
        
        # Define length of each material and total lengths
        self.length = np.array(length)
        self.totallength = np.sum(self.length)
        
        self.materialend = np.cumsum(self.length)

        self.Nx = Nx
        self.dx = self.totallength/self.Nx
        
        # Define the values of sigma_t, isotropic scattering, and isotropic source
        # in each cell
        
        self.sigmat = np.zeros(self.Nx)
        self.sigmas0 = np.zeros(self.Nx)
        self.q0 = np.zeros(self.Nx)
       
        # Define cell boundary positions
        self.surfacemesh = np.linspace(0,self.totallength,self.Nx+1)
        self.cellmesh = self.surfacemesh[:-1]
        
        a = np.nonzero(self.cellmesh < self.length[0])
        self.sigmat[a] = sigmat[0]
        self.sigmas0[a] = sigmas0[0]
        self.q0[a] = q0[0]
        
        for i in range(1,len(length)):
            a1 = self.cellmesh >= self.materialend[i-1] 
            a2 = self.cellmesh  < self.materialend[i]
            a = np.nonzero(a1*a2)
            self.sigmat[a] = sigmat[i]
            self.sigmas0[a] = sigmas0[i]
            self.q0[a] = q0[i]
                
        self.Nx = Nx
        self.Nmu = Nmu
        
        self.boundary  = boundary
        self.psif = psif
        self.psib = psib
        
        self.fname = fname
        self.it = 0
        self.accelerator = accelerator
                
        self.moment = np.zeros((self.Nx+1)*(self.Nmu),dtype="float64")
        
        self.scalarflux = np.zeros(self.Nx,dtype="float64")
        self.oldflux = self.scalarflux.copy()-99
        if (self.accelerator >= 1):
            self.oldflux2= self.oldflux.copy() - 99
        elif self.accelerator == 2:
            self.oldflux3 = self.oldflux.copy() - 99
        
        # Gauss Legendre Weights
        self.mus,self.weights = roots_legendre(self.Nmu)
                
        # Matrices         
        self.b = np.zeros((self.Nx+1)*self.Nmu,dtype="float64")
        
        # Setup for solutions
        self.matrix_setup()
        self.set_boundary()
        
        if (self.Nx+1)*(self.Nmu) <= 40:
            print(self.AB)
        
    def set_boundary(self):
        for ind in range(2):
            if self.boundary[ind] == 0:
                self.reflecting_boundaries(ind)
            elif self.boundary[ind] < 3:
                self.incidentvacuum_boundaries(ind)
            else:
                print("Only reflecting and incident/vacuum boundaries implemented")
                quit()
        
        return(None)
    
    def source_update(self):
        
        if self.it > 10 and self.accelerator == 2: # Use three Anderson variables - could lead to sharper drop
            self.oldflux3 = self.oldflux2.copy()
            self.oldflux2 = self.oldflux.copy()
            self.oldflux = self.scalarflux.copy()
        else: # storage of new variables
            self.oldflux2 = self.oldflux.copy()
            self.oldflux = self.scalarflux.copy()
        self.update_scalarflux()
        if self.it > 10 and self.accelerator == 2: # s2 synthetic acceleration
            
            self.s2syntheticacceleration()
            
        elif self.it > 10 and self.accelerator == 1: # Anderson acceleration minimization of residual difference
            c = np.dot(self.scalarflux-self.oldflux,self.oldflux-self.oldflux2)
            b = np.dot(self.oldflux-self.oldflux2,self.oldflux-self.oldflux2)
            a = np.dot(self.scalarflux-self.oldflux,self.scalarflux-self.oldflux)
            
            self.scalarflux = self.scalarflux * (b-c) + self.oldflux* (a-c)
            self.scalarflux = self.scalarflux / (a-c + b-c)
            
        self.rhs_update()
        return(None)
    
    def s2syntheticacceleration(self):
        """
        S2 synthetic acceleration technique as detailed in Adams Larsen 2002
        We set up a matrix to solve for S2 corrections to the scalar flux
        Adding these to the scalar flux may accelerate the source iteration technique
        """
        
        
        self.S2SA = np.zeros([5,2*(self.Nx+1)]) # Banded matrix 
        self.difference_rhs = np.zeros(2*(self.Nx+1))
        self.s2correction = np.zeros(2*(self.Nx+1))
        
        for i in range(0,self.Nx):
            self.S2SA[0,3+2*i] = -self.dx * self.sigmas0[i]/4
            
            self.S2SA[1,2+2*i] =  np.sqrt(1.0/3.0)+(self.sigmat[i]/2-self.sigmas0[i]/4) * self.dx 
            self.S2SA[1,3+2*i] = -np.sqrt(1.0/3.0)+(self.sigmat[i]/2-self.sigmas0[i]/4) * self.dx 
            
            self.S2SA[2,1+i*2:1+(i+1)*2] = -self.dx * self.sigmas0[i]/4 
            
            self.S2SA[3,2*i] = -np.sqrt(1.0/3.0)+(self.sigmat[i]/2-self.sigmas0[i]/4)*self.dx 
            self.S2SA[3,1+2*i] = np.sqrt(1.0/3.0)+(self.sigmat[i]/2-self.sigmas0[i]/4)*self.dx
            
            self.S2SA[4,2*i] = -self.dx * self.sigmas0[i]/4 
            
            self.difference_rhs[1+i*2:1+(i+1)*2] = self.sigmas0[i]/2 *self.dx* (self.scalarflux[i]-self.oldflux[i])
        
        # Zero boundary condition at left
        self.S2SA[2,0] = 1
        self.difference_rhs[0] = 0
        
        # Reflecting boundary condition at right        
        self.S2SA[2,-1] = 1
        self.difference_rhs[2*self.Nx+1] = 0
        
        self.s2correction = solve_banded((2,2),self.S2SA,self.difference_rhs)
        # print(self.s2correction)
        
        self.scalarflux_correction = (self.s2correction[0:-2:2]+self.s2correction[2::2])/2
        self.scalarflux_correction += (self.s2correction[1:-2:2]+self.s2correction[3::2])/2
        
        self.scalarflux += self.scalarflux_correction
    
    def reactionrates(self):
            
        self.absorb = (self.sigmat-self.sigmas0) * self.scalarflux
        self.scatterrate = self.sigmas0 * self.scalarflux
    
    def solve(self):
                
        diff = np.amax(np.abs(self.oldflux-self.scalarflux))
        self.it = 0

        while diff > 10**(-10) and self.it < 10**5:
                    
            if self.it % 1 == 0:
                print("Iteration ",self.it," Difference ",diff)
                
                if self.it > 10:
                    print("Spectral Radius",
                          np.abs(np.amax(self.scalarflux-self.oldflux)/np.amax(self.oldflux-self.oldflux2)))
            
            # Solve for the moments or discrete ordinates
            
            self.moment = solve_banded((self.nd,self.nd),self.AB,self.b)
            # if (self.Nmu*(self.Nx+1))<50:
            #     print(self.moment)
            
            # Get new source according to scalar flux
            
            self.source_update()
            
            if self.accelerator == 2 and self.it > 10:
                diff = np.amax(np.abs(self.oldflux-self.scalarflux)+np.abs(self.oldflux-self.oldflux2)+np.abs(self.oldflux2-self.oldflux3))
            elif self.accelerator == 1 and self.it > 4:
                diff = np.amax(np.abs(self.oldflux-self.scalarflux)+np.abs(self.oldflux-self.oldflux2))
            else:
                diff = np.amax(np.abs(self.oldflux-self.scalarflux))
            self.it += 1
            
            
            
        self.plotscalarflux()
        self.reactionrates()

        return(None)                                           

class Ordinate1DSolver(DeterministicSolver1D):
    
        
    def matrix_setup(self):
        """
        Notice that only the boundary conditions and length scale of the 
        problem changes.
        So we define the matrix for most of the equations independent
        of the solution routine.
        We use a banded matrix for both discrete ordinate and spectral cases; 
        only neighboring cells are assumed to be coupled.
        

        Parameters
        ----------
        Returns
        -------
        None.

        """
        
        self.AB = np.zeros([self.Nmu+1,((self.Nx+1)*self.Nmu)],dtype="float64")
        self.nd = self.Nmu//2 # Number of upper and lower diagonals needed by solve_banded
        
        # Set Nmu/2 th diagonal 
        for i in range(0,self.Nx):
            self.AB[0,(i+1)*self.Nmu:(i+2)*self.Nmu] = self.mus/self.dx + self.sigmat[i]/2
            self.AB[self.Nmu,i*self.Nmu:(i+1)*self.Nmu] = self.sigmat[i]/2 - self.mus/self.dx
            self.b[i*self.Nmu+self.Nmu//2:(1+i)*self.Nmu+self.Nmu//2] = self.q0[i]/2
           
        return(None)
    
    def matrix_setup1(self):
        
        self.AB = np.zeros([self.Nmu+1,((self.Nx+1)*self.Nmu)],dtype="float64")
        self.nd = self.Nmu//2 # Number of upper and lower diagonals needed by solve_banded
        
        # Set Nmu/2 th diagonal 
        for i in range(0,self.Nx):
            self.AB[0,(i+1)*self.Nmu:(i+2)*self.Nmu] = self.mus/self.dx + self.sigmat[i]/2
            self.AB[self.Nmu,i*self.Nmu:(i+1)*self.Nmu] = self.sigmat[i]/2 - self.mus/self.dx            
           
        return(None)
            
    def reflecting_boundaries(self,ind):
        """
        Sets reflecting or incident/vacuum boundary conditions for the matrix.
        For discrete ordinates, set opposite mu values equal at boundaries.      

        Returns
        -------
        None.

        """
        if ind == 0:
            self.b[0:self.Nmu//2] = 0
            self.AB[0,self.Nmu//2:self.Nmu] = 1
            
            di = 0
            diag = 2*di + 1
            while diag < self.Nmu:
                self.AB[diag,self.Nmu//2 - 1 - di] = -1
                di += 1
                diag = 2*di + 1
            
            # self.AB[1,2] = -1
            # self.AB[3,1] = -1
            # self.AB[5,0] = -1
            
            # di = 0
            # diag = 2*di + 1
            # while diag < self.Nmu:
            #     self.AB[diag,max(self.Nmu//2-di,0)] = -1
            #     di += 1
            #     diag = 2*di + 1
            
        else: 
            self.b[-self.Nmu//2:] = 0
            self.AB[self.Nmu,-self.Nmu:] = 1
            
            di = 0
            diag = 2*di + 1
            while diag < self.Nmu:
                self.AB[diag,-1-di] = -1
                di += 1
                diag = 2*di + 1

            # self.AB[5,-3] = -1
            # self.AB[3,-2] = -1
            # self.AB[1,-1] = -1
            
            # di = 0
            # diag = self.Nmu - 1 - 2*di
            # while diag > 0:
            #     self.AB[diag,min(-self.Nmu+1+di,-1)] = -1
            #     di += 1
            #     diag = self.Nmu -1 - 2*di
            
        
        return(None)
    
    def incidentvacuum_boundaries(self,ind):
        """
        Sets incident and vacuum boundary conditions for the matrix and RHS.
        Returns
        -------
        None.

        """
        if ind == 0:
            self.AB[0,self.Nmu//2:self.Nmu] = 1
            self.b[0:self.Nmu//2] = self.psif[ind]
        else:
            self.AB[self.Nmu,self.Nx*self.Nmu:self.Nx*self.Nmu+self.Nmu//2] = 1
            self.b[-self.Nmu//2:] = self.psib[ind]
        
        return(None)
    
    def update_scalarflux(self):
                    
        self.scalarflux = np.zeros(self.Nx)
        # Integrate with Gaussian quadrature and average
        for i in range(0,self.Nx):
            self.scalarflux[i] = (np.dot(self.moment[i*self.Nmu:(i+1)*self.Nmu],self.weights)
                                  + np.dot(self.moment[(i+1)*self.Nmu:(i+2)*self.Nmu],self.weights))/2
        
        return(None)
        
    def rhs_update(self):
        
        for i in range(0,self.Nx):
            self.b[self.Nmu//2+i*self.Nmu:self.Nmu//2+(i+1)*self.Nmu] = (self.q0[i] +
                        self.sigmas0[i]*(self.scalarflux[i]))/2
            
        return(None)
    
    def plotscalarflux(self):
        
        plt.figure()
        plt.plot(self.cellmesh,self.scalarflux,"k",label="Scalar Flux")
        #plt.plot(self.surfacemesh,self.moment[::self.Nmu],"r8",label="Most Negative Angular Flux")
        #plt.plot(self.surfacemesh,self.moment[self.Nmu-1::self.Nmu],"b8",label="Most Positive Angular Flux")
        plt.xlabel("Position (cm)")
        plt.ylabel("Flux")
        if len(self.length) == 1:
            plt.title("$S_{64}$ Scalar Flux "+str(self.Nx)+" Cells "+ "Scattering " + ff(self.sigmas0[0],2))
        else:
            plt.title("$S_{64}$ Scalar Flux "+str(self.Nx)+" Cells ")
        plt.legend()
        if self.fname[-1] == "/": 
            if not os.path.exists(self.fname+"ordinate/"):
                os.mkdir(self.fname+"ordinate/")
            plt.savefig(self.fname+"ordinate/Nx"+str(self.Nx),bbox_inches="tight")
        else: 
            plt.savefig(self.fname+"ordinate"+str(self.Nx),bbox_inches="tight")
        plt.show()
        plt.close()
        
        self.legendremoments()
        
        return(None)
        
    def legendremoments(self):
        
        self.legmoments = np.zeros([10,self.Nx+1])
        for i in range(0,10):
            legpolymu = legendre_p(i,self.mus)
            for x in range(0,self.Nx+1):
                self.legmoments[i,x] = np.sum(self.weights*legpolymu*self.moment[x*self.Nmu:(x+1)*self.Nmu])
                
        return(None)
                
    def hw6_source(self):
        
        a = np.nonzero(self.cellmesh < 40)
        self.q0[a] = 0.5*(np.exp(-2*self.cellmesh[a])+np.exp(-2*(self.dx+self.cellmesh[a])))
        a = np.nonzero(self.cellmesh >= 40)
        self.q0[a] = 0.5*(np.exp(-80)*np.exp(-10*(self.cellmesh[a]-40)) + np.exp(-80)*np.exp(-10*(self.cellmesh[a]+self.dx-40)))
        
        self.q0 *= self.sigmas0/2
        
        return(None)
    
    
            
    def homework6(self,not1st=False):
        
        if not not1st:
            self.hw6_source()
            
        self.matrix_setup()
        self.set_boundary()
        
        self.solve()
        
        return(None)
            
