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
                 fname="p2",accelerator=1):
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
            2 - Anderson acceleration three variable (not working)

        Returns
        -------
        None
        """
        
        # Define length of each material and total lengths
        self.length = np.array(length)
        self.totallength = np.sum(self.length)

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
        
        a = np.nonzero(self.surfacemesh[:-1] < self.length[0])
        self.sigmat[a] = sigmat[0]
        self.sigmas0[a] = sigmas0[0]
        self.q0[a] = q0[0]
        
        for i in range(1,len(length)):
            a = np.nonzero(self.surfacemesh[:-1] >= self.length[i-1] 
                           and self.surfacemesh[:-1]  < self.length[i])
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
        elif self.it > 4 and self.accelerator == 1: # Anderson acceleration storage of new variables
            self.oldflux2 = self.oldflux.copy()
            self.oldflux = self.scalarflux.copy()
        else:
            self.oldflux = self.scalarflux.copy()
        self.update_scalarflux()
        if self.it > 10 and self.accelerator == 2: # Anderson acceleration three variables
            # Compute dot products needed for solution
            A = np.dot(self.scalarflux-self.oldflux,self.scalarflux-self.oldflux)
            B = np.dot(self.oldflux-self.oldflux2,self.oldflux-self.oldflux2)
            C = np.dot(self.oldflux2-self.oldflux3,self.oldflux2-self.oldflux3)
            D = np.dot(self.scalarflux-self.oldflux,self.oldflux-self.oldflux2)
            E = np.dot(self.scalarflux-self.oldflux,self.oldflux2-self.oldflux3)
            F = np.dot(self.oldflux-self.oldflux2,self.oldflux2-self.oldflux3)
            
            # Set up matrix elements to minimize combination residual
            a11 = A + C - 2*E
            a22 = B + C - 2*F
            a12 = C + D - E - F
            
            b1 = C - E
            b2 = C - F
            
            # Solve 
            k = (a22 * b1 - a12 * b2)/(a11*a22 - a12**2)
            l = (a11 * b2 - a12 * b1)/(a11*a22 - a12**2)
            
            print("k",k,"l",l,"1-k-l",1-k-l)
            print("Determinant",a11*a22 - a12**2)
            
            # Combination scalar flux
            self.scalarflux = k * self.scalarflux + l * self.oldflux + (1-k-l)*self.oldflux2
            
        elif self.it > 10 and self.accelerator == 1: # Anderson acceleration minimization of residual difference
            c = np.dot(self.scalarflux-self.oldflux,self.oldflux-self.oldflux2)
            b = np.dot(self.oldflux-self.oldflux2,self.oldflux-self.oldflux2)
            a = np.dot(self.scalarflux-self.oldflux,self.scalarflux-self.oldflux)
            
            self.scalarflux = self.scalarflux * (b-c) + self.oldflux* (a-c)
            self.scalarflux = self.scalarflux / (a-c + b-c)
            
        self.rhs_update()
        return(None)
    
    def reactionrates(self):
            
        self.absorb = (self.sigmat-self.sigmas0) * self.scalarflux
        self.scatterrate = self.sigmas0 * self.scalarflux
    
    def solve(self):
                
        diff = np.amax(np.abs(self.oldflux-self.scalarflux))
        self.it = 0

        while diff > 10**(-5) and self.it < 1000000:
                    
            if self.it % 500 == 0:
                print("Iteration ",self.it," Difference ",diff)
            
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
                    
        self.scalarflux = np.zeros(self.Nx+1)
        # Integrate with Gaussian quadrature and average
        for i in range(0,self.Nx+1):
            self.scalarflux[i] = np.dot(self.moment[i*self.Nmu:(i+1)*self.Nmu],self.weights)
        
    def rhs_update(self):
        
        for i in range(0,self.Nx):
            self.b[self.Nmu//2+i*self.Nmu:self.Nmu//2+(i+1)*self.Nmu] = (self.q0[i] +
                        self.sigmas0[i]*(self.scalarflux[i]+self.scalarflux[i+1])/2)/2
    
    def plotscalarflux(self):
        
        plt.plot(self.cellmesh,self.scalarflux,"k",label="Scalar Flux")
        plt.plot(self.surfacemesh,self.moment[::self.Nmu],"r8",label="Most Negative Angular Flux")
        plt.plot(self.surfacemesh,self.moment[self.Nmu-1::self.Nmu],"b8",label="Most Positive Angular Flux")
        plt.xlabel("Position (cm)")
        plt.ylabel("Flux")
        plt.title("Scalar and Angular Fluxes Scattering ")
        plt.legend()
        plt.savefig(self.fname+"ordinate"+str(self.Nx))
        plt.close()
        
        self.legendremoments()
        
    def legendremoments(self):
        
        self.legmoments = np.zeros([10,self.Nx+1])
        for i in range(0,10):
            legpolymu = legendre_p(i,self.mus)
            for x in range(0,self.Nx+1):
                self.legmoments[i,x] = np.sum(self.weights*legpolymu*self.moment[x*self.Nmu:(x+1)*self.Nmu])
        
class Spectral1DSolver(DeterministicSolver1D):
        
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
        
        self.AB = np.zeros([self.Nmu+3,((self.Nx+1)*self.Nmu)],dtype="float64")
        self.nd = self.Nmu//2 + 1 # Number of upper and lower diagonals needed by solve_banded
        
        for i in range(0,self.Nx):
            self.AB[0,2+i*self.Nmu:2+(i+1)*self.Nmu] = np.array([0,1/self.dx])
            self.AB[1,2+i*self.Nmu:2+(i+1)*self.Nmu] = np.array([self.sigmat[i]/2,self.sigmat[i]/2])
            self.AB[2,1+i*self.Nmu:1+(i+1)*self.Nmu] = np.array([-1/self.dx,1/(3*self.dx)])
            self.AB[3,i*self.Nmu:(i+1)*self.Nmu] = np.array([self.sigmat[i]/2,self.sigmat[i]/2])
            self.AB[4,i*self.Nmu:(i+1)*self.Nmu] = np.array([-1/(3*self.dx),0])
            self.b[1+self.Nmu*i] = self.q0[i]
            
        return(None)
            
    def reflecting_boundaries(self,ind):
        """
        Sets reflecting or incident/vacuum boundary conditions for the matrix.
        For discrete ordinates, set opposite mu values equal at boundary.
        For spherical harmonics, set odd moments to zero at each boundary.
        See Exnihilo manual page 22.        

        Returns
        -------
        None.

        """
        if ind == 0:
            self.AB[self.Nmu-1,1] = 1
            self.b[0] = 0
        if ind == 1: 
            self.AB[self.Nmu,-1] = 1
            self.b[-1] = 0
                    
        return(None)
    
    def incidentvacuum_boundaries(self,ind):
        """
        Sets incident and vacuum boundary conditions for the matrix and RHS.
        Booundary 1 - Marshak boundary conditions are used for spherical harmonics.
        See Exnihilo manual page 22.
        Boundary 2 - fix currents
        

        Returns
        -------
        None.

        """
        if (self.boundary[ind]==1):
            # First row
            if ind == 0:
                self.AB[2,0] = 1
                self.AB[1,1] = 1/2
                self.b[0] = self.psif[0]
            
            if ind == 1:
                self.AB[2,-1] = 1
                self.AB[3,-2] = -1/2
                self.b[-1] = self.psib[0]
                
        elif (self.boundary[ind]==2):
            if ind == 0:
                self.AB[self.Nmu-1,1] = 1
                self.b[0] = self.psif[1]
            if ind == 1: 
                self.AB[self.Nmu,-1] = 1
                self.b[-1] = self.psib[1]
        
        return(None)

    def update_scalarflux(self):
        """
        Updates scalar flux
        Zeroth moment for spherical harmonics
        
        Returns
        -------
        None.
        """
        self.scalarflux = self.moment[::self.Nmu]
        
    def rhs_update(self):
        
        for i in range(0,self.Nx):
            self.b[1+i*self.Nmu] = self.q0[i] + self.sigmas0[i] * self.scalarflux[i]
            
    def plotscalarflux(self):
                    
        plt.plot(self.surfacemesh,self.moment[::self.Nmu],"b-",label="Scalar Flux")
        plt.plot(self.surfacemesh,self.moment[1::self.Nmu],"r:",label="Current")
        
        plt.xlabel("Position (cm)")
        plt.ylabel("Scalar Flux and Current")
        plt.title("Moments of $P_2$ Solution Scattering ")
        plt.legend()
        plt.savefig(self.fname+"spectral"+str(self.Nx))
        plt.close()
        
        self.discreteordinate()
        
    def discreteordinate(self):
        
        self.dos = np.zeros([64,self.Nx+1],dtype="float64")
        self.mus,self.weights = roots_legendre(64)
        for mu in range(0,self.Nmu):
            legpoly = legendre_p(mu,self.mus)
            self.dos += (2*mu+1)/2 * np.transpose(self.moment[mu::self.Nmu,None] @ legpoly)        