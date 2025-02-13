#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 19:43:09 2025

@author: ehansen
"""

import numpy as np
from numpy import format_float_positional as ff
from scipy.linalg import solve_banded
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

class Ordinate1DSolver:
    def __init__(self,sigmas0,sigmat=1,q0=1,Nx=10,Nmu=2,boundary=0,timer=False,sname=0,fname="p2"):
        """

        Parameters
        ----------
        sigmas0 : double
            Isotropic Scattering Cross Section.
        sigmat : double
            Transport Cross Section.
        q0: external source
        Nx : int
            Number of mesh cells.
        Nmu : int
            Number of angular  discrete ordinates
        boundary: integer
            0 - use reflecting boundary conditions
            1 - use incident/vacuum boundary conditions
        timer: if true, don't plot the solution

        Returns
        -------
        None

        """        
        # Define needed parameters
        self.sigmas0 = sigmas0
        self.sigmat = sigmat
        self.q0 = q0
        self.Nx = Nx
        self.Nmu = Nmu
        self.boundary  = boundary
        self.timer = timer
        self.sname = sname
        self.fname = fname
        
        if self.Nmu % 2 == 1:
            print("Must use an even number of ordinates")
            quit()
        
        self.length = 100
        self.xleft = np.arange(0,self.length+self.length/self.Nx,self.length/self.Nx)
        self.dx = self.length/self.Nx
                
        # Will solve for Nphi ordinates or angles at each node
        # Matrices will be constructed as blocks at each position
        
        self.moment = np.zeros((self.Nx+1)*(self.Nmu),dtype="float64")
        
        self.scalarflux = np.zeros(self.Nx,dtype="float64")
        self.oldflux = -99+self.scalarflux.copy()
        self.oldflux2 = -31+self.scalarflux.copy()
        
        self.source = np.zeros((self.Nx+1)*(self.Nmu),dtype="float64")
        
        # Gauss Legendre Weights
        self.mus,self.weights = roots_legendre(self.Nmu)
        
        # Matrices 
        self.AB = np.zeros([self.Nmu+1,((self.Nx+1)*self.Nmu)],dtype="float64")
        
        self.b = np.zeros((self.Nx+1)*self.Nmu,dtype="float64")
        
        # Setup for solutions
        self.matrix_setup()
        
        self.set_boundary()
        
        if (self.Nx+1)*(self.Nmu) <= 40:
            print(self.AB)
        
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
                
        # We use a source iteration method
        # so don't use weights for scalar flux here
        
        # Set Nmu/2 th diagonal 
        for i in range(0,self.Nx):
            self.AB[0,(i+1)*self.Nmu:(i+2)*self.Nmu] = self.mus/self.dx + self.sigmat/2
            self.AB[self.Nmu,i*self.Nmu:(i+1)*self.Nmu] = self.sigmat/2 - self.mus/self.dx
            
        self.b[self.Nmu//2:-self.Nmu//2] = self.q0/2
           
        return(None)
            
    def reflecting_boundaries(self):
        """
        Sets reflecting or incident/vacuum boundary conditions for the matrix.
        For discrete ordinates, set opposite mu values equal at boundaries.      

        Returns
        -------
        None.

        """
        self.incidentvacuum_boundaries()
        if self.Nmu != 2:
            print("Reflecting boundaries not yet implemented, reverting to incident")
        else:
            self.AB[1,0] = -1
            self.AB[1,-1] = -1
            self.b[0:self.Nmu//2] = 0
            self.b[-self.Nmu//2:] = 0
            
        
        return(None)
    
    def incidentvacuum_boundaries(self):
        """
        Sets incident and vacuum boundary conditions for the matrix and RHS.
        
        
        
        Marchuk boundary conditions are used for spherical harmonics.
        See Exnihilo manual page 22 and Mathematica notebook for details.

        Returns
        -------
        None.

        """
        self.AB[0,self.Nmu//2:self.Nmu] = 1
        self.AB[self.Nmu,self.Nx*self.Nmu:self.Nx*self.Nmu+self.Nmu//2] = 1
        self.b[0:self.Nmu//2] = 1
        self.b[-self.Nmu//2:] = 0
        
        return(None)
        
    
    def set_boundary(self):
        if self.boundary == 0:
            self.reflecting_boundaries()
        elif self.boundary == 1:
            self.incidentvacuum_boundaries()
        else:
            print("Only reflecting and incident/vacuum boundaries implemented")
            quit()
        
        return(None)
    
    def update_scalarflux(self):
        
        self.oldflux = self.scalarflux.copy()
        vector = np.zeros(self.Nx+1)
        for i in range(0,self.Nx+1):
            vector[i] = np.dot(self.moment[i*self.Nmu:(i+1)*self.Nmu],self.weights)
        self.scalarflux= (vector[:-1]+vector[1:])/2
        
    def source_update(self):
        
        self.update_scalarflux()
        for i in range(0,self.Nx):
            self.b[self.Nmu//2+i*self.Nmu:self.Nmu//2+(i+1)*self.Nmu] = (self.q0 + self.sigmas0*self.scalarflux[i])/2
    
    def plotscalarflux(self):
        
        plt.plot(self.xleft[:-1],self.scalarflux,"k",label="Scalar Flux")
        plt.plot(self.xleft,self.moment[::self.Nmu],"r8",label="Most Negative Angular Flux")
        plt.plot(self.xleft,self.moment[self.Nmu-1::self.Nmu],"b8",label="Most Positive Angular Flux")
        plt.xlabel("Position (cm)")
        plt.ylabel("Flux")
        plt.title("Scalar and Angular Fluxes Scattering "+ff(self.sigmas0,5))
        plt.legend()
        plt.savefig(self.fname+"ordfluxes"+str(self.boundary)+str(int(self.q0))+str(self.sname)+str(self.Nx))
        plt.close()
        
    def solve(self):
                
        diff = np.amax(np.abs(self.oldflux-self.scalarflux))
        self.it = 0

        while diff > 10**(-10) and self.it < 1000000:
            
            if self.it % 1000 == 0:
                print("Iteration ",self.it," Difference ",diff)
            
            # Solve for the moments or discrete ordinates
            self.oldflux = self.scalarflux.copy()
            self.moment = solve_banded((self.Nmu//2,self.Nmu//2),self.AB,self.b)
            # if (self.Nmu*(self.Nx+1))<50:
            #     print(self.moment)
            
            # Get new source according to scalar flux
            
            self.source_update()
            
            diff = np.amax(np.abs(self.oldflux-self.scalarflux))
            self.it += 1
        if not self.timer:
            self.plotscalarflux()
            
