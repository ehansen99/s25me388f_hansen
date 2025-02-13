#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:44:06 2025

@author: ehansen
"""

import numpy as np
from numpy import format_float_positional as ff
from scipy.linalg import solve_banded
from scipy.special import roots_legendre
import matplotlib.pyplot as plt

class Spectral1DSolver:
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
            Number of angular spherical harmonics
        boundary: integer
            0 - use reflecting boundary conditions
            1 - use incident/vacuum boundary conditions
        timer : logical
            If true, doesn't plot solution at the end of calculation

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
        
        if self.Nmu != 2:
            print("Only two degrees of freedom supported at this time")
            quit()
        
        self.length = 100
        self.xleft = np.arange(0,self.length+self.length/self.Nx,self.length/self.Nx)
        self.dx = self.length/self.Nx
                
        # Will solve for Nphi ordinates or angles at each node
        # Matrices will be constructed as blocks at each position
        
        self.moment = np.zeros((self.Nx+1)*(self.Nmu),dtype="float64")
        
        self.scalarflux = np.zeros(self.Nx,dtype="float64")
        self.oldflux = self.scalarflux.copy()-99
        
        # Gauss Legendre Weights
        self.mus,self.weights = roots_legendre(self.Nmu)
                
        # Matrices 
        self.AB = np.zeros([self.Nmu+3,((self.Nx+1)*self.Nmu)],dtype="float64")
        
        self.b = np.zeros((self.Nx+1)*self.Nmu,dtype="float64")
        
        # Setup for solutions
        self.matrix_setup()
        self.set_boundary()
        
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

        for i in range(0,self.Nx):
            self.AB[0,2+i*self.Nmu:2+(i+1)*self.Nmu] = np.array([0,1/self.dx])
            self.AB[1,2+i*self.Nmu:2+(i+1)*self.Nmu] = np.array([self.sigmat/2,self.sigmat/2])
            self.AB[2,1+i*self.Nmu:1+(i+1)*self.Nmu] = np.array([-1/self.dx,1/(3*self.dx)])
            self.AB[3,i*self.Nmu:(i+1)*self.Nmu] = np.array([self.sigmat/2,self.sigmat/2])
            self.AB[4,i*self.Nmu:(i+1)*self.Nmu] = np.array([-1/(3*self.dx),0])
            self.b[1+self.Nmu*i] = self.q0
            
        return(None)
            
    def reflecting_boundaries(self):
        """
        Sets reflecting or incident/vacuum boundary conditions for the matrix.
        For discrete ordinates, set opposite mu values equal at boundary.
        For spherical harmonics, set odd moments to zero at each boundary.
        See Exnihilo manual page 22.        

        Returns
        -------
        None.

        """
        self.AB[self.Nmu,-1] = 1
        self.AB[self.Nmu-1,1] = 1
        
        self.b[0] = 0
        self.b[-1] = 0
        
        return(None)
    
    def incidentvacuum_boundaries(self):
        """
        Sets incident and vacuum boundary conditions for the matrix and RHS.
        Booundary 1 - Marshak boundary conditions are used for spherical harmonics.
        See Exnihilo manual page 22.
        Boundary 2 - vacuum boundary conditions as referenced from diffusion slides,
        

        Returns
        -------
        None.

        """
        if (self.boundary==1):
            # First row
            self.AB[2,0] = 1
            self.AB[1,1] = 1/2
            self.b[0] = 1
        
            # Last row of matrix
            self.AB[2,-1] = 1
            self.AB[3,-2] = -1/2 # Switch sign because the flux is opposite
        
            self.b[-1] = 0
        elif (self.boundary==2):
            # First row - set scalar flux to one to match positive
            # I'm not sure about this because we can't set both the scalar flux and 
            # spatial variation, however 
            self.AB[2,0] = 1
            self.AB[1,1] = 0
            self.b[0] = 1
        
            # Last row of matrix 0- extrapolation distance
            self.AB[2,-1] = 0.7104/self.sigmat
            self.AB[3,-2] = 1
        
            self.b[-1] = 0
            
            
        
        return(None)
        
    
    def set_boundary(self):
        if self.boundary == 0:
            self.reflecting_boundaries()
        elif self.boundary >= 1:
            self.incidentvacuum_boundaries()
        else:
            print("Only reflecting and incident/vacuum boundaries implemented")
            quit()
        
        return(None)
        
    def update_scalarflux(self):
        """
        Updates scalar flux - 
        Uses Gaussian quadrature for discrete ordinates, 
        Zeroth moment for spherical harmonics
        
        Returns
        -------
        None.

        """
        self.scalarflux = (self.moment[:-2:self.Nmu]+self.moment[self.Nmu::self.Nmu])/2
        
    def source_update(self):
        
        self.update_scalarflux()
        for i in range(0,self.Nx):
            self.b[1+i*self.Nmu] = self.q0 + self.sigmas0 * self.scalarflux[i]
            
    def plotscalarflux(self):
                    
        plt.plot(self.xleft,self.moment[::self.Nmu],"b-",label="Scalar Flux")
        plt.plot(self.xleft,self.moment[1::self.Nmu],"r:",label="Current")
        
        plt.xlabel("Position (cm)")
        plt.ylabel("Scalar Flux and Current")
        plt.title("Moments of $P_2$ Solution Scattering "+ff(self.sigmas0,5))
        plt.legend()
        plt.savefig(self.fname+"specfluxes"+str(self.boundary)+str(int(self.q0))+str(self.sname)+str(self.Nx))
        plt.close()
        
    def solve(self):
                
        diff = np.amax(np.abs(self.oldflux-self.scalarflux))
        it = 0
        if (self.Nmu*self.Nx) < 50:
            print(self.AB)
            
        while diff > 10**(-10) and it < 1000000:
            
            self.oldflux = self.scalarflux.copy()
            if it % 1000 == 0: 
                print("Iteration ",it," Difference ",diff)
            
            # Solve for the moments or discrete ordinates
            self.moment = solve_banded((2,2),self.AB,self.b)
            
            # Get new source according to scalar flux
            self.source_update()
            
            diff = np.amax(np.abs(self.oldflux-self.scalarflux))
            it += 1
        if self.timer == False:
            self.plotscalarflux()