#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 18:09:03 2025

@author: ehansen
"""

import numpy as np
from numpy.random import rand as rand
import matplotlib.pyplot as plt


class MonteCarlo1DSolver:
    def __init__(self,length,sigmat,sigmas,q0,Nx,NP,
                 boundary,psif,psib):
        """        
        Parameters
        ----------
        length : double
            List of problem lengths .
        sigmat : double
            Transport cross section list.
        sigmas : double
            Scattering cross section list.
        q0 : double
            List of sources in materials
        Nx : 
            Total number of cells
        NP : integer
            number of particles to test
        boundary : integer tuple for side boundaries
            0 : reflecting - anything generated at edges comes back in, also no source
            1 : incident/vacuum flux given by psif,psib
        psif,psib: prescribed forward and backward fluxes for boundary conditions
        Returns
        -------
        None.

        """
                
        self.length = np.array(length)
        self.q0 = np.array(q0)
        
        self.Nx = Nx
        self.NP = NP
        self.totallength = np.sum(self.length)
        self.dx = self.totallength/self.Nx
        
        self.smesh = np.linspace(0,self.totallength,self.Nx+1)
        self.xmesh = self.smesh[:-1]
        
        a = np.nonzero(self.xmesh < self.length[0])
        
        self.sigmat = np.zeros(self.Nx)
        self.sigmas = np.zeros(self.Nx)

        self.sigmat[a] = sigmat[0]
        self.sigmas[a] = sigmas[0]
        
        
        self.boundary = boundary
        self.psif = psif
        self.psib = psib
        
        for i in range(1,len(length)):
            a = np.nonzero(self.xmesh >= self.length[i-1] and self.xmesh < self.length[i])
            self.sigmat[a] = sigmat[i]
            self.sigmas[a] = sigmas[i]
            
        self.probscatter = self.sigmas/(self.sigmat)
        self.probabsorb = 1-self.probscatter

        self.reflect = False # key to reset scatter angle or not
        
        # Tally for absorption (0), scattering (1), distance(2)
        # + current (2), -current(3)
        # track length (4)
        self.celltally = np.zeros([3,self.Nx])
        self.surfacetally = np.zeros([2,self.Nx+1])
                          
    def get_sourceposition(self):
        """
        Returns
        -------
            x - initial position of a particle in the source
        """
        
        # Assume isotropic source in box
        # Then uniform probability distribution 1/length, 
        # CDF x/length = rand() : x = totallength * rand
        
        """Incorporate boundaries, multimaterials"""
        
        # Integrate total source
        left = 1
        right = 1
        if self.boundary[0]==0:
            left = 0
        if self.boundary[1]==0:
            right = 0
        
        self.total = left*(self.psib[0]+self.psif[0]) + np.dot(self.q0,self.length) + right*(self.psib[1]+self.psif[1])
        prob = rand(1)*self.total
        
        if prob < left*(self.psib[0]+self.psif[0]):
            x = 0
            self.surface = True
            if prob < left*self.psib[0]:
                self.forward = False
            else:
                self.forward = True
        elif prob > self.total - right*(self.psib[1]+self.psif[1]):
            x = self.totallength
            self.surface = True
            if prob > self.total - right*self.psif[1]:
                self.forward = True
            else:
                self.forward=False
        else:
            self.surface = False
            x = (prob - left*(self.psib[0]+self.psif[0]))/np.sum(self.q0)
                    
        # print("x ",x)
        
        return(x)
    
    def get_cell(self):
        
        if self.x < 0 or self.x > self.totallength:
            return(None)
        else:
            a = np.argwhere(self.xmesh <= self.x)
            return(a[-1])
        
    def get_sourceangle(self):
        """
        Returns
        -------
        mu - value of mu traveling initially
        """
        
        # Isotropic source, PDF uniform 1/2, CDF = 1/2(mu+1) = rand
        # mu = 2*rand - 1
        
        prob = rand(1)
        if not self.surface:            
            mu = np.cos(prob * np.pi)
            self.forward =mu > 0 
        else:
            # Isotropic but positive
            mu = np.cos(prob * np.pi/2)
            if not self.forward:
                mu = - mu
        
        return(mu)
    
    def get_scatterangle(self):
        """
        Returns
        -------
        angle of particle after scattering reaction
        """
        
        # Isotropic scattering, PDE 1/2, CDF (mu+1)/2
        
        mu = np.cos(np.pi*rand(1))
        
        return(mu)
        
    def get_travellength(self):
        """
        Arguments:
            x : double, position of particle (to allow for scattering 
                                              within material)
        Returns
        ------
        travel length of particle after initialization or scattering
        """
        
        prob = rand(1)
        distance = -np.log(prob)/self.sigmat[self.cell]
        
        return(distance)
        
        
    def whichreaction(self):
        
        prob = rand(1)
        if prob <= self.probscatter[self.cell]:
            rtype = 1
        else:
            rtype = 0
        
        return(rtype)
    
    def tally(self):
        
        if not self.reflect:
            if self.rtype == 0:
                self.celltally[0,self.cell] += 1
            if self.rtype == 1:
                self.celltally[1,self.cell] += 1
            # Track length
        self.celltally[2,self.oldcell] += self.distance

        # Forward and backward current
        if self.mu > 0:
            original = (self.smesh >= self.oldx)
            final = (self.smesh <= self.x)
            
            a = np.nonzero(original * final)
            # print("Surface Forward",a)
            self.surfacetally[0,a] += 1
        else:
            # print(self.x,self.oldx)
            original = (self.smesh <= self.oldx)
            final = (self.smesh >= self.x)
            a = np.nonzero(original * final)
            # print("Surface Backward",a)
            self.surfacetally[1,a] += 1
        
        return(None)

    def getmoments(self):
        
        self.scalarflux_rates = (self.celltally[0,:]+self.celltally[1,:])/(self.part * self.dx * self.sigmat)*self.total
        self.scalarflux_distance = self.celltally[2,:]/(self.part * self.dx)*self.total
        self.current = (self.surfacetally[0,:]-self.surfacetally[1,:])/(self.part)*self.total
        
        self.sfrerr = 2*np.sqrt(self.celltally[0,:]+self.celltally[1,:])/(self.part * self.dx * self.sigmat)*self.total
        self.sfderr = 2*np.sqrt(self.celltally[2,:])/(self.part * self.dx)*self.total
        self.curerr = 2*np.sqrt(self.surfacetally[0,:]+self.surfacetally[1,:])/(self.part)*self.total
        
    def plotmoments(self):
        
        plt.errorbar(self.xmesh,self.scalarflux_rates,yerr=self.sfrerr,color="mediumturquoise",ecolor="black",label="Scalar Flux Rates")
        plt.errorbar(self.xmesh,self.scalarflux_distance,yerr=self.sfderr,color="tomato",ecolor="black",label="Scalar Flux Distance")
        plt.errorbar(self.smesh,self.current,yerr=self.curerr,color="goldenrod",ecolor="black",label="Current")
        plt.legend(loc="upper right")
        
        plt.savefig("mc"+str(self.Nx))
        plt.show()
        plt.close()
        
    def simulation(self):
        
        # Number of particles
        
        self.part = 0
        
        self.sfrerr = 10*np.ones_like(self.xmesh)
        self.sfderr = 10*np.ones_like(self.xmesh)
        self.curerr = 10*np.ones_like(self.smesh)
        err = 10
        while self.part < self.NP and err > 10**(-3):
            
            if self.part % 10000 == 0:
                print("Particle ",self.part)
                print("Max Currents",np.amax(self.surfacetally[0,:]),np.amax(self.surfacetally[1,:]))
            
            self.x = self.get_sourceposition()
            self.cell = self.get_cell()
            self.rtype = 1
                        
            self.iteration = 0
            while self.rtype != 0 and (self.x >= 0 or self.boundary[0]==0) and (self.x <= self.totallength or self.boundary[1]==0):
                
                if self.iteration == 0:
                    self.mu = self.get_sourceangle()
                    self.rtype = self.whichreaction()
                    self.distance = self.get_travellength()
                elif not self.reflect:
                    self.mu = self.get_scatterangle()
                    self.rtype = self.whichreaction()
                    self.distance = self.get_travellength()
                
                self.move = self.mu * self.distance
                self.oldx = np.copy(self.x)
                self.x += self.move
                #print(self.distance)
                #print(self.x,self.oldx)
                self.oldcell = self.cell.copy()
                self.cell = self.get_cell()
                
                # Adjust for reflecting boundary conditions
                # In the event of a reflection, we assume the particle returns
                # to its original cell with the reaction type there
                
                while (self.x < 0 and self.boundary[0]==0):
                    self.reflect = True
                    self.x = 0
                    self.cell = 0
                    self.tally()
                    
                    self.mu = -self.mu
                    self.x = self.oldx.copy()
                    self.oldx = 0
                    self.cell = self.oldcell.copy()
                    self.oldcell = 0
                                        
                while (self.x > self.totallength and self.boundary[1]==0):
                    self.reflect = True
                    self.x = self.totallength.copy()
                    self.cell = -1
                    self.tally()
                    
                    self.mu = - self.mu
                    self.x = self.oldx.copy()
                    self.oldx = self.totallength.copy()
                    self.cell = self.oldcell.copy()
                    self.oldcell = -1
                self.reflect = False
                
                # Tally for future reactions
                self.tally()
                    
                self.iteration += 1
            self.part += 1
            if self.part % 10000 == 0:
                self.getmoments()
                err = (np.sum(self.sfrerr)+np.sum(self.sfderr)+np.sum(self.curerr))/(3*self.Nx+1)
                print("Avg Uncertainty",err)
                    
        self.getmoments()
        self.plotmoments()
        
        return(None)
        