#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 18:09:03 2025

@author: ehansen
"""

import numpy as np
from numpy.random import rand as rand
import matplotlib.pyplot as plt
from numpy import format_float_positional as ff
import os



class MonteCarlo1DSolver:
    def __init__(self,length,sigmat,sigmas,q0,Nx,NP,
                 boundary,psif,psib,name):
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
        self.cumulativelength = np.cumsum(self.length)
        self.lengthrange = range(len(self.length))        
        
        
        self.q0 = np.array(q0)
        
        self.Nx = Nx
        self.NP = NP
        self.totallength = np.sum(self.length)
        self.dx = self.totallength/self.Nx
        
        self.surfacemesh = np.linspace(0,self.totallength,self.Nx+1)
        self.cellmesh = self.surfacemesh[:-1]
        
        a = np.nonzero(self.cellmesh < self.length[0])
        
        self.sigmatmat = np.array(sigmat)
        self.sigmasmat = np.array(sigmas)
        
        self.sigmat = np.zeros(self.Nx)
        self.sigmas = np.zeros(self.Nx)

        self.sigmat[a] = sigmat[0]
        self.sigmas[a] = sigmas[0]
        
        self.boundary = boundary
        self.psif = psif
        self.psib = psib
        
        self.name = name
        
        for i in range(1,len(length)):
            a = np.nonzero(self.cellmesh >= self.length[i-1] and self.cellmesh < self.length[i])
            self.sigmat[a] = sigmat[i]
            self.sigmas[a] = sigmas[i]
            
        self.probscatter = self.sigmas/(self.sigmat)
        self.probabsorb = 1-self.probscatter

        self.reflect = False # key to reset scatter angle or not
        
        # Tally for reactions(0), distance(1)
        # + current (0), -current(1)
        self.celltally = np.zeros([4,self.Nx])
        self.surfacetally = np.zeros([4,self.Nx+1])
        
        # Integrate total source
        self.left = 1
        self.right = 1
        if self.boundary[0]==0:
            self.left = 0
        if self.boundary[1]==0:
            self.right = 0
        
        # Get total source per unit length area - will be used for normalization
        self.total = self.left*(self.psib[0]+self.psif[0]) + np.dot(self.q0,self.length) + self.right*(self.psib[1]+self.psif[1])
                          
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
        
        
        prob = rand(1)*self.total
        
        if prob < self.left*(self.psib[0]+self.psif[0]):
            x = 0
            self.surface = True
            if prob < self.left*self.psib[0]:
                self.forward = False
            else:
                self.forward = True
        elif prob > self.total - self.right*(self.psib[1]+self.psif[1]):
            x = self.totallength
            self.surface = True
            if prob > self.total - self.right*self.psif[1]:
                self.forward = True
            else:
                self.forward=False
        else:
            self.surface = False
            x = (prob - self.left*(self.psib[0]+self.psif[0]))/np.sum(self.q0)
                    
        # print("x ",x)
        
        return(x)
    
    def get_cell(self):
        
        if self.x < 0 or self.x > self.totallength:
            return(None)
        else:
            a = np.argwhere(self.cellmesh <= self.x)
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

        # Find distance in mean free paths
        prob = rand(1)
        distance = -np.log(prob)
        
        if len(self.length) == 1:
            return(distance/self.sigmatmat[0])
        else:
            # Find projected x distance in mean free paths
            xdistance = self.mu * distance
            
            currentplace = self.x
            travel = 0
        
            # Convert x lengths of travel into mean free paths
            if self.mu > 0 :
                a = np.nonzero(self.cumulativelength-currentplace > 0)
                ii = a[0] # which material particle is in
                # xdistance in mean free paths to leave material
                leavemat = (self.cumulativelength[ii]-currentplace)*self.sigmatmat[ii]
                while xdistance > leavemat:
                    travel += leavemat/(self.mu*self.sigmatmat[ii])
                    xdistance -= leavemat
                    currentplace = self.cumulativelength[ii]
                    ii += 1
                    leavemat = (self.cumulativelength[ii]-currentplace)*self.sigmatmat[ii]
                travel += xdistance/(self.mu*self.sigmatmat[ii])
                return(travel)
            elif self.mu < 0:
                a = np.nonzero(self.cumulativelength-currentplace < 0)
                ii = a[-1] # which material particle is in
                # xdistance in mean free paths to leave material
                leavemat = (currentplace-self.cumulativelength[ii])*self.sigmatmat[ii]
                while xdistance > leavemat:
                    travel += leavemat/(self.mu * self.sigmatmat[ii])
                    xdistance -= leavemat
                    currentplace = self.cumulativelength[ii]
                    ii -= 1
                    leavemat = (currentplace-self.cumulativelength[ii])*self.sigmatmat[ii]
                travel += xdistance/(self.mu * self.sigmatmat[ii])
                return(travel)
            else:
                return(distance/self.sigmat[self.cell])

    def whichreaction(self):
        
        prob = rand(1)
        if prob <= self.probscatter[self.cell]:
            rtype = 1
        else:
            rtype = 0
        
        return(rtype)
    
    def tally(self):
        
        if not self.reflect:
            self.celltally[0,self.cell] += 1
            # E[X^2] tallies for variance
            self.celltally[2,self.cell] += 1
            
        # Compute track length tally
        
        if self.oldcell == self.cell or self.mu == 0:
            self.celltally[1,self.oldcell] += self.distance 
            self.celltally[3,self.oldcell] += self.distance**2.0
        elif self.mu > 0:
            original = self.cellmesh > self.oldx
            final = self.cellmesh < self.x 
            a = np.argwhere(original  * final)
            currentx = self.oldx
            for ii in a[:-1]:
                xcelldist = np.abs((self.surfacemesh[ii+1]-currentx)/self.mu)
                self.celltally[1,ii] += (xcelldist)
                self.celltally[3,ii] += (xcelldist)**2.0
                currentx = self.surfacemesh[ii+1]
            xcelldist = np.abs((self.x - self.surfacemesh[a[-1]])/self.mu)
            self.celltally[1,a[-1]] += xcelldist
            self.celltally[3,a[-1]] += xcelldist**2.0
        elif self.mu < 0:
            original = self.cellmesh < self.oldx
            final = self.x < self.cellmesh 
            
            a = np.argwhere(original * final)
            currentx = self.oldx
            for ii in a[::-1]:
                xcelldist = np.abs((currentx - self.surfacemesh[ii])/self.mu)
                self.celltally[1,ii] += (xcelldist)
                self.celltally[3,ii] += (xcelldist)**2.0
                currentx = self.surfacemesh[ii]
            xcelldist = np.abs((self.x - self.surfacemesh[a[0]])/self.mu)
            self.celltally[1,a[0]-1] += xcelldist
            self.celltally[3,a[0]-1] += xcelldist**2.0

        # Forward and backward current
        if self.mu > 0:
            original = (self.surfacemesh >= self.oldx)
            final = (self.surfacemesh <= self.x)
            
            a = np.nonzero(original * final)
            # print("Surface Forward",a)
            self.surfacetally[0,a] += 1
            # E[X^2] tallies for variance
            self.surfacetally[2,a] += 1
        else:
            # print(self.x,self.oldx)
            original = (self.surfacemesh <= self.oldx)
            final = (self.surfacemesh >= self.x)
            a = np.nonzero(original * final)
            # print("Surface Backward",a)
            self.surfacetally[1,a] += 1
            self.surfacetally[3,a] += 1
            

        
        return(None)


    def getmoments(self):
        
        
        
        # Compute averages and variances
        
        self.scalarflux_rates = self.celltally[0,:]/(self.iteration)
        self.scalarflux_distance = self.celltally[1,:]/(self.iteration)
        self.current = (self.surfacetally[0,:]-self.surfacetally[1,:])/self.iteration
        
        self.sfrerr = np.sqrt(self.celltally[2,:]/self.iteration - self.scalarflux_rates**2.0)
        self.sfderr = np.sqrt(self.celltally[3,:]/self.iteration - self.scalarflux_distance**2.0)
        # Be a bit careful here E[(X-Y)^2] = E[X^2]+E[Y^2] - 2 E[XY]
        # In every case since mu is either positive or negative, XY = 0, so positive and negative fluxes uncorrelated
        self.curerr = np.sqrt((self.surfacetally[2,:]+self.surfacetally[3,:])/self.iteration - self.current**2.0)
        
        # Normalize tallies to total incoming source * length from boundaries and media
        
        self.scalarflux_rates *= self.total/(self.dx * self.sigmat)
        self.scalarflux_distance *= self.total/self.dx
        self.current *= self.total
        
        # Similarly rescale errors, but with 2 * np.sqrt(1/(self.iteration-1)) 
        # (divide by sqrt(self.iteration) for uncertainty in mean
        
        self.sfrerr *= 2 * np.sqrt(1/(self.iteration-1)) * self.total/(self.dx * self.sigmat)
        self.sfderr *= 2 * np.sqrt(1/(self.iteration-1)) * self.total/(self.dx)
        self.curerr *= 2 * np.sqrt(1/(self.iteration-1)) * self.total
        
        
        
    def plotmoments(self):
        
        plt.errorbar(self.cellmesh,self.scalarflux_rates,yerr=self.sfrerr,
                     color="mediumturquoise",ecolor="black",label="Scalar Flux Rates")
        plt.errorbar(self.cellmesh,self.scalarflux_distance,yerr=self.sfderr,
                     color="tomato",ecolor="black",label="Scalar Flux Distance")
        plt.errorbar(self.surfacemesh,self.current,yerr=self.curerr,
                     color="goldenrod",ecolor="black",label="Current")
        plt.title("Scalar Flux and Current Scattering " + ff(self.sigmas[0],5))
        plt.legend(loc="upper right")
        
        if not os.path.exists("montecarlo/"):
            os.mkdir("montecarlo/")
        plt.savefig("montecarlo/"+self.name+"fluxcurrent"+str(self.NP//100))
        # plt.show()
        plt.close()
        
    def simulation(self):
        
        # Number of particles
        
        self.particleno = 0
        
        self.sfrerr = 10*np.ones_like(self.cellmesh)
        self.sfderr = 10*np.ones_like(self.cellmesh)
        self.curerr = 10*np.ones_like(self.surfacemesh)
        err = 10
        while self.particleno < self.NP:
            
            if self.particleno % 10000 == 0:
                print("Particle ",self.particleno)
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
                
                self.xmove = self.mu * self.distance
                self.oldx = np.copy(self.x)
                self.x += self.xmove
                #print(self.distance)
                #print(self.x,self.oldx)
                self.oldcell = self.cell.copy()
                self.cell = self.get_cell()
                
                # Adjust for reflecting boundary conditions
                # In the event of a reflection, we assume the particle returns
                # to its original cell with the reaction type there
                
                self.reflect = False
                while (self.x < 0 and self.boundary[0]==0):
                    self.reflect = True
                    self.x = 0
                    self.cell = 0
                    self.tally() #keep track of distance and current tallies
                    
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
                
                # Tally for future reactions
                self.tally()
                    
                self.iteration += 1
            self.particleno += 1
            if self.particleno % 10000 == 0:
                self.getmoments()
                err = (np.sum(self.sfrerr)+np.sum(self.sfderr)+np.sum(self.curerr))/(3*self.Nx+1)
                print("Avg Uncertainty",err)
                    
        self.getmoments()
        self.plotmoments()
        
        return(None)
    
    ## Simplified Simulations Follow
    
    def simple_distanceabsorb(self,simple_side,mu):
        """
        One number per particle history - generate distance travelled
        Arguments:
            simple_side - either 0 or the total length of cells
            mu - fixed incident angle
        """
        
        
        self.particleno = 0
        self.iteration = 0 
        
        while self.particleno < self.NP:
            if self.particleno % 5000 == 0:
                print("Particle No.", self.particleno)
                
            self.particleno += 1
            self.iteration += 1
            
            self.x = simple_side
            self.mu = mu
            
            self.cell = self.get_cell()
            self.distance = self.get_travellength()
            self.xmove = self.mu * self.distance
            
            self.oldx = np.copy(self.x)
            self.x += self.xmove
            self.oldcell = self.cell.copy()
            self.cell = self.get_cell()
            
            self.rtype = 0
                        
            # We know the particle will be absorbed so just tally
            
            self.tally()

        print(self.iteration)
        self.getmoments()
        self.plotmoments()           
        
    def simple_locationdirectionreflect(self,travellength=0):
        """
        Two numbers per particle history - generate source position and mu
        with reflecting boundary condition
        Arguments:
            simple_side - either 0 or the total length of cells
            mu - fixed incident angle
        """
        self.particleno = 0
        self.iteration = 0 
        
        while self.particleno < self.NP:
            if self.particleno % 5000 == 0:
                print("Particle No.", self.particleno)
                
            self.particleno += 1
            self.iteration += 1
            self.reflect = False

            
            self.x = self.get_sourceposition()
            self.mu = self.get_sourceangle()
            
            self.cell = self.get_cell()
            
            if travellength == 0:
                self.distance = self.get_travellength()
            else:
                self.distance = travellength
            
            self.xmove = self.mu * self.distance
            
            self.oldx = np.copy(self.x)
            self.x += self.xmove
            self.oldcell = self.cell.copy()
            self.cell = self.get_cell()
            
            self.rtype = 0

            # Reflecting boundary conditions
            if self.x < 0:
                
                self.reflect = True
                
                self.x = 0
                self.cell = self.get_cell()
                
                self.tally()
                
                self.oldx = 0
                self.oldcell = 0
                self.mu = - self.mu
                self.x -= self.xmove
                self.cell = self.get_cell()
                                                
            if self.x > self.totallength:
                
                self.reflect= True
                
                self.x = self.totallength.copy()
                self.cell = self.get_cell()
                
                self.tally()
                
                self.oldcell = self.cell.copy()
                self.oldx = np.copy(self.x)
                self.mu = - self.mu
                self.x -= self.xmove
                self.cell = self.get_cell()
                
            self.reflect = False
            self.tally()
            
        self.getmoments()
        self.plotmoments()
            
                    
        
    def simple_angledistanceabsorb(self,simple_side):
        """
        Incident flux in pure absorber

        Parameters
        ----------
        simple_side : initial position of the particle

        Returns
        -------
        None.

        """
        self.particleno = 0
        self.iteration = 0 
        
        while self.particleno < self.NP:
            if self.particleno % 5000 == 0:
                print("Particle No.", self.particleno)
            self.particleno += 1
            self.iteration += 1
            
            
            self.x = simple_side
            self.cell = self.get_cell()
            
            self.surface = True
            self.forward = (simple_side < self.totallength/2)
            self.mu = self.get_sourceangle()
            self.distance = self.get_travellength()
            self.xmove = self.mu * self.distance
            
            self.oldx = np.copy(self.x)
            self.x += self.xmove
            self.oldcell = self.cell.copy()
            self.cell = self.get_cell()
            
            self.rtype = 0
            
            self.tally()
            
        self.getmoments()
        self.plotmoments()
            
            
            
        
        
        
        