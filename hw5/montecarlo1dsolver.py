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
        self.materialend = np.cumsum(self.length)
        # Get start of materials 
        self.materialstart = np.zeros_like(self.materialend)
        self.materialstart[1:] = self.materialend[:-1]
        self.lengthrange = range(len(self.length))        
        
        self.q0 = np.array(q0)
        
        self.Nx = Nx
        self.NP = NP
        self.totallength = np.sum(self.length)
        self.dx = self.totallength/self.Nx
        
        self.surfacemesh = np.linspace(0,self.totallength,self.Nx+1)
        self.cellmesh = self.surfacemesh[:-1]
        
        a = np.nonzero(self.cellmesh < self.length[0])
        
        self.sigmat_material = np.array(sigmat)
        self.sigmas_material = np.array(sigmas)
        
        # Find length of each material in mean free paths
        self.meanfreepath_material = self.length*self.sigmat_material
        
        self.sigmat = np.zeros(self.Nx)
        self.sigmas = np.zeros(self.Nx)

        self.sigmat[a] = sigmat[0]
        self.sigmas[a] = sigmas[0]
        
        self.boundary = boundary
        self.psif = psif
        self.psib = psib
        
        self.name = name
        
        for i in range(1,len(length)):
            a1 = self.cellmesh >= self.materialend[i-1]
            a2 = self.cellmesh < self.materialend[i]
            a = np.nonzero(a1*a2)
            self.sigmat[a] = sigmat[i]
            self.sigmas[a] = sigmas[i]
            
        self.probscatter = self.sigmas/(self.sigmat)
        self.probabsorb = 1-self.probscatter
        
        # Tally for reactions(0), distance(1), reactions^2 (2), distance^2 (3)
        # Use squared tallies for variance
        self.celltally = np.zeros([4,self.Nx])
        # + current (0), -current(1), +current^2 (2), -current^2 (3)
        self.surfacetally = np.zeros([4,self.Nx+1])
        
        # Integrate total source
        
        self.left = 1
        self.right = 1
        if self.boundary[0] == 0:
            self.left = 0
        if self.boundary[1] == 0:
            self.right = 0
        
        # Get total source per unit length area - will be used for normalization
        self.total = self.left*self.psif[0] + np.dot(self.q0,self.length) + self.right*self.psib[1]
        
        self.reflect = False
                          
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
        
        
        # Choose whether to start from a boundary or interior
        prob = rand(1)
        
        if prob < self.left * self.psif[0]/self.total:
            x = [0]
            self.surface = True
            self.forward = True
        elif prob > 1- self.right * self.psib[1]/self.total:
            x = [self.totallength]
            self.surface = True
            self.forward = False
        else:
            # Find initial position of particle from region sources
            self.surface = False
            prob = rand(1)
            # CDF of interior sources
            sourcecdf = np.cumsum(self.q0*self.length/np.dot(self.q0,self.length))
            # Find which material the particle is in
            ii = min(np.argwhere(sourcecdf >= prob))
            if ii == 0:
                x = self.length[0] * prob/sourcecdf[ii]
            else:
                x = self.length[ii-1] + (prob-sourcecdf[ii-1])/(self.cdf[ii]-sourcecdf[ii-1]) * (self.length[ii]-self.length[ii-1])
        
        return(x[0])
    
    def get_cell(self):
        
        if self.x < 0:
            return(0)
        elif self.x > self.totallength:
            return(self.Nx-1)
        else:
            a = np.argwhere(self.cellmesh <= self.x)
            return(a[-1][0])
    
    def get_material(self):
        """
        

        Returns
        -------
        Index of material that the particle is in

        """
        
        # We handle a lot of boundary cases where x is equal to one of the elements
        # So we will push x slightly according to the sign of mu so travel is correct
        # This won't affect subsequent values because this isn't 
        
        # Boundary case - need to know sign of mu for which material to use
        ii = np.amin(np.argwhere(self.materialend - self.x >= 0))
        if np.amin(np.abs(self.materialend - self.x)) <= 10**(-10):
            if self.mu >= 0:
                return(ii+1)
            else:
                return(ii)
        else:
            return(ii)
        
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
        
    def distanceloop(self,fixedxdistance=0):
        """
        Structure - work through each material 
        Arguments:
           fixedxdistance - prescribe distance that must be travelled in x
        Returns
        ------
        travel length of particle after initialization or scattering
        """
        
        self.oldx = np.copy(self.x)
        self.oldcell = np.copy(self.cell)
                
        # Find total distance in mean free paths to travel
        if fixedxdistance == 0:
            mfpdistance = -np.log(rand(1))
        mfpdistance_remaining = mfpdistance    
        
        # Exception for mu = 0
        if self.mu == 0:
            
            self.oldx = np.copy(self.x)
            self.oldcell = np.copy(self.cell)
            
            self.distance = mfpdistance/self.sigmat[self.cell]
            self.tracklengthtally()
            self.currenttally()
            mfpdistance_remaining = 0
        
        # Move through materials and tally in each until mfp_distancereamining runs out        
        while mfpdistance_remaining > 0:
            
            self.oldx = np.copy(self.x)
            self.oldcell = np.copy(self.cell)
            
            self.material = self.get_material()
            # print(self.particleno,self.x,self.cell,self.mu,mfpdistance_remaining,self.material)
            
            if self.mu > 0:
                # Distance to leave material in mean free paths
                leavemat = (self.materialend[self.material] - self.oldx)*self.sigmat_material[self.material]/self.mu
                
                if mfpdistance_remaining > leavemat:
                    
                    self.distance = (self.materialend[self.material] - self.oldx)/self.mu
                    mfpdistance_remaining -= leavemat
                    
                    self.x = self.materialend[self.material]
                    self.cell = self.get_cell()
                else:
                    self.distance = mfpdistance_remaining/self.sigmat_material[self.material]
                    mfpdistance_remaining = 0
                    
                    self.xmove = self.mu * self.distance 
                    self.x += self.xmove
                    self.cell = self.get_cell()
                
            else:
                
                leavemat = -(self.oldx - self.materialstart[self.material])*self.sigmat_material[self.material]/self.mu
                
                if mfpdistance_remaining > leavemat:
                    
                    self.distance = -(self.oldx - self.materialstart[self.material])/self.mu
                    mfpdistance_remaining -= leavemat
                    
                    # Push particle slightly into
                    self.x = self.materialstart[self.material]
                    self.cell = self.get_cell()
                    
                    
                else:
                    self.distance = mfpdistance_remaining/self.sigmat_material[self.material]
                    mfpdistance_remaining = 0
                    self.xmove = self.mu * self.distance
                    self.x += self.xmove
                    self.cell = self.get_cell()
            
            # Tally now before reflection or leaving
            self.tracklengthtally()
            self.currenttally()
            
            if self.x <= 0:
                
                if self.reflect or self.boundary[0] == 0:
                    self.mu *= -1 # For reflecting boundary, change particle direction and allow to continue
                else:
                    self.x -= 1 # Force particle outside of domain so break particle loop
                    mfpdistance_remaining = 0 # Terminate material loop
                                
            if self.x >= self.totallength: 
                if self.reflect or self.boundary[1] == 0:
                    self.mu *= -1
                else:
                    self.x += 1 # Force particle outside boundary so break particle loop
                    mfpdistance_remaining = 0 # Terminate material loop            
                    
        
    def whichreaction(self):
        
        prob = rand(1)
        if prob <= self.probscatter[self.cell]:
            rtype = 1
        else:
            rtype = 0
        return(rtype)
        
    def reactiontally(self):
        
        if self.x >= 0 and self.x <= self.totallength:
            self.celltally[0,self.cell] += 1
            # E[X^2] tallies for variance
            self.celltally[2,self.cell] += 1
            
    def get_intra_cell_distance(self):
        
        if self.mu == 0:
            return(self.distance)
        else:
            if self.mu > 0:
                upperbound = min(self.x,self.surfacemesh[self.cell+1])
                lowerbound = max(self.oldx,self.surfacemesh[self.cell])
            else:
                upperbound = min(self.oldx,self.surfacemesh[self.cell+1])
                lowerbound = max(self.x,self.surfacemesh[self.cell])
            return((upperbound-lowerbound)/np.abs(self.mu))
            
    def tracklengthtally(self):
        
        # Check that particle position and old position update appropriately
        # There was a big with this due to numpy array standards
        
        if self.x == self.oldx and self.mu != 0 and self.iteration % 100 == 0:
            print("Old x and x agree!!\n"*10)
            print(self.x, self.oldx,self.oldcell,self.distance*self.mu,self.cell,self.iteration)
         
        # Check that particles in different cells are in different cells,
        # and that their stored distance would be appropriate
        
        if self.cell == self.oldcell and np.abs(self.x - self.oldx) > self.dx:
            print("Old cell and cell agree!!\n"*10)
            print("x, oldx, cell, oldcell")
            print(self.x,self.oldx,self.cell,self.oldcell,self.distance*self.mu)
        
        if self.oldcell == self.cell or self.mu == 0:
            
            cell_distance_traveled = self.get_intra_cell_distance()
            
            self.celltally[1,self.oldcell] += cell_distance_traveled
            self.celltally[3,self.oldcell] += cell_distance_traveled**2.0
                                
        elif self.mu > 0:
            
            if self.oldx >= self.surfacemesh[self.oldcell]:
                oldcelldist = (self.surfacemesh[self.oldcell+1]-self.oldx)/self.mu
            else:
                oldcelldist = self.dx/self.mu

            # oldcelldist = (self.surfacemesh[self.oldcell+1]-self.oldx)/self.mu
            
            trialdistance = oldcelldist
            self.celltally[1,self.oldcell] += oldcelldist
            self.celltally[3,self.oldcell] += oldcelldist**2.0
            
            for cell in range(self.oldcell+1,self.cell):
                self.celltally[1,cell] += (self.dx)/self.mu
                self.celltally[3,cell] += (self.dx/self.mu)**2.0
                trialdistance += self.dx/self.mu
            
            if self.x >= self.surfacemesh[self.cell+1]:
                finalcelldist = self.dx/self.mu
            else:
                finalcelldist = (self.x - self.surfacemesh[self.cell])/self.mu
                
            
            # finalcelldist = 0
            # finalcelldist = (self.x - self.surfacemesh[self.cell])/self.mu

            #finalcelldist = (self.x - self.surfacemesh[self.cell])/self.mu
            
            
            self.celltally[1,self.cell] += finalcelldist
            self.celltally[3,self.cell] += finalcelldist**2.0
            
            trialdistance += finalcelldist
            
            # Check on whether the tallied distance was equal or strictly less
            # than the called distance
            # It could be less if the particle leaves the system
            
            # if trialdistance - self.distance > 10**(-10):
            #     print("Track Length not distance")
            #     print("Particle ",self.particleno,"Iteration ",self.iteration)
            #     print("x ",self.x)
            #     print("mu ",self.mu)
            #     print("oldx ",self.oldx)
            #     print("cell ",self.cell)
            #     print("old cell ",self.oldcell)
            #     print("distance ",self.distance)
            #     print("trial distance" ,trialdistance)
            #     print("distance *mu > dx",(self.distance*self.mu > self.dx))
            #     print("different cells",(np.abs(self.oldcell-self.cell) > 0))
        
        elif self.mu < 0:
            
            if self.oldx <= self.surfacemesh[self.oldcell+1]:
                oldcelldist = (self.surfacemesh[self.oldcell]-self.oldx)/self.mu
            else:
                oldcelldist = -self.dx/self.mu
                
            self.celltally[1,self.oldcell] += oldcelldist
            self.celltally[3,self.oldcell] += oldcelldist**2.0
            
            for cell in range(self.cell+1,self.oldcell):
                self.celltally[1,cell] += -(self.dx)/self.mu
                self.celltally[3,cell] += (self.dx/self.mu)**2.0
            
            if self.x <= 0:
                finalcelldist = -self.dx/self.mu
            else:
                finalcelldist = (self.x - self.surfacemesh[self.cell+1])/self.mu
                
            # finalcelldist = (self.x - self.surfacemesh[self.cell+1])/self.mu
            # finalcelldist = 0
            
            #finalcelldist = -(self.surfacemesh[self.cell+1]-self.x)/self.mu
            self.celltally[1,self.cell] += finalcelldist
            self.celltally[3,self.cell] += finalcelldist**2.0
        
    
    def currenttally(self):
        
        if self.mu > 0:
            
            self.surfacetally[0,self.oldcell+1:self.cell+1] += 1
            self.surfacetally[2,self.oldcell+1:self.cell+1] += 1
            if self.x >= self.totallength:
                self.surfacetally[0,-1] += 1
                self.surfacetally[2,-1] += 1
            if self.oldx <= 0: # >=0 leads to point spike
                self.surfacetally[0,0] += 1
                self.surfacetally[2,0] += 1
                
        else:
            
            self.surfacetally[1,self.cell+1:self.oldcell+1] += 1
            self.surfacetally[3,self.cell+1:self.oldcell+1] += 1
            if self.oldx >= self.totallength:
                self.surfacetally[1,-1] += 1
                self.surfacetally[3,-1] += 1
            if self.x <= 0:
                self.surfacetally[1,0] += 1
                self.surfacetally[3,0] += 1            
        
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
            self.distanceloop()
            
            # We know the particle will be absorbed so just tally
            self.rtype = 0
            self.reactiontally()
            print(self.particleno)                        

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
        self.reflect = True
                        
        while self.particleno < self.NP:
            if self.particleno % 5000 == 0:
                print("Particle No.", self.particleno)
                
            self.particleno += 1
            self.iteration += 1
            
            self.x = self.get_sourceposition()
            self.mu = self.get_sourceangle()
            self.cell = self.get_cell()
            
            self.distanceloop(fixedxdistance=travellength)
            
            self.reactiontally()
                
            
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
            
            self.iteration += 1
            
            self.x = simple_side
            self.cell = self.get_cell()
            
            self.surface = True
            self.forward = (simple_side < self.totallength/2)
            self.mu = self.get_sourceangle()
            self.distanceloop()
            
            self.rtype = 0
            self.reactiontally()
            
        self.getmoments()
        self.plotmoments()
        
    def simulation(self):
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
            if self.particleno % 50 == 0:
                print("Particle No.", self.particleno)
                
            self.particleno += 1
            self.iteration += 1
            self.particleiteration = 0
            self.rtype = 1
            self.x = self.get_sourceposition()
            
            while self.x >=0 and self.x <= self.totallength and self.rtype == 1:
                if self.particleiteration == 0:
                    self.mu = self.get_sourceangle()
                else:
                    self.mu = self.get_scatterangle()
                self.cell = self.get_cell()
            
                self.distanceloop()
            
                self.rtype = self.whichreaction()
                self.reactiontally()
                
                self.particleiteration += 1
                
        self.getmoments()
        self.plotmoments()     
    
    def getmoments(self):
         
        # Compute averages and variances
        
        self.scalarflux_rates = self.celltally[0,:]/(self.iteration)
        self.scalarflux_distance = self.celltally[1,:]/(self.iteration)
        self.current = (self.surfacetally[0,:]-self.surfacetally[1,:])/self.iteration
        
        print(np.argwhere(self.celltally[2,:]/self.iteration - self.scalarflux_rates**2.0 < 0))
        print(np.amin(self.celltally[2,:]/self.iteration - self.scalarflux_rates**2.0))

        self.sfrerr = np.sqrt(self.celltally[2,:]/self.iteration - self.scalarflux_rates**2.0)
        print(np.argwhere(self.celltally[3,:]/self.iteration - self.scalarflux_distance**2.0 < 0))
        print(np.amin(self.celltally[3,:]/self.iteration - self.scalarflux_distance**2.0))

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
                     color="mediumturquoise",label="Scalar Flux Rates",ecolor="mediumturquoise",
                     linestyle="",marker="s",markersize=2)
        plt.errorbar(self.cellmesh,self.scalarflux_distance,yerr=self.sfderr,
                     color="tomato",label="Scalar Flux Distance",ecolor="tomato",
                     linestyle="",marker="o",markersize=2)
        plt.title("Scalar Flux Scattering "+ff(self.sigmas[0],5))
        plt.legend(loc="upper right")
        plt.ylim(-np.amax(self.scalarflux_rates)*0.5,np.amax(self.scalarflux_rates)*1.5)
        if not os.path.exists("montecarlo/"):
            os.mkdir("montecarlo/")
        plt.savefig("montecarlo/"+"flux"+self.name+str(self.NP))
        plt.close()
        
        # print(self.current)
        plt.errorbar(self.surfacemesh,self.current,yerr=self.curerr,
                     color="goldenrod",label="Current",linestyle="",marker="3")
        plt.title("Current Scattering " + ff(self.sigmas[0],5))
        plt.legend(loc="upper right")
        plt.ylim(-max(1,np.abs(np.amax(self.current)*1.5)),max(1,np.abs(np.amax(self.current)*1.5)))
        if not os.path.exists("montecarlo/"):
            os.mkdir("montecarlo/")
        plt.savefig("montecarlo/"+"current"+self.name+str(self.NP))
        # plt.show()
        plt.close()
            
            
            
        
        
        
        