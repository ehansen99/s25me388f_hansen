import numpy as np
from numpy import format_float_positional as ff
from scipy.linalg import solve_banded
from scipy.special import roots_legendre,legendre_p
import matplotlib.pyplot as plt
import os
from time import perf_counter

from numba.experimental import jitclass

class Rectangle:
    """
    Builds a Cartesian rectangular mesh cell
    Inputs:
       xleft - left x value
       ybottom - bottom y value
       hside - horizontal side length
       vside - vertical side length
    """

    def __init__(self,xleft,ybottom,hside,vside,sigmat,sigmas,q0,number,numberx,numbery):

        self.x1 = xleft
        self.y1 = ybottom

        self.x2 = xleft + hside
        self.y2 = ybottom

        self.x3 = xleft
        self.y3 = ybottom + vside

        self.x4 = xleft + hside
        self.y4 = ybottom + vside

        self.dx = hside
        self.dy = vside
        self.area = hside * vside

        # Cell material properties
        self.q0 = q0
        self.sigmat = sigmat
        self.sigmas = sigmas

        self.number = number
        self.numberx = numberx
        self.numbery = numbery

class DiamondDifference2D:
    def __init__(self,regions,boundarytype,Ntheta,Nphi,accelerator=None,fullboundary=None):
        
        """
        Initialize solver for 2D discrete ordinates code

        regions - list of 9-tuples
        (x and y bottom left corner positions, lengths in each,Nx,Ny,sigmat, sigmas, q0)
        
        First region must correspond to background and have length equal to problem size
        and problem size Nx, Ny
        All other regions must be disjoint - if cover background, background properties will be overwritten
        """

        self.regions = regions
        self.Ntheta = Ntheta
        self.Nphi = Nphi
        
        self.accelerator = accelerator

        self.boundarytype = boundarytype
        self.fullboundary = fullboundary

        self.definecellmesh()

        self.q0 = np.zeros([self.Nx,self.Ny])
        self.sigmat = np.zeros([self.Nx,self.Ny])
        self.sigmas = np.zeros([self.Nx,self.Ny])

        for cell in self.cellmesh:
            self.q0[cell.numberx,cell.numbery] = cell.q0
            self.sigmat[cell.numberx,cell.numbery] = cell.sigmat
            self.sigmas[cell.numberx,cell.numbery] = cell.sigmas
        
        self.mus,self.muweights = roots_legendre(self.Ntheta)
        # When we do an integral, phi runs from -pi to pi, so we rescale Legendre points/weights
        xs,xweights = roots_legendre(self.Nphi)
        self.phis = np.pi * (xs)
        self.phiweights = xweights*np.pi

        self.scalarflux = np.zeros([self.Nx,self.Ny])
        self.updatesource()
        self.oldflux = -99*np.ones_like(self.scalarflux)

        self.meshplots()

    def definecellmesh(self):
        
        # if self.regions[0][0] != 0 or self.regions[0][1] != 0:
        #     raise RuntimeError("First region must start at (0,0)")
        # elif self.regions[0][2] < np.sum(self.regions[1:][2]) or self.regions[0][3] < np.sum(self.regions[1:][3]):
        #     raise RuntimeError("Background region must be longer than sum of all other regions")
        # elif self.regions[0][4] < np.sum(self.regions[1:][4]) or self.regions[0][5] < np.sum(self.regions[1:][5]):
        #     raise RuntimeError("Background region uses more grid points than sum of all other regions")
        self.Nx = self.regions[0][4]
        self.Ny = self.regions[0][5]

        self.cellmesh = []
        if len(self.regions) == 1:

            region = self.regions[0]

            self.xsurface = np.linspace(region[0],region[0]+region[2],region[4]+1)
            self.ysurface = np.linspace(region[1],region[1]+region[3],region[5]+1)
            self.xcell = np.arange(region[0], region[2] + region[0], region[2] / region[4])
            self.ycell = np.arange(region[1], region[3] + region[1], region[3] / region[5])

            print(len(self.xcell),len(self.ycell))
            for i,x in enumerate(self.xcell):
                for j,y in enumerate(self.ycell):
                    self.cellmesh.append(Rectangle(x, y, region[2]/region[4], region[3]/region[5],
                                                   region[6], region[7], region[8],self.Ny*i+j,i,j))

    def angle_boundaries(self,mu,phi):
        """
        Sets the stated boundary conditions for mu and phi
        """

        # Angular flux variables for given mu, phi computation
        self.angularflux_nodes = np.zeros([self.Nx + 1, self.Ny + 1])
        self.angularflux_cells = np.zeros_like(self.scalarflux)

        # Default to vacuum boundary conditions
        angle_horizontal = 0.0
        angle_vertical = 0.0

        if self.boundarytype == "vacuum":
            angle_horizontal = 0.0
            angle_vertical = 0.0

        elif self.boundarytype == "upright":
            if mu > 0 and np.cos(phi) > 0:
                angle_horizontal = 1.0
                angle_vertical = 1.0
        elif self.boundarytype == "upleft":
            if mu < 0 and np.cos(phi) > 0:
                angle_horizontal = 1.0
                angle_vertical = 1.0
        elif self.boundarytype == "downright":
            if mu > 0 and np.cos(phi) < 0:
                angle_horizontal = 1.0
                angle_vertical = 1.0
        elif self.boundarytype == "downleft":
            if mu < 0 and np.cos(phi) < 0:
                angle_horizontal = 1.0
                angle_vertical = 1.0

        elif self.boundarytype == "full":
            angle_horizontal = self.fullboundary[(mu,phi)][0]
            angle_vertical = self.fullboundary[(mu,phi)][1]

        # Four cases : left, bottom face boundary; left, top; right, bottom; else

        if mu > 0 and np.cos(phi) > 0: # Set bottom and left boundaries
            self.angularflux_nodes[0,:] = angle_horizontal
            self.angularflux_nodes[:,0] = angle_vertical
        elif mu > 0 and np.cos(phi) < 0: # left, top
            self.angularflux_nodes[0,:] = angle_horizontal
            self.angularflux_nodes[:,-1] = angle_vertical
        elif mu < 0 and np.cos(phi) > 0: # right, bottom
            self.angularflux_nodes[-1,:] = angle_horizontal
            self.angularflux_nodes[:,0] = angle_vertical
        else:
            self.angularflux_nodes[-1,:] = angle_horizontal
            self.angularflux_nodes[:,-1] = angle_vertical

    def angularfluxsweep_coefficients(self,mu,phi,dx,dy,sigmat):
        """
        Return coefficients to solve for corner of a mesh cell using finite volume method
        upright * psi(x+1,y+1) + upleft * psi(x,y+1) + downright * psi(x+1,y) + downleft * psi(x,y) = dx * dy * rhs
        """
        upright = mu/2 * dy + np.sqrt(1-mu**2) * np.cos(phi)/2 * dx + sigmat/4 * dx * dy
        upleft = -mu/2 * dy + np.sqrt(1-mu**2) * np.cos(phi)/2 * dx + sigmat/4 * dx * dy
        downright = mu/2 * dy - np.sqrt(1-mu**2) * np.cos(phi)/2 * dx + sigmat/4 * dx * dy
        downleft = -mu/2 * dy - np.sqrt(1-mu**2) * np.cos(phi)/2 * dx + sigmat/4 * dx * dy

        return(upright,upleft,downright,downleft)


    def angularfluxsweep_loop(self,mu,phi):
        """
        Sweeps for the angular flux
        mu - polar angle from x axis
        phi - azimuthal angle from y axis in yz plane
        
        Returns
        -------
        Angular flux for given Omega
        """
        
        if mu > 0 and np.cos(phi) > 0: # solving upright

            for yi in range(0,self.Ny):
                for xi in range(0,self.Nx):

                    dx = self.xsurface[xi+1] - self.xsurface[xi]
                    dy = self.ysurface[yi+1] - self.ysurface[yi]

                    upright,upleft,downright,downleft = self.angularfluxsweep_coefficients(mu,phi,dx,dy,self.sigmat[xi,yi])

                    self.angularflux_nodes[xi+1,yi+1] = (dx * dy * self.rhs[xi,yi]
                                                         - upleft * self.angularflux_nodes[xi,yi+1]
                                                         - downright * self.angularflux_nodes[xi+1,yi]
                                                         - downleft * self.angularflux_nodes[xi,yi])/upright

        elif mu > 0 and np.cos(phi) < 0: # solving downright
             
            for yi in range(self.Ny,0,-1):
                for xi in range(0,self.Nx):

                    dx = self.xsurface[xi + 1] - self.xsurface[xi]
                    dy = self.ysurface[yi] - self.ysurface[yi-1]

                    upright, upleft, downright, downleft = self.angularfluxsweep_coefficients(mu, phi, dx, dy, self.sigmat[xi, yi-1])

                    self.angularflux_nodes[xi+1,yi-1] = (dx * dy * self.rhs[xi,yi-1]
                                                         - upleft * self.angularflux_nodes[xi, yi]
                                                         - upright * self.angularflux_nodes[xi+1,yi]
                                                         - downleft * self.angularflux_nodes[xi,yi-1])/downright

        elif mu < 0 and np.cos(phi) > 0: # solving upleft
            
            for yi in range(0,self.Ny):
                for xi in range(self.Nx,0,-1):

                    dx = self.xsurface[xi] - self.xsurface[xi-1]
                    dy = self.ysurface[yi+1] - self.ysurface[yi]

                    upright,upleft,downright,downleft = self.angularfluxsweep_coefficients(mu, phi, dx, dy, self.sigmat[xi-1, yi])

                    self.angularflux_nodes[xi-1,yi+1] = (dx * dy * self.rhs[xi-1,yi]
                                                         - downright * self.angularflux_nodes[xi,yi]
                                                         - upright * self.angularflux_nodes[xi,yi+1]
                                                         - downleft * self.angularflux_nodes[xi-1,yi])/upleft
            
        else: # solving downleft

            for yi in range(self.Ny,0,-1):
                for xi in range(self.Nx,0,-1):

                    dx = self.xsurface[xi] - self.xsurface[xi-1]
                    dy = self.ysurface[yi] - self.ysurface[yi-1]

                    upright, upleft, downright, downleft = self.angularfluxsweep_coefficients(mu, phi, dx, dy, self.sigmat[xi-1,yi-1])

                    self.angularflux_nodes[xi-1,yi-1] = (dx * dy * self.rhs[xi-1,yi-1]
                                                         - upright * self.angularflux_nodes[xi,yi]
                                                         - upleft * self.angularflux_nodes[xi-1,yi]
                                                         - downright * self.angularflux_nodes[xi,yi-1])/downleft

        self.angularflux_cells = (self.angularflux_nodes[:-1,:-1]+self.angularflux_nodes[:-1,1:]
                                  +self.angularflux_nodes[1:,:-1]+self.angularflux_nodes[1:,1:])/4
        
    def updatesource(self):
        
        self.rhs = (self.q0 + self.sigmas * self.scalarflux)/(4*np.pi)
        
    def src_iteration(self):
        
        error = 99
        iteration = 0
        
        # Source Iteration Loop
        time1 = perf_counter()
        while iteration < 10**5 and error > 10**(-5):
            
            if iteration % 10 == 0:
                print("Iteration "+str(iteration) + " Error " + ff(error,5))
                
            self.oldflux = np.copy(self.scalarflux)
            self.scalarflux = np.zeros_like(self.scalarflux)
            self.currentx = np.zeros_like(self.scalarflux)
            self.currenty = np.zeros_like(self.scalarflux)
            
            # Loop to obtain next scalar flux
            for mit,mu in enumerate(self.mus):
                for pit,phi in enumerate(self.phis):

                    # print(mu,phi)
                    self.angle_boundaries(mu,phi)
                    self.angularfluxsweep_loop(mu, phi)
                    
                    # Given Gaussian quadrature, we can expand out the integral like this
                    self.scalarflux += self.muweights[mit]*self.phiweights[pit] \
                        * self.angularflux_cells  
                        
                    self.currentx += self.muweights[mit]*self.phiweights[pit] \
                        * self.angularflux_cells * mu 
                        
                    self.currenty += self.muweights[mit]*self.phiweights[pit] \
                        * self.angularflux_cells * np.sqrt(1-mu**2) * np.cos(phi)
            
            
            # Acceleration?
            if self.accelerator == "diffusion":
                self.diffusionpreconditioner()
            
            self.updatesource()
            
            # Prepare for next iteration
            iteration += 1
            error = np.amax(np.abs(self.scalarflux-self.oldflux))

        time2 = perf_counter()
        print("Time for Loop " + ff(time2-time1,2))
        print("Iterations "+str(iteration))

        self.resultplots()
            
    def meshplots(self):
        
        X,Y = np.meshgrid(self.xcell,self.ycell,indexing="ij")
        
        # Source plot
        plt.contourf(X,Y,self.q0,cmap="YlGnBu",vmin=0,vmax=2)
        plt.colorbar()
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.title("Material Sources")
        plt.show()
        plt.close()
        
        # Scattering Ratio Plot
        plt.contourf(X,Y,self.sigmas/self.sigmat,cmap="YlOrBr",vmin=0,vmax=1)
        plt.colorbar()
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.title("Scattering Ratio of Geometry")
        
        plt.show()
        plt.close()
        
    def resultplots(self):
        
        X,Y = np.meshgrid(self.xcell,self.ycell,indexing="ij")
        
        plt.contourf(X,Y,self.scalarflux,cmap="YlGnBu",levels=100)
        plt.colorbar()
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.title("Scalar Flux")
        plt.show()
        plt.close()
        
        X,Y = np.meshgrid(self.xcell,self.ycell,indexing="ij")
        
        plt.contourf(X,Y,self.currentx,cmap="RdYlBu",levels=100)
        plt.colorbar()
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.title("X Current")
        plt.show()
        plt.close()
        
        X,Y = np.meshgrid(self.xcell,self.ycell,indexing="ij")
        
        plt.contourf(X,Y,self.currenty,cmap="RdYlBu",levels=100)
        plt.colorbar()
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.title("Y Current")
        plt.show()
        plt.close()

    def diffusionpreconditioner(self):







        raise NotImplementedError("Diffusion preconditioner not implemented")
