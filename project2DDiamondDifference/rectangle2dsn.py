import numpy as np
from numpy import format_float_positional as ff
from scipy.linalg import solve_banded
from scipy.special import roots_legendre,legendre_p
import matplotlib.pyplot as plt
import os
from time import perf_counter

from numba.experimental import jitclass

class DiamondDifference2D:
    def __init__(self,domain,regiontype,boundarytype,Ntheta,Nphi,accelerator=None,fullboundary=None,fname=None,show=False):
        
        """
        Initialize solver for 2D discrete ordinates code

        regions - list of 9-tuples
        (x and y bottom left corner positions, lengths in each,Nx,Ny,sigmat, sigmas, q0)
        
        First region must correspond to background and have length equal to problem size
        and problem size Nx, Ny
        All other regions must be disjoint - if cover background, background properties will be overwritten
        """

        self.domain = domain
        self.regiontype = regiontype
        self.Ntheta = Ntheta
        self.Nphi = Nphi
        
        self.accelerator = accelerator

        self.boundarytype = boundarytype
        self.fullboundary = fullboundary

        self.definecellmesh()

        self.fname = fname
        self.show = show

        self.mus,self.muweights = roots_legendre(self.Ntheta)
        # When we do an integral, phi runs from -pi to pi, so we rescale Legendre points/weights
        xs,xweights = roots_legendre(self.Nphi)
        self.phis = np.pi * (xs+1) / 2
        self.phiweights = xweights*np.pi/2

        self.scalarflux = np.zeros([self.Nx,self.Ny])
        self.updatesource()
        self.oldflux = -99*np.ones_like(self.scalarflux)

        self.meshplots()

    def insidecircletest(self,pointlist,xcenter,ycenter,radius):

        for point in pointlist:
            x = point[0]
            y = point[1]
            if (x-xcenter)**2 + (y-ycenter)**2 > radius**2:
                return(False)
        return(True)

    def definecellmesh(self):
        
        # if self.regions[0][0] != 0 or self.regions[0][1] != 0:
        #     raise RuntimeError("First region must start at (0,0)")
        # elif self.regions[0][2] < np.sum(self.regions[1:][2]) or self.regions[0][3] < np.sum(self.regions[1:][3]):
        #     raise RuntimeError("Background region must be longer than sum of all other regions")
        # elif self.regions[0][4] < np.sum(self.regions[1:][4]) or self.regions[0][5] < np.sum(self.regions[1:][5]):
        #     raise RuntimeError("Background region uses more grid points than sum of all other regions")
        self.Nx = self.domain[4]
        self.Ny = self.domain[5]

        self.q0 = np.zeros([self.Nx,self.Ny])
        self.sigmat = np.zeros([self.Nx,self.Ny])
        self.sigmas = np.zeros([self.Nx,self.Ny])

        if self.regiontype == "null": # One material throughout domain

            self.xsurface = np.linspace(self.domain[0],self.domain[0]+self.domain[2],self.domain[4]+1)
            self.ysurface = np.linspace(self.domain[1],self.domain[1]+self.domain[3],self.domain[5]+1)
            self.xcell = np.arange(self.domain[0], self.domain[2] + self.domain[0], self.domain[2] / self.domain[4])
            self.ycell = np.arange(self.domain[1], self.domain[3] + self.domain[1], self.domain[3] / self.domain[5])

            self.q0 = self.domain[8] * np.ones([self.Nx,self.Ny])
            self.sigmat = self.domain[6] * np.ones([self.Nx,self.Ny])
            self.sigmas = self.domain[7] * np.ones([self.Nx,self.Ny])

        if self.regiontype == "circ_lamp" or self.regiontype == "post": # Resolve a circular region in the middle 20% of the domain
            # Either an isotropic source or an absorber
            # absorbing medium

            xlamp1_center = 0.5 * self.domain[2]
            ylamp1_center = 0.5 * self.domain[3]
            lamp_radius = 0.1 * self.domain[2]

            # Resolve the area occupied by the circular source with half of the grid points
            left_points = self.Nx//4
            right_points = self.Nx//4 + 1
            side_points = left_points + right_points
            center_points = self.Nx//2
            if side_points + center_points != self.Nx + 1:
                # Add extra points to the sides
                extra_cells = self.Nx + 1 - (side_points + center_points)
                if np.mod(extra_cells,2) == 0:
                    left_points += extra_cells//2
                    right_points += extra_cells//2
                else:
                    left_points += extra_cells//2 + 1
                    right_points += extra_cells//2

            self.xsurface = np.hstack((np.linspace(0,0.4*self.domain[2],left_points+1)[:-1],
                                       np.linspace(0.4*self.domain[2],0.6*self.domain[2],center_points+1)[:-1],
                                       np.linspace(0.6*self.domain[2],self.domain[2],right_points)))
            self.ysurface = np.hstack((np.linspace(0,0.4*self.domain[3],left_points+1)[:-1],
                                       np.linspace(0.4*self.domain[3],0.6*self.domain[3],center_points+1)[:-1],
                                       np.linspace(0.6*self.domain[3],self.domain[3],right_points)))

            self.xcell = self.xsurface[:-1]
            self.ycell = self.ysurface[:-1]

            print(np.shape(self.xsurface),np.shape(self.ysurface),np.shape(self.xcell),np.shape(self.ycell))

            for icell,x in enumerate(self.xcell):
                for jcell,y in enumerate(self.ycell):

                    xtuples = [(x,y),(self.xsurface[icell+1],self.ysurface[jcell+1]),
                               (x,self.ysurface[jcell+1]),(self.xsurface[icell+1],y)]

                    if self.insidecircletest(xtuples,xlamp1_center,ylamp1_center,lamp_radius):
                        if self.regiontype == "circ_lamp":
                            self.sigmat[icell, jcell] = 1
                            self.sigmas[icell, jcell] = 0
                            self.q0[icell, jcell] = 1
                        else:
                            self.sigmat[icell, jcell] = 10
                            self.sigmas[icell, jcell] = 2
                            self.q0[icell, jcell] = 0
                    else:
                        self.sigmat[icell,jcell] = self.domain[6]
                        self.sigmas[icell,jcell] = self.domain[7]
                        self.q0[icell,jcell] = self.domain[8]

        if self.regiontype == "circ_lamp4": # Put four 10% domain size sources in each quadrant of domain, regular mesh

            if self.domain[4] < 20:
                raise RuntimeError("Not enough cells used to resolve four circular lamps")

            xlamp1_center = 0.25*self.domain[2]
            ylamp1_center = 0.25*self.domain[3]

            xlamp2_center = 0.25 * self.domain[2]
            ylamp2_center = 0.75 * self.domain[3]

            xlamp3_center = 0.75 * self.domain[2]
            ylamp3_center = 0.25 * self.domain[3]

            xlamp4_center = 0.75 * self.domain[2]
            ylamp4_center = 0.75 * self.domain[3]

            lamp_radius = 0.05 * self.domain[2]

            # Define xcell and ycell - still using cells from domain

            self.xsurface = np.linspace(self.domain[0], self.domain[0] + self.domain[2], self.domain[4] + 1)
            self.ysurface = np.linspace(self.domain[1], self.domain[1] + self.domain[3], self.domain[5] + 1)
            self.xcell = np.arange(self.domain[0], self.domain[2] + self.domain[0], self.domain[2] / self.domain[4])
            self.ycell = np.arange(self.domain[1], self.domain[3] + self.domain[1], self.domain[3] / self.domain[5])

            for icell,x in enumerate(self.xcell):
                for jcell,y in enumerate(self.ycell):

                    xtuples = [(x, y), (self.xsurface[icell + 1], self.ysurface[jcell + 1]),
                               (x, self.ysurface[jcell + 1]), (self.xsurface[icell + 1], y)]

                    if self.insidecircletest(xtuples,xlamp1_center,ylamp1_center,lamp_radius) \
                            or self.insidecircletest(xtuples,xlamp2_center,ylamp2_center,lamp_radius) \
                            or self.insidecircletest(xtuples,xlamp3_center,ylamp3_center,lamp_radius) \
                            or self.insidecircletest(xtuples,xlamp4_center,ylamp4_center,lamp_radius):
                        self.sigmat[icell, jcell] = 1
                        self.sigmas[icell, jcell] = 0
                        self.q0[icell, jcell] = 1
                    else:
                        self.sigmat[icell,jcell] = self.domain[6]
                        self.sigmas[icell,jcell] = self.domain[7]
                        self.q0[icell,jcell] = self.domain[8]

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

        elif self.boundarytype == "opendoor":
            if mu == np.amax(self.mus):
                abovecondition = self.ysurface > 0.35 * self.domain[3]
                belowcondition = self.ysurface < 0.65 * self.domain[3]
                a = np.nonzero(abovecondition*belowcondition)
                angle_vertical = np.zeros_like(self.ysurface)
                angle_vertical[a] = 1.0

        elif self.boundarytype == "full":
            angle_horizontal = self.fullboundary[(mu,phi)][0]
            angle_vertical = self.fullboundary[(mu,phi)][1]

        # Four cases : left, bottom face boundary; left, top; right, bottom; else

        if mu > 0 and np.cos(phi) > 0: # Set bottom and left boundaries
            self.angularflux_nodes[0,:] = angle_vertical
            self.angularflux_nodes[:,0] = angle_horizontal
        elif mu > 0 and np.cos(phi) < 0: # left, top
            self.angularflux_nodes[0,:] = angle_vertical
            self.angularflux_nodes[:,-1] = angle_horizontal
        elif mu < 0 and np.cos(phi) > 0: # right, bottom
            self.angularflux_nodes[-1,:] = angle_vertical
            self.angularflux_nodes[:,0] = angle_horizontal
        else:
            self.angularflux_nodes[-1,:] = angle_vertical
            self.angularflux_nodes[:,-1] = angle_horizontal

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
        
        self.rhs = (self.q0 + self.sigmas * self.scalarflux)/(2*np.pi)
        
    def src_iteration(self):
        
        error = 99
        iteration = 0
        
        # Source Iteration Loop
        time1 = perf_counter()
        while iteration < 10**5 and error > 10**(-10):
            
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

        print(np.shape(X))
        print(np.shape(Y))
        # Mesh plot
        plt.plot(X.flatten(),Y.flatten(),"ks",markersize=0.2)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Cell Southwest Nodes")
        if self.show:
            plt.show()
        plt.close()

        # Source plot
        plt.contourf(X,Y,self.q0,cmap="YlGnBu",vmin=0,vmax=2)
        plt.colorbar()
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Material Sources")
        if self.show:
            plt.show()
        plt.close()
        
        # Scattering Ratio Plot
        plt.contourf(X,Y,self.sigmas/self.sigmat,cmap="YlOrBr",vmin=0,vmax=1)
        plt.colorbar()
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Scattering Ratio of Geometry")
        if self.show:
            plt.show()
        plt.close()
        
    def resultplots(self):

        fig, ax = plt.subplots(1)
        X,Y = np.meshgrid(self.xcell,self.ycell,indexing="ij")
        
        pltsf = ax.contourf(X,Y,self.scalarflux,cmap="RdYlBu",levels=30)
        fig.colorbar(pltsf)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Scalar Flux "+str(self.Nx)+" Axis Cells "+str(self.Ntheta*self.Nphi) +" Ordinates")
        ax.set_aspect("equal")

        fig.savefig(self.fname+"scalarflux")
        if self.show:
            plt.show()
        plt.close()

        fig, ax = plt.subplots(1)
        pltcurx = ax.contourf(X,Y,self.currentx,cmap="RdYlBu",levels=30)
        fig.colorbar(pltcurx)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        fig.suptitle("X Current "+str(self.Nx)+" Axis Cells "+str(self.Ntheta*self.Nphi) +" Ordinates")
        fig.savefig(self.fname+"currentx")
        if self.show:
            plt.show()
        plt.close()

        fig, ax = plt.subplots(1)
        pltcury = ax.contourf(X,Y,self.currenty,cmap="RdYlBu",levels=30)
        fig.colorbar(pltcury)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        fig.suptitle("Y Current "+str(self.Nx)+" Axis Cells "+str(self.Ntheta*self.Nphi) +" Ordinates")
        fig.savefig(self.fname+"currenty")
        if self.show:
            plt.show()
        plt.close()

    def diffusionpreconditioner(self):

        raise NotImplementedError("Diffusion preconditioner not implemented")
