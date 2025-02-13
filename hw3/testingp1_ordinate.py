#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:51:51 2025

@author: ehansen
"""


import numpy as np
from scipy.linalg import solve_banded,solve
from scipy.special import roots_legendre
import matplotlib.pyplot as plt


sigmat = 1.0
sigmas = 0.1
length = 100
Nx = 10
dx = length/Nx
Nmu = 2
q0 = 1

mus,weight = roots_legendre(2)

# Use Bell and Glasstone equation, and use average of angular fluxes at edges
# for value within the cell
# (This is essentially a diamond difference or continuous finite element method
# but the algorithm is unstable without the average)

# mu (psi(mu,i+1)-psi(mu,i))/(L/N) + sigmat * (psi(mu,i+1)+psi(mu,i))/2 = q0 + sigmas * phi
# So psi(mu,i+1)*(mu/dx + sigmat/2) + psi(mu,i) *(sigmat/2 - mu/dx) = q0 + sigmas*phi

 
psi = np.zeros([Nmu*(Nx+1)])

matrix = np.zeros([Nmu*(Nx+1),Nmu*(Nx+1)])


b = psi.copy()
b2 = psi.copy()
x = psi.copy()
y = psi.copy()

# Reflecting boundary conditions for starting point
matrix[0,1] = 1
matrix[0,0] = -1
b[0] = 0

# Equations - take dx to infinity limit for now
# My first attempt was singular without the derivative terms

# for i in range(Nx):
#     matrix[1+Nmu*i,i*Nmu] = sigmat/2 - mus[0]/dx
#     matrix[1+Nmu*i,i*Nmu+2] = sigmat/2 + mus[0]/dx
#     matrix[2+Nmu*i,i*Nmu+1] = sigmat/2 - mus[1]/dx
#     matrix[2+Nmu*i,i*Nmu+3] = sigmat/2 + mus[1]/dx
#     b[1+Nmu*i] = q0
#     b[2+Nmu*i] = q0

# # Last equation - reflecting boundary condition at end
# matrix[1+Nmu*Nx,-1] = 1
# matrix[1+Nmu*Nx,-2] = -1
# b[-1 ] = 0

scalarflux = np.zeros(Nx)
oldflux = scalarflux.copy()+10

def update_scalarflux(x):
    # Integrate using quadrature in the two points and
    scalarflux = np.zeros(Nx)
    for i in range(Nx):
        scalarflux[i] = (np.sum(weight*x[i*Nmu:(i+1)*Nmu]) + np.sum(weight*x[(i+1)*Nmu:(i+2)*Nmu]))/2
    return(scalarflux)

# print(matrix)

# i = 0
# while np.amax(np.abs(scalarflux-oldflux)) > 10**(-10):
#     oldflux = scalarflux
#     x = solve(matrix,b)
    
#     scalarflux = update_scalarflux(x)
#     b[1:-1:2] = q0 + sigmas*scalarflux
#     b[2:-1:2] = q0 + sigmas*scalarflux
#     b2[1:-1:2] = q0 + sigmas*scalarflux
#     b2[2:-1:2] = q0 + sigmas*scalarflux
    
#     print("Iteration ",i," Difference ", np.amax(np.abs(scalarflux-oldflux)))
#     i += 1 
    
# plt.plot(np.arange(11)*length/10,x[1::2],"b",label="Positive")
# plt.plot(np.arange(11)*length/10,x[0::2],"r:",label="Negative")

# plt.xlabel("x position (cm)")
# plt.ylabel("Angular Flux (neutron/(cm^2 mu MeV s))")
# plt.title("Angular Fluxes in Homogeneous Media")
# if np.amax(scalarflux < 3):
#     plt.ylim(-3,3)
# plt.legend()
# plt.show()


# Repeat the problem but using a banded matrix

b[:Nmu//2] = 0
b[-Nmu//2:] = 0
b[Nmu//2:-Nmu//2] = q0/2

# Need Nmu//2 upper diagonals, Nmu//2 lower diagonals
AB = np.zeros([Nmu+1,Nmu*(Nx+1)])
for i in range(0,Nx):
    AB[0,Nmu+i*Nmu:Nmu+(i+1)*Nmu] = sigmat/2 + mus/dx
    AB[Nmu,i*Nmu:(i+1)*Nmu] = sigmat/2 - mus/dx
# Boundary conditions - we'll generalize reflecting later
AB[0,Nmu//2] = 1
AB[1,0] = -1
AB[1,-1] = -1
AB[2,-2] = 1

print("Matrix \n")
print(AB)

i = 0
print("Solutions \n")
while np.amax(np.abs(scalarflux-oldflux)) > 10**(-10):
    oldflux = scalarflux
    x = solve_banded((Nmu//2,Nmu//2),AB,b)
    print(x)
    
    scalarflux = update_scalarflux(x)
    b[1:-1:2] = (q0/2 + sigmas*scalarflux/2)
    b[2:-1:2] = (q0/2 + sigmas*scalarflux/2)
    
    print("Iteration ",i," Difference ", np.amax(np.abs(scalarflux-oldflux)))
    i += 1
    
plt.plot(np.arange(Nx+1)*length/Nx,x[1::2],"b",label="Positive")
plt.plot(np.arange(Nx+1)*length/Nx,x[0::2],"r:",label="Negative")
plt.plot(dx/2+np.arange(Nx)*length/Nx,scalarflux,"k",label="Scalar Flux")

plt.xlabel("x position (cm)")
plt.ylabel("Angular Flux (neutron/(cm^2 mu MeV s))")
plt.title("Angular Fluxes in Homogeneous Media")
if np.amax(scalarflux < 3):
    plt.ylim(-3,3)
plt.legend()
plt.show()
    