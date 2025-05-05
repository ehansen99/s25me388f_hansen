#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 22:50:23 2025

@author: ehansen
"""

from deterministic1dsolver import Ordinate1DSolver,Spectral1DSolver
from montecarlo1dsolver import MonteCarlo1DSolver
import numpy as np
import matplotlib.pyplot as plt

sourcereflectleft = False
if sourcereflectleft:
    ordinate = Ordinate1DSolver([25,5], [0.1,2], [0,1.8], [1,0], 640, 2, (1,0), (0,0), (0,0),
                       fname="testsourcereflectoraccspec",accelerator=2)
    ordinate.solve()
    plt.plot(ordinate.surfacemesh,ordinate.legmoments[1,:])
    plt.show()
    plt.close()

    montecarlo = MonteCarlo1DSolver([25,5], [0.1,2], [0,1.8], [1,0], 640, 4*10**4, 
                                 (1,0), (0,0), (0,0), "sourcereflector")

    montecarlo.simulation()

sourcereflectright = False
if sourcereflectright:
    ordinate = Ordinate1DSolver([5,25], [2,0.1], [1.8,0], [0,1], 640, 2, (0,1), (0,0), (0,0),
                                fname="testsourcereflectoraccspec",accelerator=2)
    ordinate.solve()
    plt.plot(ordinate.surfacemesh,ordinate.legmoments[1,:])
    plt.show()
    plt.close()


    montecarlo = MonteCarlo1DSolver([5,25], [2,0.1], [1.8,0], [0,1], 640, 4*10**4, 
                                    (0,1), (0,0), (0,0), "backsourcereflector")
    
    montecarlo.simulation()

scatterleft = False
if scatterleft:

    montecarlo2 = MonteCarlo1DSolver([10], [1], [1], [0], 640, 10**4, (1,1),
                                 (1,0), (0,0), "leftsource")
    montecarlo2.simulation()

    # plt.plot(np.log10(montecarlo1.scalarflux_distance),np.log10(montecarlo2.scalarflux_distance))
    # plt.show()
    # plt.close()
    
