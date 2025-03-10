#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 22:50:23 2025

@author: ehansen
"""

from deterministic1dsolver import Ordinate1DSolver,Spectral1DSolver
from montecarlo1dsolver import MonteCarlo1DSolver

ordinate = Spectral1DSolver([50,10], [0.1,2], [0,1.8], [1,0], 10240, 2, (1,0), (0,0), (0,0),
                       fname="testsourcereflectoraccspec",accelerator=2)

ordinate.solve()

ordinate = Ordinate1DSolver([20,80], [10,0.01],[2,0.006], [0,0], 10240, 64, (1,1), (0,0), (0,1),
                       fname="testabsorbairaccord",accelerator=2)

ordinate.solve()

# mc = MonteCarlo1DSolver([100], [1], [1], [0], 100,10**3, (1,1), (1,0), (0,0),
#                         "testmultimatreflect")



# mc.simulation()
# mc.simulation()