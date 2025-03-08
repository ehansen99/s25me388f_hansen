#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 22:50:23 2025

@author: ehansen
"""

from montecarlo1dsolver import MonteCarlo1DSolver

mc = MonteCarlo1DSolver([100], [1], [0], [1], 160,10**5, (1,1), (0,0), (0,0), 
                        "testreflect4")

mc.simple_locationdirectionreflect(1)