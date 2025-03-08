#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 22:50:23 2025

@author: ehansen
"""

from montecarlo1dsolver import MonteCarlo1DSolver

mc = MonteCarlo1DSolver([100], [1], [0], [1], 160,10**4, (0,0), (0,0), (0,0), 
                        "reflecttest")

mc.simple_locationdirectionreflect()