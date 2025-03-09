#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 22:50:23 2025

@author: ehansen
"""

from montecarlo1dsolver import MonteCarlo1DSolver

mc = MonteCarlo1DSolver([2,3], [1,2], [0,0], [0,0], 160,10**4, (1,1), (1,0), (0,0),
                        "testmultimat")

mc.simple_locationdirectionreflect()