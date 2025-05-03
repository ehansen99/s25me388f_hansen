from rectangle2dsn import DiamondDifference2D

# Isotropic Source for Vacuum Boundary
region = [(0,0,10,10,100,100,1,0.9,1)]

solver = DiamondDifference2D(region,"vacuum",16,16,accelerator=None,fullboundary=None)
solver.src_iteration()

# Small Source in Middle

# Four Rods

#