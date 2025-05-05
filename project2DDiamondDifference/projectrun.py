from rectangle2dsn import DiamondDifference2D
import numpy as np
import matplotlib.pyplot as plt
import os

Nx = 50
Ny = 50
Ntheta = 16
Nphi = 16

def isosource(Nx,Ny,Ntheta,Nphi,prefix=""):
    # domain arbitrary scattering material for test
    domain = (0, 0, 10, 10, Nx, Ny, 1, 0.9999, 1)
    solver = DiamondDifference2D(domain, "null", "vacuum", Ntheta, Nphi, accelerator=None, fullboundary=None,
                                 fname=prefix+"vacuumisource"+str(Nx)+str(Ny)+str(Ntheta)+str(Nphi),show=True)
    solver.src_iteration()

    return(solver.scalarflux)

def opendoor(Nx,Ny,Ntheta,Nphi,prefix=""):
    # domain air
    domain = (0, 0, 20, 10, Nx, Ny, 0.01, 0.006, 0)
    solver = DiamondDifference2D(domain, "null", "opendoor", Ntheta, Nphi, accelerator=None, fullboundary=None,
                                 fname=prefix+"opendoor"+str(Nx)+str(Ny)+str(Ntheta)+str(Nphi))
    solver.src_iteration()

    return(solver.scalarflux)

def opendoorpost(Nx,Ny,Ntheta,Nphi,prefix=""):
    # domain air
    domain = (0, 0, 20, 10, Nx, Ny, 0.01, 0.006, 0)
    solver = DiamondDifference2D(domain, "post", "opendoor", Ntheta, Nphi, accelerator=None, fullboundary=None,
                                 fname=prefix+"opendoorpost"+str(Nx)+str(Ny)+str(Ntheta)+str(Nphi))
    solver.src_iteration()

    return(solver.scalarflux)

def lamp(Nx,Ny,Ntheta,Nphi,prefix=""):
    # domain absorber
    domain = (0, 0,1,1,Nx,Ny,1,0.2,0)
    solver = DiamondDifference2D(domain, "circ_lamp", "vacuum", Ntheta, Nphi, accelerator=None, fullboundary=None,
                                 fname=prefix+"lamp"+str(Nx)+str(Ny)+str(Ntheta)+str(Nphi))
    solver.src_iteration()

    return(solver.scalarflux)

def lamp4(Nx,Ny,Ntheta,Nphi,prefix=""):
    # domain absorber
    domain = (0, 0, 1, 1, Nx, Ny, 1, 0.2, 0)
    solver = DiamondDifference2D(domain, "circ_lamp4", "vacuum", Ntheta, Nphi, accelerator=None, fullboundary=None,
                                 fname=prefix+"lamp4"+str(Nx)+str(Ny)+str(Ntheta)+str(Nphi))
    solver.src_iteration()

    return(solver.scalarflux)

def uprightfluxblock(Nx,Ny,Ntheta,Nphi,prefix=""):

    domain = (0, 0, 20, 10, Nx, Ny, 0.01, 0.006, 0)
    solver = DiamondDifference2D(domain, "post", "upright", Ntheta, Nphi, accelerator=None, fullboundary=None,
                                 fname=prefix + "uprightfluxblock" + str(Nx) + str(Ny) + str(Ntheta) + str(Nphi))
    solver.src_iteration()

    return(solver.scalarflux)

Nspacetest = [40,60,80]
Nangletest = [2,4,8]

isosrcspacediffs = []
lamp4spacediffs = []
opendoorspacediffs = []
opendoorpostspacediffs = []

isosrc60mudiffs = []
lamp460mudiffs = []
opendoor60mudiffs = []
opendoorpost60mudiffs = []

isosrc120mudiffs = []
lamp4120mudiffs = []
opendoor120mudiffs = []
opendoorpost120mudiffs = []

isosrc60phidiffs = []
lamp460phidiffs = []
opendoor60phidiffs = []
opendoorpost60phidiffs = []

isosrc120phidiffs = []
lamp4120phidiffs = []
opendoor120phidiffs = []
opendoorpost120phidiffs = []

def rayeffects():
    opendoor(120,120,4,4,"rayfx")
    lamp4(120,120,4,4,"rayfx")
    lamp(120,120,4,4,"rayfx")

    opendoor(120, 120, 16, 16, "rayfx")
    lamp4(120, 120, 16, 16, "rayfx")
    lamp(120, 120, 16, 16, "rayfx")



def convergence():

    Nxbest = 120
    Nybest = 120
    Nthetabest = 16
    Nphibest = 16

    # Best converged solutions to problems
    if False: #os.path.exists("bestsolutions.npz"):
        prevresults = np.load("bestsolutions.npz")

        isosourcebest = prevresults["isb120"]
        lamp4best = prevresults["lamp4120"]
        opendoorbest = prevresults["od120"]
        opendoorpostbest = prevresults["odp120"]

        isosource60best = prevresults["isb60"]
        lamp460best = prevresults["lamp460"]
        opendoorpost60best = prevresults["od60"]
        opendoor60best = prevresults["odp60"]
    else:

        isosourcebest = isosource(Nxbest,Nybest,Nthetabest,Nphibest)
        lamp4best = lamp4(Nxbest,Nybest,Nthetabest,Nphibest)
        opendoorpostbest = opendoorpost(Nxbest,Nybest,Nthetabest,Nphibest)
        opendoorbest = opendoor(Nxbest,Nybest,Nthetabest,Nphibest)

        isosource60best = isosource(60, 60, Nthetabest, Nphibest)
        lamp460best = lamp4(60, 60, Nthetabest, Nphibest)
        opendoorpost60best = opendoorpost(60, 60, Nthetabest, Nphibest)
        opendoor60best = opendoor(60, 60, Nthetabest, Nphibest)

        np.savez("bestsolutions.npz",isb120=isosourcebest,lamp4120=lamp4best,od120=opendoorbest,odp120=opendoorpostbest,
                 isb60=isosource60best,lamp460=lamp460best,od60=opendoor60best,odp60=opendoorpost60best)



    # Test spatial convergence
    for n in Nspacetest:

        isosrcn = isosource(n,n,Nthetabest,Nphibest)
        lamp4n = lamp4(n,n,Nthetabest,Nphibest)
        opendoorn = opendoor(n,n,Nthetabest,Nphibest)
        opendoorpostn = opendoorpost(n,n,Nthetabest,Nphibest)

        # Calculate L2 difference between points where meshes overlap
        if n == 40:

            isosrcspacediffs.append(np.sqrt(np.sum((isosrcn - isosourcebest[::3,::3])**2))/np.size(isosrcn))
            lamp4spacediffs.append(np.sqrt(np.sum((lamp4n - lamp4best[::3, ::3]) ** 2)) / np.size(lamp4n))
            opendoorspacediffs.append(np.sqrt(np.sum((opendoorn - opendoorbest[::3, ::3]) ** 2)) / np.size(opendoorn))
            opendoorpostspacediffs.append(np.sqrt(np.sum((opendoorpostn - opendoorpostbest[::3, ::3]) ** 2)) / np.size(opendoorpostn))

        if n == 60:
            isosrcspacediffs.append(np.sqrt(np.sum((isosrcn - isosourcebest[::2, ::2]) ** 2)) / np.size(isosrcn))
            lamp4spacediffs.append(np.sqrt(np.sum((lamp4n - lamp4best[::2, ::2]) ** 2)) / np.size(lamp4n))
            opendoorspacediffs.append(
                np.sqrt(np.sum((opendoorn - opendoorbest[::2, ::2]) ** 2)) / np.size(opendoorn))
            opendoorpostspacediffs.append(
                np.sqrt(np.sum((opendoorpostn - opendoorpostbest[::2, ::2]) ** 2)) / np.size(opendoorpostn))

        if n == 80:
            isosrcspacediffs.append(4*np.sqrt(np.sum((isosrcn[::2, ::2] - isosourcebest[::3, ::3]) ** 2)) / np.size(isosrcn))
            lamp4spacediffs.append(4*np.sqrt(np.sum((lamp4n[::2, ::2] - lamp4best[::3, ::3]) ** 2)) / np.size(lamp4n))
            opendoorspacediffs.append(
                4*np.sqrt(np.sum((opendoorn[::2, ::2] - opendoorbest[::3, ::3]) ** 2)) / np.size(opendoorn))
            opendoorpostspacediffs.append(
                4*np.sqrt(np.sum((opendoorpostn[::2, ::2] - opendoorpostbest[::3, ::3]) ** 2)) / np.size(opendoorpostn))

    for n in [120]:
        for na in Nangletest:
            isosrcphin = isosource(n,n,Nthetabest,na)
            lamp4phin = lamp4(n,n,Nthetabest,na)
            opendoorphin = opendoor(n,n,Nthetabest,na)
            opendoorpostphin = opendoorpost(n,n,Nthetabest,na)

            isosrcmun = isosource(n,n,na,Nphibest)
            lamp4mun = lamp4(n,n,na,Nphibest)
            opendoormun = opendoor(n,n,na,Nphibest)
            opendoorpostmun = opendoorpost(n,n,na,Nphibest)

            # Compute L2 difference between scalar fluxes
            if n == 60:
                isosrc60mudiffs.append(np.sqrt(np.sum((isosrcmun-isosource60best)**2))/np.size(isosrcmun))
                lamp460mudiffs.append(np.sqrt(np.sum((lamp4mun-lamp460best)**2))/np.size(lamp4mun))
                opendoor60mudiffs.append(np.sqrt(np.sum((opendoormun-opendoor60best)**2))/np.size(opendoormun))
                opendoorpost60mudiffs.append(np.sqrt(np.sum((opendoorpostmun-opendoorpost60best)**2))/np.size(opendoorpostmun))

                isosrc60phidiffs.append(np.sqrt(np.sum((isosrcphin-isosource60best)**2))/np.size(isosrcphin))
                lamp460phidiffs.append(np.sqrt(np.sum((lamp4phin-lamp460best)**2))/np.size(lamp4phin))
                opendoor60phidiffs.append(np.sqrt(np.sum((opendoorphin-opendoor60best)**2))/np.size(opendoorphin))
                opendoorpost60phidiffs.append(np.sqrt(np.sum((opendoorpostphin-opendoorpost60best)**2))/np.size(opendoorpostphin))

            else:

                isosrc120mudiffs.append(np.sqrt(np.sum((isosrcmun - isosourcebest) ** 2)) / np.size(isosrcmun))
                lamp4120mudiffs.append(np.sqrt(np.sum((lamp4mun - lamp4best) ** 2)) / np.size(lamp4mun))
                opendoor120mudiffs.append(np.sqrt(np.sum((opendoormun - opendoorbest) ** 2)) / np.size(opendoormun))
                opendoorpost120mudiffs.append(
                    np.sqrt(np.sum((opendoorpostmun - opendoorpostbest) ** 2)) / np.size(opendoorpostmun))

                isosrc120phidiffs.append(np.sqrt(np.sum((isosrcphin - isosourcebest) ** 2)) / np.size(isosrcphin))
                lamp4120phidiffs.append(np.sqrt(np.sum((lamp4phin - lamp4best) ** 2)) / np.size(lamp4phin))
                opendoor120phidiffs.append(np.sqrt(np.sum((opendoorphin - opendoorbest) ** 2)) / np.size(opendoorphin))
                opendoorpost120phidiffs.append(
                    np.sqrt(np.sum((opendoorpostphin - opendoorpostbest) ** 2)) / np.size(opendoorpostphin))


    plt.plot(Nspacetest,isosrcspacediffs,"rs",label="Isotropic Source")
    plt.plot(Nspacetest,lamp4spacediffs,"ro",label="Four Sources")
    plt.plot(Nspacetest,opendoorspacediffs,"bs",label="Incoming Flux")
    plt.plot(Nspacetest,opendoorpostspacediffs,"bo",label="Incoming Flux Barrier")
    plt.xlabel("Number of Spatial Points")
    plt.ylabel("L2 Average Difference")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.title("Spatial Convergence of 2D Transport Problems")
    plt.savefig("spatialconvergence")
    plt.show()
    plt.close()

    # plt.plot(Nangletest, isosrc60mudiffs, "rs", label="Isotropic Source")
    # plt.plot(Nangletest, lamp460mudiffs, "ro", label="Four Sources")
    # plt.plot(Nangletest, opendoor60mudiffs, "bs", label="Incoming Flux")
    # plt.plot(Nangletest, opendoorpost60mudiffs, "bo", label="Incoming Flux Barrier")
    # plt.xlabel("Number of mu Values")
    # plt.ylabel("L2 Average Difference")
    # plt.yscale("log")
    # plt.legend(loc="upper right")
    # plt.title("Convergence in Mu of 2D Transport Problems at 60 Cells per Axis")
    # plt.savefig("muconvergence60")
    # plt.show()
    # plt.close()

    plt.plot(Nangletest, isosrc120mudiffs, "rs", label="Isotropic Source")
    plt.plot(Nangletest, lamp4120mudiffs, "ro", label="Four Sources")
    plt.plot(Nangletest, opendoor120mudiffs, "bs", label="Incoming Flux")
    plt.plot(Nangletest, opendoorpost120mudiffs, "bo", label="Incoming Flux Barrier")
    plt.xlabel("Number of mu Values")
    plt.ylabel("L2 Average Difference")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.title("Convergence in Mu of 2D Transport Problems at 120 Cells per Axis")
    plt.savefig("muconvergence60")
    plt.show()
    plt.close()

    # plt.plot(Nangletest, isosrc60phidiffs, "rs", label="Isotropic Source")
    # plt.plot(Nangletest, lamp460phidiffs, "ro", label="Four Sources")
    # plt.plot(Nangletest, opendoor60phidiffs, "bs", label="Incoming Flux")
    # plt.plot(Nangletest, opendoorpost60phidiffs, "bo", label="Incoming Flux Barrier")
    # plt.xlabel("Number of phi Values")
    # plt.ylabel("L2 Average Difference")
    # plt.yscale("log")
    # plt.legend(loc="upper right")
    # plt.title("Convergence in Phi of 2D Transport Problems at 60 Cells per Axis")
    # plt.savefig("phiconvergence60")
    # plt.show()
    # plt.close()

    plt.plot(Nangletest, isosrc120phidiffs, "rs", label="Isotropic Source")
    plt.plot(Nangletest, lamp4120phidiffs, "ro", label="Four Sources")
    plt.plot(Nangletest, opendoor120phidiffs, "bs", label="Incoming Flux")
    plt.plot(Nangletest, opendoorpost120phidiffs, "bo", label="Incoming Flux Barrier")
    plt.xlabel("Number of phi Values")
    plt.ylabel("L2 Average Difference")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.title("Convergence in Phi of 2D Transport Problems at 120 Cells per Axis")
    plt.savefig("phiconvergence120")
    plt.show()
    plt.close()

isosource(80,80,8,8)
#opendoorpost(60,60,16,16,"diffusion")
#uprightfluxblock(60,60,16,16,"diffusion")
#rayeffects()
#convergence()