# import numpy as np
# from testCupyVectorizedIsing import *
from testVectorizedIsing import *
from datetime import *
# from matplotlib import animation

# note: current parallelism only allows odd system sizes
size = 101
steps = 18000

# starting dimension, range of dimensions
dimLow = 2
dimRange = 1

# temperature
tempLow = 50  # Kelvins
deltaTemp = 5
numOfTemps = 1
# k = 1
# j = 1
for j in range(numOfTemps):
    for i in range(dimRange):
        dim = i + dimLow
        temp = tempLow + j * deltaTemp
        mySys = Ising(name="testingNDIsing", N=size, D=dim, T=temp) #H=np.power(1.0, -15))#, K=k, J=j)
        # print spec
        print("N = " + str(size) + ", D = " + str(dim))
        # print(mySys.evenLattice)
        # print(mySys.oddLattice)
        # print data shape
        # print(mySys.system.shape)
        t = (datetime.now())
        # print(t)
        # mySys.stepForward()
        # plot the initial state
        name = "visualization_d=" + str(dim) + "_n=" + str(size) + "_t=" + str(temp)
        # os.mkdir(os.getcwd() + "/" + name)
        path = os.getcwd() + "/" + name
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)
        # generate hyperplane for plotting
        hyperplane = [slice(mySys.n),slice(mySys.n)]
        for j in range(dim-2):
            hyperplane.append([0])
        hyperplane = tuple(hyperplane)

        # mySys.stepForward()
        # print("Odd Lattice")
        # print(mySys.oddLattice)
        # print("Even Lattice")
        # print(mySys.evenLattice)
        # imgs = []
        # mySys.visualizeTwoDGrid("pre_" + name, hyperplane)
        for l in range(steps):
            # mySys.visualizeTwoDGrid(hyperplane=hyperplane).savefig(path + "/" + str(l) + "_" + name)
            # plt.close()
            mySys.stepForward()
        mySys.visualizeMagnetizationPhaseSpace(name, hyperplane).savefig(path + "/" + str(l) + "_" + name + "_phasespace")
        plt.close()
        mySys.visualizeMagnetizationAutocovariance(name, hyperplane).savefig(path + "/" + str(l) + "_" + name + "_autocovariance")
        plt.close()
        mySys.visualizeMagnetization(name, hyperplane).savefig(path + "/" + str(l) + "_" + name + "_magnetization")
        plt.close()
        mySys.visualizeTotalEnergy(name, hyperplane).savefig(path + "/" + str(l) + "_" + name + "_total_energy") 
        plt.close()
        mySys.visualizeTwoDGrid(hyperplane=hyperplane).savefig(path + "/" + str(l) + "_" + name)
        plt.close()
        print(datetime.now() - t)
