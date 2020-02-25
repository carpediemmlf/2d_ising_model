# import numpy as np
# from testCupyVectorizedIsing import *
from testVectorizedIsing import *
from datetime import *
# from matplotlib import animation

# note: current parallelism only allows odd system sizes
size = 31
steps = 5000

# starting dimension, range of dimensions
dimLow = 2
dimRange = 1

# output figure type
figureType = "png"
# temperature
tempLow = 5  # Kelvins
deltaTemp = 40
numOfTemps = 7
# k = 1
# j = 1
for j in range(numOfTemps):
    for i in range(dimRange):
        dim = i + dimLow
        temp = tempLow + j * deltaTemp
        mySys = Ising(name="testingNDIsing", N=size, D=dim, T=temp) #H=np.power(1.0, -15))#, K=k, J=j)
        # print spec
        print("N = " + str(size) + ", D = " + str(dim))
        t = (datetime.now())
        name = "visualization_d=" + str(dim) + "_n=" + str(size) + "_t=" + str(temp)
        # create figure storage folder
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

        for l in range(steps):
            # mySys.visualizeTwoDGrid(hyperplane=hyperplane).savefig(path + "/" + str(l) + "_" + name)
            # plt.close()
            mySys.stepForward()
        # warning: use phasespace to estimate equilibriate time needed before plotting other plots
        mySys.visualizeMagnetizationPhaseSpace(name, hyperplane).savefig(path + "/" + str(l) + "_" + name + "_phasespace." + figureType)
        plt.close()
        mySys.visualizeMagnetizationAutocovariance(name, hyperplane).savefig(path + "/" + str(l) + "_" + name + "_autocovariance." + figureType)
        plt.close()
        mySys.visualizeMagnetization(name, hyperplane).savefig(path + "/" + str(l) + "_" + name + "_magnetization." + figureType)
        plt.close()
        mySys.visualizeTotalEnergy(name, hyperplane).savefig(path + "/" + str(l) + "_" + name + "_total_energy." + figureType)
        plt.close()
        mySys.visualizeTwoDGrid(hyperplane=hyperplane).savefig(path + "/" + str(l) + "_" + name + "." + figureType)
        plt.close()
        print(datetime.now() - t)
