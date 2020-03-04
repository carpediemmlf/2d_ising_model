# import numpy as np
# from testCupyVectorizedIsing import *
from testVectorizedIsing import *
from datetime import *
# from matplotlib import animation

# note: current parallelism only allows odd system sizes
size = 51
steps = 30000

# starting dimension, range of dimensions
dimLow = 2
dimRange = 1

# output figure type
figureType = "png"
# temperature
tempLow = 206 # Kelvins
deltaTemp = 40
numOfTemps = 1
# k = 1
# j = 1
"""
for j in range(numOfTemps):
    for i in range(dimRange):
        dim = i + dimLow
        temp = tempLow + j * deltaTemp
        mySys = Ising(name="testingNDIsing", N=size, D=dim, T=temp) #H=np.power(1.0, -15))#, K=k, J=j)
        # inquireMySys = InquireIsing(name="testingNDIsing", N=size, D=dim, T=temp, numberOfMeasurements=15)
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

"""
dimLow = 2
size = 11
dimRange = 1
for i in range(dimRange):
    dim = dimLow +i
    name = "visualization_d=" + str(dim) + "_n=" + str(size)
    lowerTemperature = 20
    dataPoints = 10
    deltaTemperature = 0.1
    steps = 10000
    numberOfMeasurements = 3
    path = os.getcwd() + "/" + name
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    t = (datetime.now())
    inquireMySys = InquireIsing(name="testingNDIsing", N=size, D=dim, numberOfMeasurements=numberOfMeasurements, dataPoints=dataPoints, lowerTemperature=lowerTemperature, deltaTemperature=deltaTemperature, steps=steps)
    # mySys = Ising(name="testingNDIsing", N=size, D=dim, numberOfMeasurements=numberOfMeasurements)
    # for i in range(1000):
    #     mySys.stepForward()
    name = name + "_lowerTemperature=" + str(lowerTemperature) + "_deltaTemperature=" + str(deltaTemperature) + "_dataPoints=" + str(dataPoints)

    # inquire about properties
    inquireMySys.visualizeBinderCumulant().savefig(path + "/" + name + "_binder_cumulants." + figureType)
    # inquireMySys.visualizeStationaryMagnetization().savefig(path + "/" + name + "_stationary_magnetizations." + figureType)
    # inquireMySys.visualizeCorrelationTime().savefig(path + "/" + name + "_correlation_times." + figureType)
    plt.close()
    print(datetime.now() - t)
