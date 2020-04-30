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

dimLow = 2
size = 41
dimRange = 1
for i in range(dimRange):
    dim = dimLow +i
    name = "visualization_d=" + str(dim) + "_n=" + str(size)
    lowerTemperature = 50
    dataPoints = 40
    deltaTemperature = 5
    steps = 5000
    numberOfMeasurements = 4
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
    print(inquireMySys.numberOfMeasurements)
    name = name + "_lowerTemperature=" + str(lowerTemperature) + "_deltaTemperature=" + str(deltaTemperature) + "_dataPoints=" + str(dataPoints)

    inquireMySys.visualizeEquilibriumTime().savefig(path + "/" + name + "_equilibrium_times." + figureType)

    # inquire about properties
    # inquireMySys.visualizeBinderCumulant().savefig(path + "/" + name + "_binder_cumulants." + figureType)
    # inquireMySys.visualizeStationaryMagnetization().savefig(path + "/" + name + "_stationary_magnetizations." + figureType)
    # inquireMySys.visualizeCorrelationTime().savefig(path + "/" + name + "_correlation_times." + figureType)
    # inquireMySys.visualizeSpecificHeatiCapacityPerSite().savefig(path + "/" + name + "_specific_heat_capacities_per_site." + figureType)
    plt.close()
    print(datetime.now() - t)
