# import numpy as np
# from testCupyVectorizedIsing import *
from testVectorizedIsing import *
from datetime import *
# from matplotlib import animation

# note: current parallelism only allows odd system sizes

# output figure type
figureType = "png"

# starting dimension, range of dimensions
dimLow = 2
size = 15
dimRange = 1
for i in range(dimRange):
    dim = dimLow +i
    name = "visualization_d=" + str(dim) + "_n=" + str(size)
    lowerTemperature = 190
    dataPoints = 3
    deltaTemperature = 1
    steps = 20000

    _n = 11
    _delta = 4
    # not to be confused, this is the number of systems with different side length
    _size = 3

    numberOfMeasurements = 16
    path = os.getcwd() + "/" + name
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    t = (datetime.now())
    inquireMySys = InquireIsing(name="testingNDIsing", N=size, D=dim, numberOfMeasurements=numberOfMeasurements, dataPoints=dataPoints, lowerTemperature=lowerTemperature, deltaTemperature=deltaTemperature, steps=steps)
    print(inquireMySys.numberOfMeasurements)
    name = name + "_lowerTemperature=" + str(lowerTemperature) + "_deltaTemperature=" + str(deltaTemperature) + "_dataPoints=" + str(dataPoints)

    # inquire about properties
    # inquireMySys.visualizeBinderCumulant().savefig(path + "/" + name + "_binder_cumulants." + figureType)
    # inquireMySys.visualizeStationaryMagnetization().savefig(path + "/" + name + "_stationary_magnetizations." + figureType)
    inquireMySys.visualizeCorrelationTime(n=_n, delta=_delta, size=_size, ).savefig(path + "/" + name + "_correlation_times." + figureType)
    # inquireMySys.visualizeSpecificHeatiCapacityPerSite().savefig(path + "/" + name + "_specific_heat_capacities_per_site." + figureType)
    plt.close()
    print(datetime.now() - t)
