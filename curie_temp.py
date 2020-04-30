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
size = 31
dimRange = 1
for i in range(dimRange):
    dim = dimLow +i
    name = "visualization_d=" + str(dim) + "_n=" + str(size)
    lowerTemperature = 199
    dataPoints = 10
    deltaTemperature = 0.6
    steps = 10000

    _n = 11
    _delta = 8
    # not to be confused, this is the number of systems with different side length
    _size = 4

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
    print(inquireMySys.numberOfMeasurements)
    name = name + "_lowerTemperature=" + str(lowerTemperature) + "_deltaTemperature=" + str(deltaTemperature) + "_dataPoints=" + str(dataPoints)

    # inquire about properties
    inquireMySys.visualizeCurieTemperatureAgainstTime(n=_n, delta=_delta, length_eles=_size, tLow=lowerTemperature, accuracy=deltaTemperature, temp_eles=dataPoints).savefig(path + "/" + name + "_curie_temperatures." + figureType)
    plt.close()
    print(datetime.now() - t)
