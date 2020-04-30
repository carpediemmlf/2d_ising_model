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
size = 61
dimRange = 1
for i in range(dimRange):
    dim = dimLow +i
    name = "visualization_d=" + str(dim) + "_n=" + str(size)
    lowerTemperature = 50
    dataPoints = 10
    deltaTemperature = 20

    deltaH = np.power(10.0, -2.0)
    period = 10000

    steps = 10000

    _n = 31
    _delta = 32
    # not to be confused, this is the number of systems with different side length
    _size = 3

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
    name = name + "_lowerTemperature=" + str(lowerTemperature) + "_delta_H=" + str(deltaH) + "_period=" + str(period)
    # inquireMySys.hysteresis(temperature=lowerTemperature, delta_H=deltaH, period=period).savefig(path + "/" + name + "_hysteresis." + figureType)
    # plt.close()
    # print(datetime.now() - t)

    # inquire about properties
    data = [[], []]
    for i in range(dataPoints):
        a = inquireMySys.hysteresis(temperature=lowerTemperature + i * deltaTemperature , delta_H=deltaH, period=period)
        data[0].append(a[0])
        data[1].append(a[1])
    p = plt.figure()
    plt.plot(data[0], data[1], "+k")
    title = "Dissipation against temperature, dH/dt = 0.01T/au, period = 10000, n = 61"
    title = "\n".join(wrap(title, 60))
    plt.title(title)
    plt.xlabel("Temperature / K")
    plt.ylabel("Dissipation energy per site / T * A * m^2")
    p.savefig(path + "/" + name + "_hysteresis_dissipation." + figureType)
    plt.close()
    print(datetime.now() - t)
