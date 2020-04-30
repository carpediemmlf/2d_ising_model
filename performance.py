# import numpy as np
# from testCupyVectorizedIsing import *
from testVectorizedIsing import *
from datetime import *
from scipy import stats
from sigfig import round
# from matplotlib import animation
t = (datetime.now())

# output figure type
figureType = "png"

ys = []

stepSize = 1

# checkerboard
dimLow = 2
size = 3
dimRange = 1
singlemc = []
lattice = [] 
steps = 80
length = 60
for i in range(length):
    t = (datetime.now())
    
    mySys = Ising(name="testingNDIsing", N=size+i*stepSize, D=2, T=190)
    for _ in range(steps):
        mySys.stepForward()
    delta_t = ((datetime.now()) - t) / steps
    lattice.append(size+i)
    singlemc.append(delta_t.total_seconds())
lattice = np.array(lattice)
singlemc = np.array(singlemc)
p = plt.figure()
plt.plot(lattice, singlemc)
plt.xlabel("2d square lattice side length")
plt.ylabel("Single Monte Carlo Step / s")
plt.title("Checkerboard single MC step against 2d lattice size")
p.savefig("checkerboard_performance.png")

ys.append(singlemc)

# metropolis
from testModuleIsing import *

dimLow = 2
size = 3
dimRange = 1
singlemc = []
lattice = []
steps = 80
length = 60
for i in range(length):
    t = (datetime.now())

    mySys = Ising(name="testingNDIsing", N=size+i*stepSize, D=2, T=190)
    for _ in range(steps):
        mySys.stepForward()
    delta_t = ((datetime.now()) - t) / steps
    lattice.append(size+i)
    singlemc.append(delta_t.total_seconds())
lattice = np.array(lattice)
singlemc = np.array(singlemc)
p = plt.figure()
plt.plot(lattice, singlemc)
plt.xlabel("2d square lattice side length")
plt.ylabel("Single Monte Carlo Step / s")
plt.title("Metropolis single MC step against 2d lattice size")
p.savefig("metropolis_performance.png")

ys.append(singlemc)

# log comparison
p = plt.figure()
slope, intercept, r_value, p_value, std_err = stats.linregress(lattice,np.log10(ys[1]))
plt.plot(lattice, np.log10(ys[1]), label="Metropolis, y = " + str(round(float(slope), sigfigs=3)) + " * x + " + str(round(float(intercept), sigfigs=3)))
slope, intercept, r_value, p_value, std_err = stats.linregress(lattice,np.log10(ys[0]))
plt.plot(lattice, np.log10(ys[0]), label="Checkerboard, y = " + str(round(float(slope), sigfigs=3)) + " * x + " + str(round(float(intercept), sigfigs=3)))
plt.legend()
plt.xlabel("log10 (2d square lattice side length)")
plt.ylabel("log10 (Single Monte Carlo Step) / s")
plt.title("Metropolis and Checkerboard single MC step against 2d lattice size")
p.savefig("comparison_performance.png")

print((datetime.now()) - t)
