# common computing and statistical packages
import numpy as np
# from numpy.random import randint, uniform
# import scipy as sp
from scipy.ndimage import convolve, generate_binary_structure
import statsmodels.tsa.stattools
import random
import time
# import matplotlib.pyplot as pyplot
import copy

# a powerful plotter package for jupyter-notebook
# import bokeh
# from bokeh.plotting import figure, output_file, show, save
# from bokeh.io import output_notebook

# extend the cell width of the jupyter notebook
# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))

# paralell computing
# import ipyparallel
# import socket
import os  # writing to path
import sigfig  # rounding for plots
from textwrap import wrap
# from mpi4py import MPI

# save temporary data
# import csv

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
# from matplotlib.animation import FuncAnimation
# output_notebook()

# -------------------------------------------------------------------------------

# SI Units ommited


class Ising:
    # iron from https://www.southampton.ac.uk/~rpb/thesis/node18.html
    def __init__(self, name="none", N=100, H=0, T=273.15, D=2, J=1.21 * np.power(10.0, -21), randomFill=True, K=1.38064852 * np.power(10.0, -23), M=2.22 * np.power(10.0, (-23))):
        self.removeUnderflow = np.power(10, 0)
        self.j = np.longdouble(J)  # / self.removeUnderflow  # 'numerical' coupling constant
        self.h = np.longdouble(H)  # external field strength
        self.n = N  # lattice size
        self.m = np.longdouble(M)  # single spin magnetic moment
        self.k = np.longdouble(K)  # boltzman constant
        self.t = np.longdouble(T)  # / self.removeUnderflow # temperature in kelvins
        self.d = D  # system dimension
        self.tc = 2*J/(K*np.arcsinh(1))  # theoretical tc for 2d by onsanger
        self.timeStep = 0  # initialize timestep marker
        # working
        self.kernel = generate_binary_structure(self.d, 1)
        self.ground = np.full(np.repeat(self.n, self.d), 0)
        # storage of system physical property time series: time stamp, total magnetization, total energy,
        self.systemDataTimeSeries = [[], [], []]

        # define the system with or without initial values
        spins = [1, -1]
        if randomFill:
            self.system = np.random.choice(spins, tuple(np.repeat(self.n, self.d)))
        else:
            # dangerously, for future importing of existing system
            self.system = np.empty(tuple(np.repeat(self.n, self.d)))
        self.system = np.asarray(self.system)
        choices = list(range(self.n**self.d))

        # warning: optimization only works in odd sized dimensions!!!
        oddSites = choices[0::2]
        # evenSites = choices[1::2]
        self.dimensions = range(self.d)
        self.oddLattice = np.full(tuple(np.repeat(self.n, self.d)), False)
        for choice in oddSites:
            self.oddLattice[tuple([int(np.floor(choice % self.n**(x+1) / self.n**x)) for x in self.dimensions])] = True
        self.evenLattice = np.invert(copy.deepcopy(self.oddLattice))
        # iniitalize initial energies using initial state
        self.updateEnergies()
        
        # save initial system data at time stamp=0
        self.systemDataTimeSeries[0].append(self.timeStep)
        self.systemDataTimeSeries[1].append(self.totalMagnetization())
        # care: coordination number normalization when accounting for total energy
        self.systemDataTimeSeries[2].append(np.sum(self.interactionEnergies) / 2)

    def totalMagnetization(self):
        return self.m * np.sum(self.system)

    def localEnergy(self, coords):  # periodic bc
        # coords a list contain d integer indices to specify the ising lattice in d dimensional space
        sumOfNeighbours = 0
        for i in range(len(coords)):  # traverse in f/b directions
            coordsCopy = copy.deepcopy(coords)  # deep copy by default
            coordsCopy[i] = (coords[i] + 1) % self.n
            sumOfNeighbours += self.system[tuple([[x] for x in coordsCopy])][0]
            coordsCopy[i] = (coordsCopy[i] - 2) % self.n
            sumOfNeighbours += self.system[tuple([[x] for x in coordsCopy])][0]

        coords = tuple([[x] for x in coords])
        return (- self.j * (self.system[coords]) * sumOfNeighbours +
                self.m * self.h * self.system[coords])[0]
        # coupling energy + external field interaction

    # function for parallel update
    def updateEnergies(self):
        self.interactionEnergies = \
                (-self.j) * (convolve(self.system, self.kernel, mode='wrap') - self.system) * self.system + \
                self.m * self.h * self.system

    def flip(self, coords):
        energy = self.localEnergy(coords)
        # print(energy)
        coords = tuple([[x] for x in coords])
        if energy >= 0:  # flip
            self.system[coords] *= -1
        else:
            boltzmanFactor = np.exp(2*energy/(self.k * self.t))
            # p = random.uniform(0, 1)
            if random.randint(0, 1) < boltzmanFactor: self.system[coords] = -self.system[coords]

    def visualizeMagnetization(self, path="noPath.png", hyperplane=None):
        # plots the total magnetization with time
        plt.close()
        fig = plt.figure()
        plt.plot(self.systemDataTimeSeries[0], self.systemDataTimeSeries[1], "+k")
        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", N = "+str(self.n)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, T = "+str(sigfig.round(float(self.t), sigfigs=4)) + "K, Time = "+str(self.timeStep)+"au", 60)))
        plt.xlabel("Time steps / a.u.")
        plt.ylabel("Total magnetization / Am^2")
        return fig

    def visualizeTotalEnergy(self, path="noPath.png", hyperplane=None):
        # plots the total magnetization with time
        plt.close()
        fig = plt.figure()
        plt.plot(self.systemDataTimeSeries[0], self.systemDataTimeSeries[2], "+k")
        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", N = "+str(self.n)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, T = "+str(sigfig.round(float(self.t), sigfigs=4)) + "K, Time = "+str(self.timeStep)+"au", 60)))
        plt.xlabel("Time steps / a.u.")
        plt.ylabel("Total energy / J")
        return fig


    def visualizeTwoDGrid(self, path="noPath.png", hyperplane=None):
        # safety measure: close all plots
        plt.close()
        # hyperplane should be an integer indexing list
        cmap = mpl.colors.ListedColormap(['white', 'black'])
        bounds = [-1, 0, 1]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # data reshape
        data = copy.deepcopy(self.system[hyperplane])
        if data.shape == tuple([self.n, self.n]):
            pass
        else:
            data = data[:, :, 0]
        # plot
        fig, axes = plt.subplots()
        fig = plt.figure()
        axes = fig.add_subplot(111)
        img = axes.imshow(data, interpolation='nearest', cmap=cmap, norm=norm, animated=True)

        plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[-1, 0, 1])
        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", N = "+str(self.n)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, T = "+str(sigfig.round(float(self.t), sigfigs=4)) + "K, Time = "+str(self.timeStep)+"au", 60)))
        # plt.show()
        # plt.savefig(path, dpi=1000)
        return fig
        # plt.close()

    def stepForward(self):
        # stepping through the lattice and update randomly
        # improved method: divide into two sub-lattices and vectorize for each sub lattice to allow batch processing
        # note that a site cannot be updated twice within a single step, and two neighbouring sites should not be updated simultaneously
        self.timeStep = self.timeStep + 1
        
        # oddSites
        # self.updateEnergies()
        # print(2*self.interactionEnergies/self.k * self.t)
        # print(self.interactionEnergies)
        boltzmanFactor = np.exp(2 * self.interactionEnergies / (self.k * self.t))
        # print(boltzmanFactor)
        evenDist = np.random.uniform(0, 1, size=np.repeat(self.n, self.d))
        temp1 = np.greater(self.interactionEnergies, self.ground)
        temp2 = np.greater(boltzmanFactor, evenDist)
        # print("temp1")
        # print(temp1)
        # print("temp2")
        # print(temp2)
        # print("evenDist")
        # print(evenDist)
        criteria = np.logical_and(self.oddLattice, np.logical_or(temp1, temp2))
        self.system = np.where(criteria, -self.system, self.system)
        
        # evenSites
        self.updateEnergies()
        # print(2*self.interactionEnergies/self.k * self.t)
        # print(self.interactionEnergies)
        boltzmanFactor = np.exp(2 * self.interactionEnergies/(self.k * self.t))
        # print(boltzmanFactor)
        evenDist = np.random.uniform(0, 1, size=np.repeat(self.n, self.d))
        temp1 = np.greater(self.interactionEnergies, self.ground)
        temp2 = np.greater(boltzmanFactor, evenDist)
        # print("temp1")
        # print(temp1)
        # print("temp2")
        # print(temp2)
        # print("evenDist")
        # print(evenDist)
        criteria = np.logical_and(self.evenLattice, np.logical_or(temp1, temp2))
        self.system = np.where(criteria, -self.system, self.system)

        # CRITICAL NOTICE: ALL STATE VARIABLES MUST BE UPDATED AFTER A self.system UPDATE
        self.updateEnergies()
        # record system data
        self.systemDataTimeSeries[0].append(self.timeStep)
        self.systemDataTimeSeries[1].append(self.totalMagnetization())
        # care: coordination number normalization when accounting for total energy
        self.systemDataTimeSeries[2].append(np.sum(self.interactionEnergies) / 2)


# ----------------------------------------------------------------------------------------
# calculate the time evolution of a 2 D ising model system given the predefined parameters


# working now: investigate the hysteresis effect in a 2 d system
# the hysteresis by cycling h
# measure the remnant field
# measure the external field needed for total negating of the field 
# measure the hysteresis loop energy

def calcSys(name="newSys", N=100, H=0, T=150, t=50, stabalize=True, stabalize_length=10, D=2):
    # test
    # N:-> system size
    # H:-> external field strength
    # T:-> temperature
    # t:-> number of steps
    # stabalize:-> whether conduct initial evolution of about 10 steps to equilibriate
    newSys = Ising(name, N, H, T, D)

    if stabalize:
        for i in range(stabalize_length):
            newSys.stepForward()
    # dispose of the pre evolution series

    # step through n steps
    for i in range(t):
        newSys.magnetizationTimeSeries[0].append(newSys.timeStep)
        newSys.magnetizationTimeSeries[1].append(newSys.calcMag())
        newSys.stepForward()

    # output to static HTML file
    output_file(name + "_sys" + ".html")
    # create a new plot with a title and axis labels
    p = figure(title="2D ferromagnetic system, N = " + str(N) + ", T = " + str(T) + "K, H = " + str(H) + "T, T_c = " + str(newSys.tc) + "K",
               x_axis_label='t/step', y_axis_label='magnetization/A*m^2')
    # add a line renderer with legend and line thickness
    p.line(newSys.magnetizationTimeSeries[0], newSys.magnetizationTimeSeries[1], legend_label="Magnet.", line_width=2)
    # save to html
    save(p)
    return newSys
# ------------------------------------------------------------------------------------
# returns the number of steps needed for a 1/e drop in autocovariance for a single run of a system


def singleLag(name="newAutocovMag", sys=None):
    # exytract a copy of the total magnetization time series
    magnetizationTimeSeries = sys.magnetizationTimeSeries

    # magnetization normalized to zero mean
    meanMag = np.mean(magnetizationTimeSeries[1])
    # def demeanMag(t):
    #     return magnetizationTimeSeries[1][t] - meanMag

    centredMag = np.array([magnetizationTimeSeries[1][x] - meanMag for x in range(len(magnetizationTimeSeries[0]))])

    # pad an extra zero at the end to make autocovariance have the same length as original time series: n-1 -> n
    centredMag = np.append(centredMag, [meanMag])
    autocovMag = statsmodels.tsa.stattools.acovf(centredMag, fft=True)
    normalizationConstant = autocovMag[0]
    # print(normalizationConstant)
    normalizedAutocovMag = np.array([x/normalizationConstant for x in autocovMag])
    lagtime = 0
    expectedDropReached = False
    for i in range(len(normalizedAutocovMag)):
        if normalizedAutocovMag[i] <= np.exp(-1.0):
            expectedDropReached = True
            lagtime = i
            break
    # output to static HTML file
    output_file(name + "_single_lag" + ".html")
    # create a new plot with a title and axis labels
    p = figure(title="2D ferromagnetic system, N = " + str(sys.n) + ", T = " + str(sys.t) + "K, H = " + str(sys.h) + "T, T_c = " +str(sys.tc) + "K", \
                x_axis_label='t/step', y_axis_label='magnetization/A*m^2')
    # add a line renderer with legend and line thickness
    p.line(magnetizationTimeSeries[0], normalizedAutocovMag[:-1], legend_label="Autocov. Magnet.", line_width=2)
    # save
    save(p)
    # return the number of steps needed to get a 1/e drop in auto correlation
    return lagtime

def meanLag(name="none", N=10, H=0, T=300, t=50, cycles=10):
    lagtimes = []
    for i in range(cycles):
        lagtimes.append(singleLag(name, calcSys(name, N, H, T, t)))
    mean = np.mean(lagtimes)
    return mean
