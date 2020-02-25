# common computing and statistical packages
import numpy as np
import numpy.linalg as linalg
# from numpy.random import randint, uniform
# import scipy as sp
from scipy.ndimage import convolve, generate_binary_structure
import statsmodels.tsa.stattools
import random
import time

from itertools import cycle
from sklearn.cluster import Birch, MiniBatchKMeans, KMeans, SpectralClustering
from sklearn import mixture
# from sklearn.datasets import make_blobs
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
from textwrap3 import wrap
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
        # figure sizes
        self.figureScale = 2
        self.figureDpi = 300 # matplotlib defaults to dpi=100
        # in inches
        self.figureHeight = 4.8
        self.figureWidth = 6.4
        # working
        self.kernel = generate_binary_structure(self.d, 1)
        self.ground = np.full(np.repeat(self.n, self.d), 0)
        # storage of system physical property time series: time stamp, total magnetization, total energy,
        self.systemDataTimeSeries = [[], [], []]
        # Time needed to reach steady state from initial state, initially set to zero
        # will be updated if systems takes time t to move into equilibrium
        # will remain zero if the system was initially in equilibrium
        self.equilibriumTime = 0
        # plotting colors
        self.colors_ = cycle(mpl.colors.TABLEAU_COLORS.keys())

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
        zerothSites = choices[0::4]
        firstSites = choices[1::4]
        secondSites = choices[2::4]
        thirdSites = choices[3::4]

        self.dimensions = range(self.d)
        
        self.zerothLattice = np.full(tuple(np.repeat(self.n, self.d)), False)
        for choice in zerothSites:
            self.zerothLattice[tuple([int(np.floor(choice % self.n**(x+1) / self.n**x)) for x in self.dimensions])] = True
        
        # self.dimensions = range(self.d)
        self.firstLattice = np.full(tuple(np.repeat(self.n, self.d)), False)
        for choice in firstSites:
            self.firstLattice[tuple([int(np.floor(choice % self.n**(x+1) / self.n**x)) for x in self.dimensions])] = True
        
        # self.dimensions = range(self.d)
        self.secondLattice = np.full(tuple(np.repeat(self.n, self.d)), False)
        for choice in secondSites:
            self.secondLattice[tuple([int(np.floor(choice % self.n**(x+1) / self.n**x)) for x in self.dimensions])] = True
        
        # self.dimensions = range(self.d)
        self.thirdLattice = np.full(tuple(np.repeat(self.n, self.d)), False)
        for choice in thirdSites:
            self.thirdLattice[tuple([int(np.floor(choice % self.n**(x+1) / self.n**x)) for x in self.dimensions])] = True
       
        # in each step randomly update all sublattices
        self.sublattices = np.array([self.zerothLattice, self.firstLattice, self.secondLattice, self.thirdLattice])

        # self.evenLattice = np.invert(copy.deepcopy(self.oddLattice))
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
        fig = plt.figure(figsize=(self.figureWidth * self.figureScale, self.figureHeight * self.figureScale), dpi=self.figureDpi)
        # fig = plt.figure()
        plt.plot(self.systemDataTimeSeries[0], self.systemDataTimeSeries[1], "+k")
        plt.axvline(x=self.equilibriumTime)
        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", N = "+str(self.n)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, T = "+str(sigfig.round(float(self.t), sigfigs=4)) + "K, Time = "+str(self.timeStep)+"au", 60)))
        plt.xlabel("Time steps / a.u.")
        plt.ylabel("Total magnetization / Am^2")
        return fig

    def visualizeTotalEnergy(self, path="noPath.png", hyperplane=None):
        # plots the total magnetization with time
        plt.close()
        # fig = plt.figure()
        fig = plt.figure(figsize=(self.figureWidth * self.figureScale, self.figureHeight * self.figureScale), dpi=self.figureDpi)
        plt.plot(self.systemDataTimeSeries[0], self.systemDataTimeSeries[2], "+k")
        plt.axvline(x=self.equilibriumTime)
        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", N = "+str(self.n)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, T = "+str(sigfig.round(float(self.t), sigfigs=4)) + "K, Time = "+str(self.timeStep)+"au", 60)))
        plt.xlabel("Time steps / a.u.")
        plt.ylabel("Total energy / J")
        return fig

    def visualizeMagnetizationAutocovariance(self, path="noPath.png", hyperplane=None):
        plt.close()
        fig = plt.figure(figsize=(self.figureWidth * self.figureScale, self.figureHeight * self.figureScale), dpi=self.figureDpi)
        # returns an auto-covariance plot
        magnetizationAutocovariance = statsmodels.tsa.stattools.acovf(self.systemDataTimeSeries[1], demean=True, fft=True)
        normalizedMagnetizationAutocovariance = magnetizationAutocovariance/magnetizationAutocovariance[0]
        # fig = plt.figure()
        plt.plot(self.systemDataTimeSeries[0], normalizedMagnetizationAutocovariance, "+k")
        plt.axvline(x=self.equilibriumTime)
        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", N = "+str(self.n)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, T = "+str(sigfig.round(float(self.t), sigfigs=4)) + "K, Time = "+str(self.timeStep)+"au", 60)))
        plt.xlabel("Time steps / a.u.")
        plt.ylabel("Autocovariance of magnetization")
        return fig

    # data cleaning function that demean and normalize data using its absolute magnitude
    def demeanNormalize(self, ndarray):
        ndarray = copy.deepcopy(ndarray)
        ndarray = ndarray / np.mean(np.absolute(ndarray))
        ndarray = ndarray - np.mean(ndarray)
        return ndarray

    # helper cluster data plotter from scikit-learn, note plots at 1, 2, n sub graph for comparison with the original
    def plotClustersEstimateEquilibriumTime(self, X, model, subplot):
        Y_ = model.predict(X)
        covariances = model.covariances_
        means = model.means_
        for i, (mean, covar, color) in enumerate(zip(
            means, covariances, self.colors_)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            subplot.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(subplot.bbox)
            ell.set_alpha(0.5)
            subplot.add_artist(ell)

        # identify number of steps to equilibrium, if any steps needed at all
        modelLabels = Y_
        labels, counts = np.unique(modelLabels[modelLabels >= 0], return_counts=True)
        if np.absolute(counts[0] - counts[1]) / self.timeStep < 0.5:
            print("Already in equilibrium state")
        else:
            self.equilibriumTime = np.amin(counts)
            print("Time to equilibrium: " + str(self.equilibriumTime))

    def visualizeMagnetizationPhaseSpace(self, path="noPath.png", hyperplane=None):
        # plots the total magnetization with its time
        plt.close()
        fig = plt.figure(figsize=(self.figureWidth * self.figureScale * 2, self.figureHeight * self.figureScale), dpi=self.figureDpi)
        # import and reshape data
        # scikit-learn only takes columns of attributes
        magnetization = self.demeanNormalize(self.systemDataTimeSeries[1])
        magnetization = np.reshape(magnetization, (magnetization.size, 1))
        magnetizationGradient = self.demeanNormalize(np.gradient(self.systemDataTimeSeries[1]))
        magnetizationGradient = np.reshape(magnetizationGradient, (magnetizationGradient.size, 1))

        # original data
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(magnetization, magnetizationGradient, "+k")
        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", N = "+str(self.n)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, T = "+str(sigfig.round(float(self.t), sigfigs=4)) + "K, Time = "+str(self.timeStep)+"au", 60)))
        plt.xlabel("demeaned and magnitude normalized Magnetization / a.u.")
        plt.ylabel("d(demeaned and magnitude normalized Magnetization)/dt / a.u.")

        # clustering using scikit.learn
        X = np.hstack((magnetization, magnetizationGradient))
        # model = Birch(threshold=0.5, n_clusters=2)
        # model = KMeans(n_clusters=2)
        model = mixture.GaussianMixture(n_components=2, covariance_type='full')
        model.fit(X)
        ax = fig.add_subplot(1, 2, 2)
        self.plotClustersEstimateEquilibriumTime(X, model, ax)
        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", N = "+str(self.n)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, T = "+str(sigfig.round(float(self.t), sigfigs=4)) + "K, Time = "+str(self.timeStep)+"au", 60)))
        plt.xlabel("demeaned and magnitude normalized Magnetization / a.u.")
        plt.ylabel("d(demeaned and magnitude normalized Magnetization)/dt / a.u.")
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
        fig = plt.figure(figsize=(self.figureWidth * self.figureScale, self.figureHeight * self.figureScale))
        axes = fig.add_subplot(111)
        img = axes.imshow(data, interpolation='nearest', cmap=cmap, norm=norm, animated=True)

        plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds, ticks=[-1, 0, 1])
        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", N = "+str(self.n)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, T = "+str(sigfig.round(float(self.t), sigfigs=4)) + "K, Time = "+str(self.timeStep)+"au", 60)))
        return fig

    # warning: do not run on its own
    # this is a helper function fo stepForward
    def updateSublattice(self, sublattice):
        boltzmanFactor = np.exp(2 * self.interactionEnergies / (self.k * self.t))
        evenDist = np.random.uniform(0, 1, size=np.repeat(self.n, self.d))
        temp1 = np.greater(self.interactionEnergies, self.ground)
        temp2 = np.greater(boltzmanFactor, evenDist)
        criteria = np.logical_and(sublattice, np.logical_or(temp1, temp2))
        self.system = np.where(criteria, -self.system, self.system)
        self.updateEnergies()

    def stepForward(self):
        # stepping through the lattice and update randomly
        # improved method: divide into two sub-lattices and vectorize for each sub lattice to allow batch processing
        # note that a site cannot be updated twice within a single step, and two neighbouring sites should not be updated simultaneously
        self.timeStep = self.timeStep + 1

        np.random.shuffle(self.sublattices)
        # print(self.sublattices[0])
        for sublattice in self.sublattices:
            self.updateSublattice(sublattice)

        # record system data
        self.systemDataTimeSeries[0].append(self.timeStep)
        self.systemDataTimeSeries[1].append(self.totalMagnetization())
        # care: coordination number normalization when accounting for total energy to avoid double counting
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
    # stablize:-> whether conduct initial evolution of about 10 steps to equilibriate
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


def visualizeMagnetizationAutocovariance(self, path="noPath.png", hyperplane=None):
    # returns an auto-covariance plot
    magnetizationAutocovariance = statsmodels.tsa.stattools.acovf(self.systemDataTimeSeries[1], fft=True)
    plt.close()
    fig = plt.figure()
    plt.plot(self.systemDataTimeSeries[0], magnetizationAutocovariance[:-1], "+k")
    plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", N = "+str(self.n)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, T = "+str(sigfig.round(float(self.t), sigfigs=4)) + "K, Time = "+str(self.timeStep)+"au", 60)))
    plt.xlabel("Time steps / a.u.")
    plt.ylabel("Autocovariance of magnetization")
    return fig



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
