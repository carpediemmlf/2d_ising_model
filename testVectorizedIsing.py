# common computing and statistical packages
import numpy as np
import numpy.linalg as linalg
import scipy as sp
import scipy.stats
import statsmodels.tsa.stattools
import random
import time
import copy

# local convolution for local interaction energy
from scipy.ndimage import convolve, generate_binary_structure
from scipy.optimize import curve_fit

# error estimators
from astropy.stats import jackknife_stats
from astropy.stats import bootstrap

# Gaussian clustering
from itertools import cycle
from sklearn import mixture

# convex hulling
from scipy.spatial import ConvexHull, convex_hull_plot_2d
"""
from numba import jit, jitclass
from numba import int64, float64, boolean, uint8

spec = [
    ('removeUnderflow', float64),
    ('j', float64),
    ('h', float64),
    ('t', float64),
    ('n', int64),
    ('m', float64),
    ('k', float64),
    ('d', int64),
    ('tc', float64),
    ('timeStep', int64),
    ('figureScale', float64),
    ('figureDpi', float64),
    ('figureHeight', float64),
    ('figureWidth', float64),
    ('kernel', int64[:]),
    ('ground', float64[:]),
    ('systemDataTimeSeries', float64[:]),
    ('correlationTimeEstimates', float64[:]),
    ('normalizedMagnetizationAutocovariance', float64[:]),
    ('meanMagnetizationWithError', float64[:]),
    ('equilibriumTime', float64),
    ('correlationTime', float64),
    ('system', int64),
    ('dimensions', int64[:]),
    ('zerothLattice', float64[:]),
    ('firstLattice', float64[:]),
    ('secondLattice', float64[:]),
    ('thirdLattice', float64[:]),
    ('randomFill', boolean),
    ('steps', int64),
    ('numberOfMeasurements', int64),
    ('dataPoints', int64),

    # ('array', float64[:]),
    ('lowerTemperature', float64),
    ('deltaTemperature', float64),
    ('deltaBlockSize', int64),
    ('name', uint8[:]),
    ('blockSizeAndVarianceInMeanMagnetization', float64[:]),
    ('meanStationaryMagnetizationAgainstTemperature', float64[:]),
    ('equilibriumTimeAgainstTemperature', float64[:]),
    ('correlationTimeAgainstTemperature', float64[:]),
    ]
"""

import os  # writing to path
import sigfig  # rounding for data in plot title
from textwrap3 import wrap

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

# -------------------------------------------------------------------------------

# SI Units ommited

# @jitclass(spec)
class Ising:
    # iron from https://www.southampton.ac.uk/~rpb/thesis/node18.html
    def __init__(self, name="none", N=100, H=0, T=273.15, D=2, J=1.21 * np.power(10.0, -21), randomFill=True, K=1.38064852 * np.power(10.0, -23), M=2.22 * np.power(10.0, (-23)), correlationTime=2, equilibriumTime=200, numberOfMeasurements=15):
        self.removeUnderflow = np.power(10, 0)
        self.j = J  # / self.removeUnderflow  # 'numerical' coupling constant
        self.h = H  # external field strength
        self.n = N  # lattice size
        self.m = M  # single spin magnetic moment
        self.k = K  # boltzman constant
        self.t = T  # / self.removeUnderflow # temperature in kelvins
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
        # storage of blocksize, measured magnetization, variance of measured magnetization, variance in that variance
        self.correlationTimeEstimates = [[], [], []]
        self.binderCumulant = None
        # normalized magnetization auto-covariance
        self.normalizedMagnetizationAutocovariance = None
        # Time needed to reach steady state from initial state, initially set to zero
        # will be updated if systems takes time t to move into equilibrium

        # mean magnetization and its error from a single measurement, returned as a 2 element list
        self.meanMagnetizationWithError = []

        self.equilibriumTime = equilibriumTime
        self.numberOfMeasurements = numberOfMeasurements
        # correlation time initially set to 2
        # can be precomputed then provided
        # Warning: THIS IS A REAL
        self.correlationTime = correlationTime
        # this cannot be provided
        self.specificHeatCapacityPerSite = None
        # plotting colors
        self.colors_ = cycle(mpl.colors.TABLEAU_COLORS.keys())

        # define the system with or without initial values
        spins = [1, -1]
        if randomFill:
            self.system = np.random.choice(spins, tuple(np.repeat(self.n, self.d)))
        else:
            # dangerously, for future importing of existing system
            # currently randomly chooses all up or all down
            self.system = np.full(tuple(np.repeat(self.n, self.d)), np.choose(1, [1, -1]))
        self.system = np.asarray(self.system)
        choices = list(range(self.n**self.d))

        # warning: optimization only works in odd sized dimensions!!!
        zerothSites = choices[0::4]
        firstSites = choices[1::4]
        secondSites = choices[2::4]
        thirdSites = choices[3::4]
        # notice we still have the difficulty of flipping large domains with smooth boundaries at especially low temperatures


        # dimension index
        self.dimensions = range(self.d)

        # zerothLattice
        self.zerothLattice = np.full(tuple(np.repeat(self.n, self.d)), False)
        for choice in zerothSites:
            self.zerothLattice[tuple([int(np.floor(choice % self.n**(x+1) / self.n**x)) for x in self.dimensions])] = True

        # firstLattice
        self.firstLattice = np.full(tuple(np.repeat(self.n, self.d)), False)
        for choice in firstSites:
            self.firstLattice[tuple([int(np.floor(choice % self.n**(x+1) / self.n**x)) for x in self.dimensions])] = True

        # secondLattice
        self.secondLattice = np.full(tuple(np.repeat(self.n, self.d)), False)
        for choice in secondSites:
            self.secondLattice[tuple([int(np.floor(choice % self.n**(x+1) / self.n**x)) for x in self.dimensions])] = True

        # thirdLattice
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

        self.stationaryMagnetization = None

    def totalMagnetization(self):
        return self.m * np.sum(self.system)

    # Warning: estimate the equilibrium time before calculating binderCumulent
    def estimateBinderCumulant(self):
        data = copy.deepcopy(self.systemDataTimeSeries)
        # crop transients
        if not self.equilibriumTime == 0:
            data[0] = data[0][slice(int(self.equilibriumTime * 2), self.timeStep)]
            data[1] = data[1][slice(int(self.equilibriumTime * 2), self.timeStep)]
            data[2] = data[2][slice(int(self.equilibriumTime * 2), self.timeStep)]
        meanToTheFourth = np.mean(np.power(data[1], 4.0))
        meanToTheSecond = np.mean(np.power(data[1], 2.0))
        self.binderCumulant = 1 - meanToTheFourth / (3 * meanToTheSecond ** 2)

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

    # function for parallel update of interaction energies
    def updateEnergies(self):
        self.interactionEnergies = \
                (-self.j) * (convolve(self.system, self.kernel, mode='wrap') - self.system) * self.system + \
                self.m * self.h * self.system

    # currently deprecated
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

    def meanStationaryMagnetization(self, path="noPath.png", hyperplane=None):
        # calculate mean stationary magnetization and its variance
        # return a two element list

        # remove the transient data
        if not self.equilibriumTime == 0:
            steadyStateDataTimeSeries = copy.deepcopy(self.systemDataTimeSeries[1][slice(int(self.equilibriumTime * 2), self.timeStep)])
        else:
            steadyStateDataTimeSeries = copy.deepcopy(self.systemDataTimeSeries[1])
        blockSize = int(self.correlationTime) + 1
        numberOfMeasurements = self.numberOfMeasurements

        # measure in blocks of size the correlation length
        averages = np.array([])
        for i in range(numberOfMeasurements):
            if (i + 1) * blockSize < len(steadyStateDataTimeSeries): # prevent running off indices
                averages = np.append(averages, [np.mean(steadyStateDataTimeSeries[slice(i * blockSize, (i + 1) * blockSize)])])
            else:
                break

        self.stationaryMagnetization = [np.mean(averages), np.sqrt(np.var(averages) / (self.numberOfMeasurements - 1))]
        return self.stationaryMagnetization

    def visualizeMagnetizationAutocovariance(self, path="noPath.png", hyperplane=None):
        plt.close()
        fig = plt.figure(figsize=(self.figureWidth * self.figureScale, self.figureHeight * self.figureScale), dpi=self.figureDpi)
        data = copy.deepcopy(self.systemDataTimeSeries)
        # crop transients
        if not self.equilibriumTime == 0:
            data[0] = data[0][slice(int(self.equilibriumTime * 2), self.timeStep)]
            data[1] = data[1][slice(int(self.equilibriumTime * 2), self.timeStep)]
            data[2] = data[2][slice(int(self.equilibriumTime * 2), self.timeStep)]

        # returns an auto-covariance plot
        magnetizationAutocovariance = statsmodels.tsa.stattools.acovf(data[1], demean=True, fft=True)
        # with non-equilibrium cropped
        normalizedMagnetizationAutocovariance = magnetizationAutocovariance/magnetizationAutocovariance[0]
        # save current autocovariance
        self.normalizedMagnetizationAutocovariance = normalizedMagnetizationAutocovariance
        # fig = plt.figure()
        plt.plot(data[0], normalizedMagnetizationAutocovariance, "+k")
        plt.axvline(x=self.equilibriumTime)
        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", N = "+str(self.n)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, T = "+str(sigfig.round(float(self.t), sigfigs=4)) + "K, Time = "+str(self.timeStep)+"au", 60)))
        plt.xlabel("Time steps / a.u.")
        plt.ylabel("Auto-covariance of magnetization")
        return fig

    # warning, only estimate the correlation time after computing the auto-covariance time series
    def estimateCorrelationTime(self):
        cropLength = 0
        cropFinished = False
        # case where correlation time is not estimated is not handled here
        for i in range(self.timeStep - int(self.equilibriumTime * 2)):
            if cropFinished:
                break
            if self.normalizedMagnetizationAutocovariance[i] <= np.exp(-2.0):
                cropLength = i
                cropFinished = True

        # this is a more sophisticated method by fitting the autocovariance curve
        """
        croppedAutocovariance = copy.deepcopy(self.normalizedMagnetizationAutocovariance[:cropLength])
        time = np.arange(cropLength)

        # print(croppedAutocovariance)
        logCroppedAutocovariance = np.log(croppedAutocovariance)
        # logTime = np.log(time)
        b, m = np.polyfit(time, logCroppedAutocovariance, 1)
        self.correlationTime = np.absolute(1 / m)
        """
        self.correlationTime = cropLength

        # successfully estimated correlation time
        return True

    def estimateSpecificHeatCapacityPerSite(self):
        data = copy.deepcopy(self.systemDataTimeSeries)
        # crop transients
        if not self.equilibriumTime == 0:
            data[0] = data[0][slice(int(self.equilibriumTime * 2), self.timeStep)]
            data[1] = data[1][slice(int(self.equilibriumTime * 2), self.timeStep)]
            data[2] = data[2][slice(int(self.equilibriumTime * 2), self.timeStep)]
        energyTimeSeriesPerSite = data[2] / np.power(self.n, self.d)
        # fluctuation dissipation theorem
        specificHeatCapacityPerSite = np.sqrt((np.var(energyTimeSeriesPerSite))/(self.k * self.t**2))
        self.specificHeatCapacityPerSite = specificHeatCapacityPerSite
        return True

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
        # for i, (mean, covar) in enumerate(zip(
        #     means, covariances)):
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
            self.equilibriumTime = 0
            print("Either already in equilibrium state or system has not reached equilibrium. \nEquilibrium time defaults to 0.")
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
    # this is a helper function for stepForward()
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
# measure the hysteresis loop energy, notice need to add impurities if you want a considerable amount of area enclosed

# measurement class to allow taking measurements across multiple systems
# @jitclass(spec)
class InquireIsing:
    def __init__(self, name="none", N=100, H=0, T=273.15, D=2, J=1.21 * np.power(10.0, -21), randomFill=True, K=1.38064852 * np.power(10.0, -23), M=2.22 * np.power(10.0, (-23)), correlationTime=20, equilibriumTime=0, steps=1500, numberOfMeasurements=15, dataPoints=10, lowerTemperature=20, deltaTemperature = 20):
        self.removeUnderflow = np.power(10, 0)
        self.j = J  # / self.removeUnderflow  # 'numerical' coupling constant
        self.h = H  # external field strength
        self.n = N  # lattice size
        self.m = M  # single spin magnetic moment
        self.k = K  # boltzman constant
        self.t = T  # / self.removeUnderflow # temperature in kelvins
        self.d = D  # system dimension
        self.tc = 2*J/(K*np.arcsinh(1))  # theoretical tc for 2d by onsanger
        self.randomFill = randomFill
        self.steps = steps
        self.numberOfMeasurements = numberOfMeasurements
        self.dataPoints = dataPoints
        self.lowerTemperature = lowerTemperature
        self.deltaTemperature = deltaTemperature
        # figure sizes
        self.figureScale = 1
        self.figureDpi = 300 # matplotlib defaults to dpi=100
        # in inches
        self.figureHeight = 4.8
        self.figureWidth = 6.4
        # Time needed to reach steady state from initial state, initially set to zero
        # will be updated if systems takes time t to move into equilibrium
        self.equilibriumTime = equilibriumTime
        self.name = name
        # correlation time initially set to 2
        # can be inputed if precomputed
        self.correlationTime = correlationTime
        # plotting colors
        # self.colors_ = cycle(mpl.colors.TABLEAU_COLORS.keys())

        # data
        # size of block, mean variance in mean steady state magnetization, variance in the variance
        # self.blockSizeAndVarianceInMeanMagnetization = [[], [], []]
        self.meanStationaryMagnetizationAgainstTemperature = None
        self.equilibriumTimeAgainstTemperature = None
        self.correlationTimeAgainstTemperature = None
        self.binderCumulantAgainstTemperature = None
        self.specificHeatCapacityPerSiteAgainstTemperature = None

    def visualizeBinderCumulant(self):
        # plots the average magnetization vs number of blocks used for measurement
        # when the variance starts to grow wildly, we choose the largest block size with non-diverging error as our correlation length (or time)
        plt.close()
        lowerTemperature = self.lowerTemperature
        deltaTemperature = self.deltaTemperature
        numberOfMeasurements = self.numberOfMeasurements
        attempts = self.dataPoints
        # temperature and equilibriumTime and error
        data = [[], [], []]
        for a in range(attempts):
            temperature = lowerTemperature + a * deltaTemperature
            data[0].append(temperature)
            binderCumulants = []
            for j in range(numberOfMeasurements):
                tempSys = Ising(name=self.name, N=self.n, H=self.h, T=temperature, D=self.d, J=self.j, randomFill=self.randomFill, K=self.k, M=self.m, equilibriumTime=self.equilibriumTime)
                for i in range(self.steps):
                    tempSys.stepForward()
                # estimate equilibrium time needed
                tempSys.visualizeMagnetizationPhaseSpace()
                tempSys.estimateBinderCumulant()
                binderCumulants.append(tempSys.binderCumulant)
                # [meanMagnetization, errorMagnetization] = tempSys.meanStationaryMagnetization()
                # errors.append(errorMagnetization)
            data[1].append(np.mean(binderCumulants))
            
            # three ways to estimate errors
            _input = np.array(binderCumulants)
            _statistics = np.mean
            
            sem = scipy.stats.sem(_input) # only works for mean
            jackknife_estimate, bias, stderr, conf_interval = jackknife_stats(np.array(_input), _statistics, 0.682)
            bootstrap_estimate = bootstrap(_input, bootfunc=_statistics)
            
            data[2].append(jackknife_estimate)
            
        # update stored data
        self.binderCumulantAgainstTemperature = data
        # fig = plt.figure()
        fig = plt.figure(figsize=(self.figureWidth * self.figureScale, self.figureHeight * self.figureScale), dpi=self.figureDpi)
        # print(self.blockSizeAndVarianceInMeanMagnetization)
        plt.errorbar(self.binderCumulantAgainstTemperature[0], self.binderCumulantAgainstTemperature[1], yerr=self.binderCumulantAgainstTemperature[2], fmt="+k")
        # plt.axvline(x=self.equilibriumTime)
        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(tempSys.d)+", N = "+str(tempSys.n)+", Tc = "+str(sigfig.round(float(tempSys.tc), sigfigs=4))+"K, Time = "+str(tempSys.timeStep)+"a.u.", 60)))
        plt.xlabel("Temperature / K")
        plt.ylabel("Binder Cumulant / 1")
        return fig



    def visualizeEquilibriumTime(self):
        # plots the average magnetization vs number of blocks used for measurement
        # when the variance starts to grow wildly, we choose the largest block size with non-diverging error as our correlation length (or time)
        plt.close()
        lowerTemperature = self.lowerTemperature
        deltaTemperature = self.deltaTemperature
        numberOfMeasurements = self.numberOfMeasurements
        attempts = self.dataPoints
        # temperature and equilibriumTime and error
        data = [[], [], []]
        for a in range(attempts):
            temperature = lowerTemperature + a * deltaTemperature
            data[0].append(temperature)
            # self.blockSizeAndVarianceInMeanMagnetization[0].append(blockSize)
            equilibriumTimes = []
            for j in range(numberOfMeasurements):
                tempSys = Ising(name=self.name, N=self.n, H=self.h, T=temperature, D=self.d, J=self.j, randomFill=self.randomFill, K=self.k, M=self.m, equilibriumTime=self.equilibriumTime)
                for i in range(self.steps):
                    tempSys.stepForward()
                # estimate equilibrium time needed
                tempSys.visualizeMagnetizationPhaseSpace()
                equilibriumTimes.append(tempSys.equilibriumTime)
                # [meanMagnetization, errorMagnetization] = tempSys.meanStationaryMagnetization()
                # errors.append(errorMagnetization)
            data[1].append(np.mean(equilibriumTimes))
            data[2].append(np.sqrt(np.var(equilibriumTimes)))
        # update stored data
        self.equilibriumTimeAgainstTemperature = data
        # fig = plt.figure()
        fig = plt.figure(figsize=(self.figureWidth * self.figureScale, self.figureHeight * self.figureScale), dpi=self.figureDpi)
        # print(self.blockSizeAndVarianceInMeanMagnetization)
        plt.errorbar(self.equilibriumTimeAgainstTemperature[0], self.equilibriumTimeAgainstTemperature[1], yerr=self.equilibriumTimeAgainstTemperature[2], fmt="+k")
        # plt.axvline(x=self.equilibriumTime)
        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(tempSys.d)+", N = "+str(tempSys.n)+", Tc = "+str(sigfig.round(float(tempSys.tc), sigfigs=4))+"K, Time = "+str(tempSys.timeStep)+"a.u.", 60)))
        plt.xlabel("Temperature / K")
        plt.ylabel("Equilibrium Time / a.u.")
        return fig

    def visualizeStationaryMagnetization(self):
        # plots the average magnetization vs number of blocks used for measurement
        # when the variance starts to grow wildly, we choose the largest block size with non-diverging error as our correlation length (or time)
        plt.close()
        lowerTemperature = self.lowerTemperature
        deltaTemperature = self.deltaTemperature
        numberOfMeasurements = self.numberOfMeasurements
        dataPoints = self.dataPoints
        # temperature and mean magnetization and error
        data = [[], [], []]
        for a in range(dataPoints):
            temperature = lowerTemperature + a * deltaTemperature

            # compute the correlation time under current temp
            # !!! to be finished !!! 

            data[0].append(temperature)
            # self.blockSizeAndVarianceInMeanMagnetization[0].append(blockSize)
            tempSys = Ising(name=self.name, N=self.n, H=self.h, T=temperature, D=self.d, J=self.j, randomFill=self.randomFill, K=self.k, M=self.m, equilibriumTime=self.equilibriumTime, correlationTime=20, numberOfMeasurements=self.numberOfMeasurements)
            for step in range(self.steps):
                tempSys.stepForward()

            # necessary facilitating computations
            tempSys.visualizeMagnetizationPhaseSpace()
            tempSys.visualizeMagnetizationAutocovariance()
            tempSys.estimateCorrelationTime()
            print(tempSys.correlationTime)

            [meanMag, error] = tempSys.meanStationaryMagnetization()
            data[1].append(meanMag)
            data[2].append(error)
        # update stored data
        self.meanStationaryMagnetizationAgainstTemperature = data
        fig = plt.figure(figsize=(self.figureWidth * self.figureScale, self.figureHeight * self.figureScale), dpi=self.figureDpi)
        plt.errorbar(self.meanStationaryMagnetizationAgainstTemperature[0], self.meanStationaryMagnetizationAgainstTemperature[1], yerr=self.meanStationaryMagnetizationAgainstTemperature[2], fmt="+k")
        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", N = "+str(self.n)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, Time = "+str(self.steps)+"a.u.", 60)))
        plt.xlabel("Temperature / K")
        plt.ylabel("Stationary Magnetization / Am^2")
        return fig

    # helper function returning correlation time for a specific temperature, keeping all other parameters as inputs
    # used in function visualizeCorrelationTime
    def correlationTimeByTemperature(self, T=200):
        correlationTimes = []
        j = 0
        while j < self.numberOfMeasurements:
            # use specified temperature
            tempSys = Ising(name=self.name, N=self.n, H=self.h, T=T, D=self.d, J=self.j, randomFill=self.randomFill, K=self.k, M=self.m, equilibriumTime=self.equilibriumTime, numberOfMeasurements=self.numberOfMeasurements)
            for step in range(self.steps):
                tempSys.stepForward()
            # estimate equilibrium time
            tempSys.visualizeMagnetizationPhaseSpace()
            # compute auto correlation time series
            tempSys.visualizeMagnetizationAutocovariance()
            # estimate correlation time
            tempSys.estimateCorrelationTime()
            j = j + 1
            correlationTimes.append(tempSys.correlationTime)
        return [np.mean(correlationTimes), np.sqrt(np.var(correlationTimes))]

    def visualizeCorrelationTime(self, n=31, delta=10, size=3):
        # plots the average magnetization vs number of blocks used for measurement
        # when the variance starts to grow wildly, we choose the largest block size with non-diverging error as our correlation length (or time)
        plt.close()
        lowerTemperature = self.lowerTemperature
        deltaTemperature = self.deltaTemperature
        numberOfMeasurements = self.numberOfMeasurements
        dataPoints = self.dataPoints

        # fig = plt.figure(figsize=(self.figureWidth * self.figureScale, self.figureHeight * self.figureScale), dpi=self.figureDpi)
        datas = []
        for i in range(size):
            # temperature and mean magnetization and error
            data = [[], [], []]
            _n = n + i * delta
            self.n = _n
            for a in range(dataPoints):
                temperature = lowerTemperature + a * deltaTemperature + i * 0.1 * deltaTemperature
                data[0].append(temperature)
                # offset the data points for visual purposes
                correlationTime = self.correlationTimeByTemperature(temperature)
                data[1].append(correlationTime[0])
                data[2].append(correlationTime[1])
            # update stored data
            self.correlationTimeAgainstTemperature = data
            # fig = plt.figure()
            datas.append(copy.deepcopy(data))
            # for j in range(size)
        # eee
        fig = plt.figure(figsize=(self.figureWidth * self.figureScale, self.figureHeight * self.figureScale), dpi=self.figureDpi) 
        # kept for record the followuing line, do not remove plz
        # plt.errorbar(self.correlationTimeAgainstTemperature[0], self.correlationTimeAgainstTemperature[1], yerr=self.correlationTimeAgainstTemperature[2], fmt="+k", label="n = "+str(int(_n)))
        for i in range(size):
            plt.errorbar(datas[i][0], datas[i][1], yerr=datas[i][2], fmt="+", label="n = "+str(n + i * delta))

        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, Time = "+str(self.steps)+"a.u.", 60)))
        plt.legend()
        plt.xlabel("Temperature / K")
        plt.ylabel("Correlation Time / a. u.")
        return fig

    def specificHeatCapacityPerSiteByTemperature(self, T=200):
        specificHeatCapacities = []
        j = 0
        while j < self.numberOfMeasurements:
            # use specified temperature
            tempSys = Ising(name=self.name, N=self.n, H=self.h, T=T, D=self.d, J=self.j, randomFill=self.randomFill, K=self.k, M=self.m, equilibriumTime=self.equilibriumTime, numberOfMeasurements=self.numberOfMeasurements)
            for step in range(self.steps):
                tempSys.stepForward()
            # estimate equilibrium time
            tempSys.visualizeMagnetizationPhaseSpace()
            tempSys.estimateSpecificHeatCapacityPerSite()
            j = j + 1
            # halted
            specificHeatCapacities.append(tempSys.specificHeatCapacityPerSite)
        return [np.mean(specificHeatCapacities), np.sqrt(np.var(specificHeatCapacities))]

    def visualizeSpecificHeatiCapacityPerSite(self):
        plt.close()
        lowerTemperature = self.lowerTemperature
        deltaTemperature = self.deltaTemperature
        dataPoints = self.dataPoints
        # temperature and mean magnetization and error
        data = [[], [], []]
        for a in range(dataPoints):
            temperature = lowerTemperature + a * deltaTemperature
            data[0].append(temperature)
            specificHeatCapacity = self.specificHeatCapacityPerSiteByTemperature(temperature)
            data[1].append(specificHeatCapacity[0])
            data[2].append(specificHeatCapacity[1])
        # update stored data
        self.specificHeatCapacityPerSiteAgainstTemperature = data
        fig = plt.figure(figsize=(self.figureWidth * self.figureScale, self.figureHeight * self.figureScale), dpi=self.figureDpi)
        plt.errorbar(self.specificHeatCapacityPerSiteAgainstTemperature[0], self.specificHeatCapacityPerSiteAgainstTemperature[1], yerr=self.specificHeatCapacityPerSiteAgainstTemperature[2], fmt="+k")
        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", N = "+str(self.n)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, Time = "+str(self.steps)+"a.u.", 60)))
        plt.xlabel("Temperature / K")
        plt.ylabel("Specific Heat Capacity per site / J * K ^ -1")
        return fig

    def curieTemp(self, n=11, tLow=190, accuracy=2, size=4):
        # curie temp, error
        data = []

        self.n = n
        specific_heats_data = [[], []]
        for i in range(size):
            temperature = tLow + i * accuracy
            specific_heat = self.specificHeatCapacityPerSiteByTemperature(temperature)[0]
            specific_heats_data[0].append(temperature)
            specific_heats_data[1].append(specific_heat)
        a = specific_heats_data[1]
        data = [specific_heats_data[0][a.index(max(a))], accuracy]
        return data

    def visualizeCurieTemperatureAgainstTime(self, n=11, delta=4, length_eles=3, tLow=199, accuracy=0.6, temp_eles=10):
        # size, curie temp, error
        data = [[], [], []]
        # print(self.k)
        for i in range(length_eles):
            _n = n + delta * i
            a = self.curieTemp(_n, tLow, accuracy, temp_eles)
            data[0].append(_n)
            data[1].append(a[0])
            data[2].append(a[1])
        fig = plt.figure(figsize=(self.figureWidth * self.figureScale, self.figureHeight * self.figureScale), dpi=self.figureDpi)
        # temperature in J/k_B units plotted
        data[1] = data[1] / (self.j/self.k)
        data[2] = data[2] / (self.j/self.k)
        plt.errorbar(data[0], data[1], yerr=data[2], fmt="+k")
        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, Time = "+str(self.steps)+"a.u.", 60)))
        plt.xlabel("Lattice Length ")
        plt.ylabel("Curie Temperature / J/k_B")
        return fig
    
    # function to fit, look for tc, n is an array
    def f(self, n, tc, a, nu):
        return tc + a * np.power(n, -1/nu)

    def finiteSizeScaling(self, n=11, delta=4, length_eles=3, tLow=199, accuracy=0.6, temp_eles=10):
        # size, curie temp, error
        data = [[], [], []]
        for i in range(length_eles):
            _n = n + delta * i
            a = self.curieTemp(_n, tLow, accuracy, temp_eles)
            data[0].append(_n)
            data[1].append(a[0])
            data[2].append(a[1])
        fig = plt.figure(figsize=(self.figureWidth * self.figureScale, self.figureHeight * self.figureScale), dpi=self.figureDpi)
        # temperature in J/k_B units plotted
        data[1] = data[1] / (self.j/self.k)
        data[2] = data[2] / (self.j/self.k)
        func = self.f
        xdata = data[0]
        ydata = data[1]
        ysigma = data[2]
        popt, pcov = curve_fit(func, xdata, ydata, sigma=ysigma, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        tc = [popt[0], perr[0]]
        a = [popt[1], perr[1]]
        nu = [popt[2], perr[2]]
        label = "Tc = "+str(tc[0])+" +- " + str(tc[1])
        plt.errorbar(data[0], data[1], yerr=data[2], fmt="+k", label=label)
        plt.legend()
        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, Time = "+str(self.steps)+"a.u.", 60)))
        plt.xlabel("Lattice Length")
        plt.ylabel("Curie Temperature / J/k_B")
        return fig

    def hysteresis(self, temperature=150, delta_H=0.01, period=10000):
        tempSys = Ising(name=self.name, N=self.n, H=self.h, T=temperature, D=self.d, J=self.j, randomFill=self.randomFill, K=self.k, M=self.m, equilibriumTime=self.equilibriumTime, numberOfMeasurements=self.numberOfMeasurements)
        # external field in tesla, internal magnetization
        data = [[], []]
        for i in range(int(period/4)):
            tempSys.h = tempSys.h + (delta_H)
            data[0].append(tempSys.h)
            tempSys.stepForward()
        # calculate over more cycles to remove the initial moving influence
        
        cycles = 3
        for j in range(cycles): 
            for i in range(int(period/4)):
                tempSys.h = tempSys.h - (delta_H)
                data[0].append(tempSys.h)
                tempSys.stepForward()
            for i in range(int(period/4)):
                tempSys.h = tempSys.h - (delta_H)
                data[0].append(tempSys.h)
                tempSys.stepForward()
            for i in range(int(period/4)):
                tempSys.h = tempSys.h + (delta_H)
                data[0].append(tempSys.h)
                tempSys.stepForward()
            for i in range(int(period/4)):
                tempSys.h = tempSys.h + (delta_H)
                data[0].append(tempSys.h)
                tempSys.stepForward()

        data[1] = tempSys.systemDataTimeSeries[1]
        xoffset = int(period/4)
        yoffset = int(period/4 + 1)
        # points = np.column_stack((data[0][xoffset], data[1][yoffset]))
        # hull = ConvexHull(points)
        switching_energy = 0
        t = int(period/4)
        for j in range(cycles): 
            for i in range(int(period/4)):
                t = t + 1
                switching_energy += delta_H * data[1][t] * (-1)
            for i in range(int(period/4)):
                t = t + 1
                switching_energy += delta_H * data[1][t] * (-1)
            for i in range(int(period/4)):
                t = t + 1
                switching_energy += delta_H * data[1][t] * (+1)
            for i in range(int(period/4)):
                t = t + 1
                switching_energy += delta_H * data[1][t] * (+1)


        """
        for i in range(int(period*10)):
            t = int(period/4 + i)
            switching_energy += delta_H * data[1][t]
        """
        # per site
        switching_energy = switching_energy / self.n**2 / cycles
        fig = plt.figure(figsize=(self.figureWidth * self.figureScale, self.figureHeight * self.figureScale), dpi=self.figureDpi)
        
        # plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
        # plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')

        # previous plotting
        xend = period + xoffset
        yend = period + yoffset
        plt.plot(data[0][xoffset:xend], data[1][yoffset:yend], "+k", label = "dissipation energy per site = " + str(sigfig.round(float(switching_energy), sigfigs=2)) + "T * A * m**2")

        plt.title("\n".join(wrap("Ising Model, Dimension = "+str(self.d)+", N = "+str(self.n)+", Tc = "+str(sigfig.round(float(self.tc), sigfigs=4))+"K, T = "+str(sigfig.round(float(temperature), sigfigs=4)) + "K, period = "+str(self.steps)+"au" + ", dH/dt = "+str(delta_H)+"T/au", 60)))
        plt.legend()
        plt.xlabel("External field / T")
        plt.ylabel("Total magnetization / A*m^2")
        return [temperature, switching_energy]
