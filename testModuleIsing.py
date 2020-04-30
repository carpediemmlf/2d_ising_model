# common computing and statistical packages
import numpy as np
# import scipy as sp
import statsmodels.tsa.stattools
import random
import time
# import matplotlib.pyplot as pyplot
import copy

# a powerful plotter package for jupyter-notebook
import bokeh
from bokeh.plotting import figure, output_file, show, save
from bokeh.io import output_notebook

# extend the cell width of the jupyter notebook
from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))

# paralell computing
import ipyparallel
import socket
import os
from mpi4py import MPI

# save temporary data
import csv

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
# output_notebook()


# -------------------------------------------------------------------------------

# class 2d sys
# SI Units ommited
class Ising: # numerical values see iron from https://www.southampton.ac.uk/~rpb/thesis/node18.html
    def __init__(self, name = "none", N = 100, H = 1, T = 273.15, D = 2, J = 1.21 * np.power(10.0, -21), randomFill = True, K = 1.38064852 * np.power(10.0, -23), M = 2.22 * np.power(10.0,(-23))):
        
        self.j = J # coupling constant
        self.h = H # external field strength
        self.n = N # lattice size
        self.m = M # single spin magnetic moment
        self.k = K # boltzman constant
        self.t = T # temperature in kelvins
        self.d = D # system dimension
        self.tc = 2*J/(K*np.arcsinh(1)) # theoretical tc for 2d by onsanger
        self.timeStep = 0 # initialize timestep marker
        
        # storage of magnetization time series
        self.magnetizationTimeSeries = [[],[]] # 

        # define the 2d system with or without initial values
        spins = [1, -1]
        if randomFill:
            self.system = np.random.choice(spins,   tuple(np.repeat(self.n,self.d)))
        else:
            self.system = np.empty(tuple(np.repeat(self.n,self.d)))
        self.system = np.asarray(self.system)
        
        # record initial state
        self.magnetizationTimeSeries[0].append(self.timeStep)
        self.magnetizationTimeSeries[1].append(self.totalMagnetization())

    # def magnetization(self):
    #     mag = sum(sum(self.system[i][j] for i in range(self.n)) for j in range(self.n)) * self.m
    #     return mag
    # def calcMag(self):
    #     mag = self.magnetization()
        #print("Magnetization: " + str(mag) + " A*m^2")
    #     return mag
    # this can be done in a very stupid manner, so let's spend a bit of time writing it well
    def totalMagnetization(self):
        # currently sum for all sites, maybe will implement to be hyperplane specifiable
        # positions = list(range(self.n**self.d))
        # coords = [int(np.floor((choice % self.n**(x+1) ) / self.n**x)) for x in dimensions]
        # dimensions = list(range(self.d))
        return self.m * sum(self.system[tuple([int(np.floor(posi % self.n**(x+1) / self.n**x)) for x in list(range(self.d))])] for posi in list(range(self.n**self.d)))
        # return mag*self.m
    
    def localEnergy(self, coords): # periodic bc
        # coords a list contain d integer indices to specify the ising lattice in d dimensional space
        sumOfNeighbours = 0
        for i in range(len(coords)): # traverse in f/b directions
            coordsCopy = copy.deepcopy(coords) # deep copy by default
            coordsCopy[i] = (coords[i] + 1) % self.n
            sumOfNeighbours += self.system[tuple([[x] for x in coordsCopy])][0]
            coordsCopy[i] = (coordsCopy[i] - 2) % self.n
            sumOfNeighbours += self.system[tuple([[x] for x in coordsCopy])][0]

        coords = tuple([[x] for x in coords])
        return (- self.j * (self.system[coords]) * sumOfNeighbours + \
            self.m * self.h * self.system[coords])[0] # coupling energy + external field interaction

    # def calcLocalEnergy(self, coords):
        # print(coords)
    #     localEnergy = self.localEnergy(coords)
    #     return localEnergy
    def flip(self, coords):
        energy = self.localEnergy(coords)
        # print(energy)
        coords  = tuple([[x] for x in coords])
        if energy >= 0: # flip
            self.system[coords] *= -1
        else:
            boltzmanFactor = np.exp(2*energy/(self.k * self.t))
            # p = random.uniform(0, 1)
            if random.randint(0,1) < boltzmanFactor: self.system[coords] = -self.system[coords]
            # else:
            #     pass

    def visualizeMagnetization(self, name="noName", hyperplane = None):
        # generate and return the total magnetization in a specified region
        pass

    def visualizeTwoDGrid(self, name = "noName", hyperplane = None):
        # safety measure: close all plots before generating the returned image
        # this function generates and returns a 2d hyper plane slice visualization
        plt.close()
        # hyperplane should be an integer indexing list
        # if hyperplane == None:
        #    hyperplane = [0 if not x == indices[0] else slice(self.n) and indices.pop() for x in range(self.d)]
        cmap = mpl.colors.ListedColormap(['white','black'])
        bounds=[-1,0,1]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # data reshape
        data = copy.deepcopy(self.system[hyperplane])
        if data.shape == tuple([self.n, self.n]):
            pass 
        else:
            data = data[:,:,0]
        # plot 
        img  = plt.imshow(data ,interpolation='nearest', cmap = cmap, norm = norm, animated=True)
        
        plt.colorbar(img,cmap=cmap,norm=norm,boundaries=bounds, ticks=[-1,0,1])
        plt.title("Ising Model, Dimension = "+str(self.d)+", N = "+str(self.n)+", Tc = "+str(int(self.tc))+"K, T = "+str(self.t) + "K, Time = "+str(self.timeStep)+"au")
        # plt.show() 
        plt.savefig(name+".png", dpi=1000)
        return img
        # plt.close()
    def stepForward(self):
        # stepping through the lattice and update randomly
        # improved method: divide into two sub-lattices and vectorize for each sub lattice to allow batch processing
        # note that a site cannot be updated twice within a single step, and two neighbouring sites should not be updated simultaneously
        self.timeStep = self.timeStep+1
        dimensions = np.array(range(self.d))
        # dimensions = np.array(dimensions)
        # at the moment only works for even lattice
        choices = list(range(self.n**self.d)) # list containing all sites: general for d dimensions
        # sublattice division to facilitate future parallelism of site updating
        oddSites = choices[1::2]
        evenSites = choices[0::2]
        def temp(choice):
            coords = [int(np.floor((choice % self.n**(x+1) ) / self.n**x)) for x in dimensions]
            # return self.flip([int(np.floor((choice % self.n**(x+1) ) / self.n**x)) for x in dimensions])
            return self.flip(coords)
        # print(oddSites)
        list(map(temp, oddSites))
        list(map(temp, evenSites))

        # record system state
        self.magnetizationTimeSeries[0].append(self.timeStep)
        self.magnetizationTimeSeries[1].append(self.totalMagnetization())

        # map(, oddSites)
        # map(, evenSites)
        # [self.flip([int(np.floor((choice % self.n**(x+1) ) / self.n**x)) for x in dimensions]) for choice in oddSites]
        # [self.flip([int(np.floor((choice % self.n**(x+1) ) / self.n**x)) for x in dimensions]) for choice in evenSites]
        # choices = list(choices)
        # random.shuffle(choices)
        # while choices:
        #     choice = choices.pop()
        #     # now recover the coordinate from this single index
        #     self.flip([int(np.floor((choice % self.n**(x+1) ) / self.n**x)) for x in dimensions])
        #     # self.flip(coords)
        
# ----------------------------------------------------------------------------------------
# calculate the time evolution of a 2 D ising model system given the predefined parameters
def calcSys(name = "newSys", N = 100, H = 0, T = 150, steps = 50, stabalize = True, stabalize_length = 10, D = 2):
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
    for i in range(steps):
        newSys.magnetizationTimeSeries[0].append(newSys.timeStep)
        newSys.magnetizationTimeSeries[1].append(newSys.calcMag())
        newSys.stepForward()
    
    # output to static HTML file
    # output_file(name +"_sys"+ ".html")
    # create a new plot with a title and axis labels
    p = figure(title="Ising system, N = " + str(N) + ", T = " + str(T) + "K, H = " + str(H) + "T, T_c = " +str(newSys.tc) + "K", \
                x_axis_label='t/step', y_axis_label='magnetization/A*m^2')
    # add a line renderer with legend and line thickness
    p.line(newSys.magnetizationTimeSeries[0], newSys.magnetizationTimeSeries[1], legend_label="Magnet.", line_width=2)
    # save to html
    save(p)
    #return (magnetizationTimeSeries, newSys)
    return newSys
# ------------------------------------------------------------------------------------
# returns the number of steps needed for a 1/e drop in autocovariance for a single run of a system
def singleLag(name = "newAutocovMag", sys = None):  
    # exytract a copy of the total magnetization time series
    magnetizationTimeSeries = sys.magnetizationTimeSeries
    
    # magnetization normalized to zero mean
    meanMag = np.mean(magnetizationTimeSeries[1])
    # def demeanMag(t):
    #     return magnetizationTimeSeries[1][t] - meanMag

    centredMag = np.array([magnetizationTimeSeries[1][x] - meanMag for x in range(len(magnetizationTimeSeries[0]))])

    # pad an extra zero at the end to make autocovariance have the same length as original time series: n-1 -> n
    centredMag = np.append(centredMag, [meanMag])
    autocovMag = statsmodels.tsa.stattools.acovf(centredMag, fft = True)
    normalizationConstant = autocovMag[0]
    # print(normalizationConstant)
    normalizedAutocovMag = np.array([x/normalizationConstant for x in autocovMag])
    
    # remember to delete: from haoyang's request 
    # print("auto covariance series")
    # print(normalizedAutocovMag)
    # lag time for 1/e drop in normalized autocovariance
    lagtime = 0
    expectedDropReached = False
    for i in range(len(normalizedAutocovMag)):
        if normalizedAutocovMag[i] <= np.exp(-1.0):
            expectedDropReached = True
            lagtime = i
            break
    # output to static HTML file
    output_file(name +"_single_lag"+ ".html")
    # create a new plot with a title and axis labels
    p = figure(title="2D ferromagnetic system, N = " + str(sys.n) + ", T = " + str(sys.t) + "K, H = " + str(sys.h) + "T, T_c = " +str(sys.tc) + "K", \
                x_axis_label='t/step', y_axis_label='magnetization/A*m^2')
    # add a line renderer with legend and line thickness
    p.line(magnetizationTimeSeries[0], normalizedAutocovMag[:-1], legend_label="Autocov. Magnet.", line_width=2)
    # save
    # print(len(magnetizationTimeSeries[0]))
    # print(len(normalizedAutocovMag))
    save(p)
    # return the number of steps needed to get a 1/e drop in auto correlation
    return lagtime

def meanLag(name = "none", N = 10, H = 0, T = 300, t = 50, cycles = 10):
    lagtimes = []
    for i in range(cycles):
        lagtimes.append(singleLag(name, calcSys(name, N, H, T, t)))
    mean=np.mean(lagtimes)
    return mean
