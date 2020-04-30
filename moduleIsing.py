# common computing and statistical packages
import numpy as np
import scipy as sp
import statsmodels.tsa.stattools
import random
import time
import matplotlib.pyplot as pyplot

# a powerful plotter package for jupyter-notebook
import bokeh
from bokeh.plotting import figure, output_file, show, save
from bokeh.io import output_notebook

# extend the cell width of the jupyter notebook
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# paralell computing
import ipyparallel
import socket

import os
from mpi4py import MPI

import csv

output_notebook()


# -------------------------------------------------------------------------------

# class 2d sys
# SI Units ommited
class TwoDIsing: # numerical values see iron from https://www.southampton.ac.uk/~rpb/thesis/node18.html
    def __init__(self, name = "none", N = 100, H = 1, T = 273.15, J = 1.21 * np.power(10.0, -21), mu = 2.22 * np.power(10.0, -23), randomFill = True, K = 1.38064852 * np.power(10.0, -23), D = 2):
        
        self.j = J # coupling constant
        self.h = H # external field strength
        self.n = N # lattice size
        self.mu = mu # chemical potential
        self.k = K # boltzman constant
        self.t = T # temperature in kelvins
        self.d  = D # system dimension
        
        self.tc = 2*J/(K*np.arcsinh(1)) # theoretical tc for 2d by onsanger
        self.timeStep = 0 # initialize timestep marker
        # storage of magnetization time series
        self.magnetizationTimeSeries = [[],[]] # 
        
        # define the 2d system with or without initial values
        spins = [1, -1]
        if randomFill:
            self.system = np.random.choice(spins, size = (N,N))
            # for row in range(N):
            #     for col in range(N):
            #         self.system[row, col] = random.choice(spins)
        else:
            self.system = np.empty([N, N])
    def magnetization(self):
        mag = sum(sum(self.system[i][j] for i in range(self.n)) for j in range(self.n)) * self.mu
        return mag
    def calcMag(self):
        mag = self.magnetization()
        #print("Magnetization: " + str(mag) + " A*m^2")
        return mag
    # this can be done in a very stupid manner, so let's spend a bit of time writing it well
    def totalEnergy(self):
        mag = sum(sum(self.system[i][j] for i in range(self.n)) for j in range(self.n)) * self.mu
        return mag
    
    def localEnergy(self, row, col): # periodic bc
        return - self.j * (self.system[row][col]) * \
        (self.system[(row-1) % self.n][(col) % self.n]+ \
         self.system[(row+1) % self.n][(col) % self.n]+ \
         self.system[(row) % self.n][(col-1) % self.n]+ \
         self.system[(row) % self.n][(col+1) % self.n])- \
         self.mu * self.h * self.system[row][col] # coupling energy + chemical potential

    def calcLocalEnergy(self, row, col):
        localEnergy = self.localEnergy(row, col)
        #print("Interaction energy at (" + str(row) +", "+ str(col) + "): " + str(localEnergy) + " J")
        return localEnergy
    def flip(self, row, col):
        energy = self.localEnergy(row, col)
        
        if energy >= 0: # flip
            self.system[row, col] = -self.system[row, col]
        else:
            boltzmanFactor = np.exp(2*energy/(self.k * self.t))
            #print(boltzmanFactor)
            p = random.uniform(0, 1)
            if p < boltzmanFactor:
                self.system[row, col] = -self.system[row, col]
            else:
                pass
    def stepForward(self):
        # stepping through the lattice and update randomly
        # note that a site cannot be updated twice within a single step
        self.timeStep = self.timeStep+1
        choices = range(self.n*self.n) # list containing all sites
        choices = list(choices)
        row = 0
        col = 0
        # random.shuffle(choices)
        while choices:
            choice = choices.pop()
            row = int(choice/self.n)
            col = choice % self.n
            self.flip(row, col)
        
# ----------------------------------------------------------------------------------------
# calculate the time evolution of a 2 D ising model system given the predefined parameters
def calcTwoDSys(name = "newSys", N = 100, H = 0, T = 150, n = 50, stabalize = True, stabalize_length = 10):
    # test
    # N:-> system size
    # H:-> external field strength
    # T:-> temperature
    # n:-> number of steps
    # stabalize:-> whether conduct initial evolution of about 10 steps to equilibriate
    newSys = TwoDIsing(name, N, H, T)

    if stabalize:
        for i in range(stabalize_length):
            newSys.stepForward()
    # dispose of the pre evolution series

    # step through n steps
    for i in range(n):
        newSys.magnetizationTimeSeries[0].append(newSys.timeStep)
        newSys.magnetizationTimeSeries[1].append(newSys.calcMag())
        newSys.stepForward()
    
    # output to static HTML file
    output_file(name +"_sys"+ ".html")
    # create a new plot with a title and axis labels
    p = figure(title="2D ferromagnetic system, N = " + str(N) + ", T = " + str(T) + "K, H = " + str(H) + "T, T_c = " +str(newSys.tc) + "K", \
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
    def demeanMag(t):
        return magnetizationTimeSeries[1][t] - meanMag

    centredMag = np.array([demeanMag(x) for x in range(len(magnetizationTimeSeries[0]))])

    # pad an extra zero at the end to make autocovariance have the same length as original time series: n-1 -> n
    centredMag = np.append(centredMag, [meanMag])
    autocovMag = statsmodels.tsa.stattools.acovf(centredMag, fft = True)
    normalizationConstant = autocovMag[0]
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
    """
    if (expectedDropReached == True):
        print("Lag time for 1/e drop in autocovariance: " + str(lagtime))
    else:
        print("Warning: 1/e drop in autocovariance not reached, please increase number of steps in the system")
    """    
    # output to static HTML file
    output_file(name +"_single_lag"+ ".html")
    # create a new plot with a title and axis labels
    p = figure(title="2D ferromagnetic system, N = " + str(sys.n) + ", T = " + str(sys.t) + "K, H = " + str(sys.h) + "T, T_c = " +str(sys.tc) + "K", \
               x_axis_label='t/step', y_axis_label='magnetization/A*m^2')
    # add a line renderer with legend and line thickness
    p.line(magnetizationTimeSeries[0], normalizedAutocovMag, legend_label="Autocov. Magnet.", line_width=2)
    # save
    save(p)
    # return the number of steps needed to get a 1/e drop in auto correlation
    return lagtime

def meanLag(name = "none", N = 10, H = 0, T = 300, n = 50, cycles = 10):
    lagtimes = []
    for i in range(cycles):
        lagtimes.append(singleLag(name, calcTwoDSys(name, N, H, T, n)))
    mean=np.mean(lagtimes)
    return mean
