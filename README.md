# SparseProp: Efficient Event-Based Simulation and Training of Sparse Recurrent Spiking Neural Networks

This repository contains the implementation code for manuscript: <br>
__SparseProp: Efficient Event-Based Simulation and Training of Sparse Recurrent Spiking Neural Networks__ <br>
## Overview
In this work we propose a novel event-based algorithm to simulate and train spiking neural networks, reducing computational cost from N to log(N) per network spike for sparse spiking networks. We provide  example implementations for recurrent networks of leaky integrate-and-fire neurons and quadratic integrate-and-fire neurons and extending the algorithm to neuron models that lack an analytical solution for the next spike time using Chebyshev polynomials.

## Installation

#### Prerequisites
- Download [Julia](https://julialang.org/downloads/) 

#### Dependencies
- Julia (>= 1.5, tested on 1.9.1)
- DataStructures, RandomNumbers, PyPlot
## Getting started
To install the required packages, run the following in the julia REPL after installing Julia:

```
using Pkg

for pkg in ["RandomNumbers", "PyPlot", "DataStructures"]
    Pkg.add(pkg)
end
```

For example, to run a spiking network of 10^5 leaky integrate-and-fire neurons
```
include("LIF_SparseProp.jl")
end
```

## Repository Overview

### LIF_SparseProp
Contains example implementation of a LIF network with \textit{SparseProp}.\
The function lifnet has input parameters 
#n: # of neurons
k: synapses per neuron
j0: synaptic. strength
τ: membrane. time constant
seedic: seed of random number generator for initial condition.
seednet: seed of random number generator for network realization.

### QIF_SparseProp
Contains example implementation of a QIF network with \textit{SparseProp}.\
The function qifnet has the same input parameters as s lifnet above.

### QIF_SparseProp_timebased.jl
Contains example implementation of a QIF network with \textit{SparseProp}.\
Here, instead of the phase representation, we use the time-based heap.


### QIF_SparseProp_inhomogeneous.jl
Contains example implementation of a QIF network with \textit{SparseProp}.\
Here, instead of the phase representation, we use the time-based heap and every neuron receives a different input current.

### EIF_SparseProp.jl

Contains example implementation of an EIF network with \textit{SparseProp}.\
The next spike time is found using a precalculated lookup table. To create the lookup table, we solve the ordinary differential equation of the exponential integrate-and-fire model with high precision using the DifferentialEquations.jl package. This solution is then transformed into a lookup table using DataInterpolations.jl, based on which the phase transition is calculated with high precision.


### EIF_SparseProp_Chebyshev.jl
Same as EIF_SparseProp.jl, but the phase transition curve is approximated by Chebyshev polynomials. This requires the packages ApproxFun.jl, DataInterpolations.jl and DifferentialEquations.jl.
<!---
### Training dynamics of eigenvalues:
Here is a visualization of the recurrent weight matrix and the eigenvalues throughout across training epochs.
![Training dynamics of networks trained on multiple signals shows first tracking of global mean input](eigenvalue_movie_2D_task.gif)
-->


### Implementation details
A full specification of packages used and their versions can be found in _packages.txt_ .\
For all calculations, a 'burn-in' period was discarded to let the network state converge to a stationary state.\
All simulations were run on a single CPU and took on the order of minutes to a few hours.



<!---
### figures/
Contains all figures of the main text and the supplement.
-->


<!---
### tex/
Contains the raw text of the main text and the supplement.
-->
