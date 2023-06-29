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


### example_code/
Example scripts for training networks on one, two and three stimuli.\
runOneStimulus.jl trains an RNN on tracking one OU-signal showing that the network becomes more tightly balanced over training epochs.\
runTwoStimuli.jl trains an RNN on two OU-signal stimulus showing that the network becomes more tightly balanced over training epochs and breaks up into two weakly-connected subnetworks.\
runTheeStimuli.jl trains an RNN on two OU-signal stimulus showing that the network becomes more tightly balanced over training epochs and breaks up into three weakly-connected subnetworks.\
![Training RNN on two signals leads to balanced subpopulations](/figures/S=2.svg?raw=true "balanced subnetworks emerge  after runTheeStimuli.jl")

<!---
### Training dynamics of eigenvalues:
Here is a visualization of the recurrent weight matrix and the eigenvalues throughout across training epochs.
![Training dynamics of networks trained on multiple signals shows first tracking of global mean input](eigenvalue_movie_2D_task.gif)
-->


### Implementation details
A full specification of packages used and their versions can be found in _packages.txt_ .\
For all calculations, a 'burn-in' period was discarded to let the network state converge to a stationary state.\
All simulations were run on a single CPU and took on the order of minutes to a few of hours.



<!---
### figures/
Contains all figures of the main text and the supplement.
-->


<!---
### tex/
Contains the raw text of the main text and the supplement.
-->
