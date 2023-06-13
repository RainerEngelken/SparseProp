using DataStructures, RandomNumbers.Xorshifts, StatsBase, PyPlot
using DifferentialEquations,PyPlot, DataInterpolations,Roots,Random

function eifnet(n, nstep, k, j0, ratewnt, τ, seedic, seednet,ΔV)
    iext = τ * sqrt(k) * j0 * ratewnt # iext given by balance equation
    J = -j0 / sqrt(k)
    ωould, cold = 1 / log(1.0 + 1 / iext), j0 / sqrt(k) / (1.0 + iext) # phase velocity EIF
    ϕth, ϕshift = 1.0, 0.0# threshold for EIF
    r = Xoroshiro128Plus(seedic) # init. random number generator
    # initialize binary heap:
    ϕ = MutableBinaryHeap{Float64,DataStructures.FasterReverse}(rand(r, n))
    spikeidx = Int64[] #initialize time
    spiketimes = Float64[] # spike raster
    postidx = rand(Int, k)

########################
# add EIF lookup table #
########################
u0 = [-10.0]
function EIF(du,u,p,t)
    du[1] = -u[1] + ΔV*exp((u[1])/ΔV) + p
end
tmax = 100.0 # this hardcoding might be problematic if tmax is not big enough, but generally works for reasonable parameters
@show tspan = (0.0,tmax) 
prob = ODEProblem(EIF,u0,tspan,iext)
condition(u,t,integrator) = (u[1]-1000)*(u[1]+1)
function affectEIF!(integrator)
    if integrator.u[1] > -0.5
        terminate!(integrator)
    end
end
cbEIF = ContinuousCallback(condition,affectEIF!)
tol = 1e-20
@time solEIF = solve(prob,Tsit5(),callback=cbEIF,reltol=tol,abstol=tol)
figure();plot(solEIF.t,solEIF.u);xlabel("t");ylabel("voltage");title("solution V(t) to EIF");grid("on")
sol2(x) = solEIF(x) .+ 1.0
tMinus1 = fzero(sol2,-1.0)

~, tMinus1Idx = findmin(abs.(solEIF.t.-tMinus1))
ωNumerical = 1/(solEIF.t[end]-solEIF.t[tMinus1Idx]) #numerically defined phase velocity. Inverse of time from v=0 to v=1
EIFtable = QuadraticInterpolation(1.0 .+unique(vcat(reverse(solEIF.u)...)),unique(reverse(-(solEIF.t .-solEIF.t[end])))) #even better because double entries are being removed, they create Infs/NaNs
EIFInversetable = QuadraticInterpolation(unique(solEIF.t[end].-solEIF.t),1.0 .+ unique(vcat(solEIF.u...))) #even better because double 

#################
# end EIF stuff #
#################


function PTC_EIF(phi,j)
tn = (1-phi)/ωNumerical
return 1-ωNumerical*EIFInversetable(EIFtable(tn)+j)
end

function ptcLookup!(ϕ, postid, ϕshift, j) # phase transition curve of EIF 
    for i in postid
        ϕ[i] = PTC_EIF(ϕ[i]+ϕshift,j) - ϕshift #(Eq. 12) 
    end
end


    for s = 1:nstep # main loop
        ϕmax, j = top_with_handle(ϕ)# get phase of next spiking neuron
        dϕ = ϕth - ϕmax - ϕshift# calculate next spike time
        ϕshift += dϕ # global shift to evolve network state
        Random.seed!(r, j + seednet) # spiking neuron index is seed of rng
        sample!(r, 1:n-1, postidx; replace = false)  # get receiving neuron index

        @inbounds for i = 1:k # avoid autapses
            postidx[i] >= j && (postidx[i] += 1)
        end
        ptcLookup!(ϕ, postidx, ϕshift,-j0/sqrt(k)) # evaluate phase transition curve
        update!(ϕ, j, -ϕshift)   # reset spiking neuron
        push!(spiketimes, ϕshift) # store spike times
        push!(spikeidx, j) # store spiking neuron index
    end
    nstep / ϕshift / n / τ * ωNumerical, spikeidx, spiketimes * τ / ωNumerical# output: rate, spike times & indices 
end

# set parameters:
#n: # of neurons, k: synapses/neuron, j0: syn. strength, τ: membr. time const.
n, nstep, k, j0, ratewnt, τ, seedic, seednet, ΔV = 10^5, 10^5, 100, 1, 1.0, 0.01, 1, 1, 0.1 #
# ΔV=0.1 corresponds approximately to experimentally fitted values of Badel 2008 in dimensionless units

# run & benchmark network with specified parameters
GC.gc()
@time rate, sidx, stimes = eifnet(n, nstep, k, j0, ratewnt, τ, seedic, seednet, ΔV)


# plot spike raster
figure()
plot(stimes, sidx, ",k", ms = 0.1)
ylabel("Neuron Index", fontsize = 20)
xlabel("Time (s)", fontsize = 20)
tight_layout()
