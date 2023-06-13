using DataStructures, RandomNumbers.Xorshifts, StatsBase, PyPlot, DifferentialEquations,PyPlot, DataInterpolations,Roots,Random,BenchmarkTools,ApproxFun

function eifnet(n, nstep, k, j0, ratewnt, τ, seedic, seednet,ΔV)
checkCorrectnessAndBenchmark = true
    iext = τ * sqrt(k) * j0 * ratewnt # iext given by balance equation
    J = -j0 / sqrt(k)
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
u0 = [-10.0] # this hardcoding might be problematic for very strong j0 where voltage has very negative excursions
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
@time solEIF = solve(prob,Tsit5(),callback=cbEIF,reltol=tol,abstol=tol,maxiters=10^6)#maxiters=10^7
@show solEIF.u[end]
figure();plot(solEIF.t,solEIF.u);xlabel("t");ylabel("voltage");title("solution V(t) to EIF");grid("on")
sol2(x) = solEIF(x) .+ 1.0
tMinus1 = fzero(sol2,-1.0)

~, tMinus1Idx = findmin(abs.(solEIF.t.-tMinus1))
ωNumerical = 1/(solEIF.t[end]-solEIF.t[tMinus1Idx]) #numerically defined phase velocity. Inverse of time from v=0 to v=1
EIFtable = QuadraticInterpolation(1.0 .+unique(vcat(reverse(solEIF.u)...)),unique(reverse(-(solEIF.t .-solEIF.t[end])))) #even better because double entries are being removed, they create Infs/NaNs
EIFInversetable = QuadraticInterpolation(unique(solEIF.t[end].-solEIF.t),1.0 .+ unique(vcat(solEIF.u...))) #even better because double 


PTC_EIF(phi,j)= 1-ωNumerical*EIFInversetable(EIFtable((1-phi)/ωNumerical)+j)


@show phiMin = 1-ωNumerical*solEIF.t[end]

##############################################
# add EIF Chebyshev polynomial approximation #
##############################################

S = Chebyshev(phiMin..1.0)
p = points(S,20)
println("evalutating data points of v = PTC_EIF.(p,-j0/sqrt(k))")
@time v = PTC_EIF.(p,-j0/sqrt(k))
println("doing Chebyshev")
@time PTC_Chebyshev = Fun(S,ApproxFun.transform(S,v))

function ptcChebyshev!(ϕ, postid, ϕshift, j) # phase transition curve of EIF 
    for i in postid
        ϕ[i] = PTC_Chebyshev(ϕ[i]+ϕshift) - ϕshift #(Eq. 12) 
    end
end

if checkCorrectnessAndBenchmark

# spot-check correctness:
phitest = 0.1
@show PTC_Chebyshev(phitest)-PTC_EIF(phitest,-j0/sqrt(k))
@show PTC_Chebyshev(phitest)
@show PTC_EIF(phitest,-j0/sqrt(k))

phitest = 0.99
@show PTC_Chebyshev(phitest)-PTC_EIF(phitest,-j0/sqrt(k))
@show PTC_Chebyshev(phitest)
@show PTC_EIF(phitest,-j0/sqrt(k))

# benchmarking of PTC lookup vs PTC lif and PTC Chebyshev:
println("PTC_lookup")
@btime $PTC_EIF($0.1,$0.1)
PTClif(phi,c) = -ωNumerical * log(exp(-(phi) / ωNumerical) + c)
println("PTC_LIF")
@btime $PTClif($0.1,$0.1)
println("PTC_Chebyshev")
@btime $PTC_Chebyshev($0.1)


# plot PTC:
figure();
# plot PTC:
phi = -0:0.01:1
plot(phi,PTC_EIF.(phi,- j0 / sqrt(k)),alpha=0.3)
plot(phi,PTC_Chebyshev.(phi),alpha=0.3)
plot(0:0.1:1,-0:0.1:1,":k")
xlabel(L"\phi")
ylabel(L"d(\phi)")
figure();
# plot PRC:
phi = -0:0.01:1
plot(phi,PTC_EIF.(phi,- j0 / sqrt(k))-phi,alpha=0.3)
plot(phi,PTC_Chebyshev.(phi)-phi,alpha=0.3)
title("PTC")
xlabel(L"\phi")
ylabel(L"d(\phi)-\phi")

end

#############
# main loop #
#############
    for s = 1:nstep # main loop
        ϕmax, j = top_with_handle(ϕ)# get phase of next spiking neuron
        dϕ = ϕth - ϕmax - ϕshift# calculate next spike time
        ϕshift += dϕ # global shift to evolve network state
        Random.seed!(r, j + seednet) # spiking neuron index is seed of rng
        sample!(r, 1:n-1, postidx; replace = false)  # get receiving neuron index

        @inbounds for i = 1:k # avoid autapses
            postidx[i] >= j && (postidx[i] += 1)
        end
        ptcChebyshev!(ϕ, postidx, ϕshift,-j0/sqrt(k)) # evaluate phase transition curve
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
