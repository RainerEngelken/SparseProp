using DataStructures, RandomNumbers.Xorshifts, StatsBase, PyPlot

function lifnet(n, nstep, k, j0, ratewnt, τ, seedic, seednet)
    iext = τ * sqrt(k) * j0 * ratewnt / 1000 # iext given by balance equation
    ω, c = 1 / log(1.0 + 1 / iext), j0 / sqrt(k) / (1.0 + iext) # phase velocity LIF
    ϕth, ϕshift = 1.0, 0.0# threshold for LIF
    r = Xoroshiro128Plus(seedic) # init. random number generator
    # initialize binary heap:
    ϕ = MutableBinaryHeap{Float64,DataStructures.FasterReverse}(rand(r, n))
    spikeidx = Int64[] #initialize time
    spiketimes = Float64[] # spike raster
    postidx = rand(Int, k)

    for s = 1:nstep# main loop

        ϕmax, j = top_with_handle(ϕ)# get phase of next spiking neuron
        dϕ = ϕth - ϕmax - ϕshift# calculate next spike time
        ϕshift += dϕ # global shift to evolve network state
        Random.seed!(r, j + seednet) # spiking neuron index is seed of rng
        sample!(r, 1:n-1, postidx; replace = false)  # get receiving neuron index

        @inbounds for i = 1:k # avoid autapses
            postidx[i] >= j && (postidx[i] += 1)
        end
        ptc!(ϕ, postidx, ϕshift, ω, c) # evaluate phase transition curve
        update!(ϕ, j, -ϕshift)   # reset spiking neuron
        push!(spiketimes, ϕshift) # store spike times
        push!(spikeidx, j) # store spiking neuron index
    end
    nstep / ϕshift / n / τ * ω, spikeidx, spiketimes * τ / ω# output: rate, spike times & indices 
end

function ptc!(ϕ, postid, ϕshift, ω, c) # phase transition curve of LIF 
    for i in postid
        ϕ[i] = -ω * log(exp(-(ϕ[i] + ϕshift) / ω) + c) - ϕshift #(Eq. 12) 
    end
end

# set parameters:
#n: # of neurons, k: synapses/neuron, j0: syn. strength, τ: membr. time const.
n, nstep, k, j0, ratewnt, τ, seedic, seednet = 10^5, 10^5, 100, 1, 1.0, 0.01, 1, 1
# quick run to compile code
@time lifnet(100, 1, 10, j0, ratewnt, τ, seedic, seednet);

# run & benchmark network with specified parameters
GC.gc();
@time rate, sidx, stimes = lifnet(n, nstep, k, j0, ratewnt, τ, seedic, seednet);

# plot spike raster
plot(stimes, sidx, ",k", ms = 0.1)
ylabel("Neuron Index", fontsize = 20)
xlabel("Time (s)", fontsize = 20)
tight_layout()