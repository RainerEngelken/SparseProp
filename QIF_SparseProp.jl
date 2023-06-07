using DataStructures, RandomNumbers.Xorshifts, StatsBase, PyPlot

function qifnet(n, nstep, k, j0, ratewnt, τ, seedic, seednet)
    iext = τ * √k * j0 * ratewnt # iext given by balance equation
    ω = 2 * √iext # phase velocity QIF
    c = -j0 / √k * 2 / ω
    ϕth, ϕshift = π, 0.0# threshold for QIF
    rng = Xoroshiro128Plus(seedic) # init. random number generator
    # initialize binary heap:
    ϕ = MutableBinaryHeap{Float64,DataStructures.FasterReverse}(2π * (0.5 .- rand(rng, n)))
    spikeidx = Int64[] #initialize time
    spiketimes = Float64[] # spike raster
    postidx = Array{Int64,1}(undef, k)

    for s = 1:nstep# main loop
        ϕmax, j = top_with_handle(ϕ)# get phase of next spiking neuron
        dϕ = ϕth - ϕmax - ϕshift# calculate next spike time
        ϕshift += dϕ # global shift to evolve network state
        Random.seed!(rng, j + seednet) # spiking neuron index is seed of rng
        sample!(rng, 1:n-1, postidx; replace = false)  # get receiving neuron index

        @inbounds for i = 1:k # avoid autapses
            postidx[i] >= j && (postidx[i] += 1)
        end
        ptc!(ϕ, postidx, ϕshift, ω, c) # evaluate phase transition curve
        update!(ϕ, j, -π - ϕshift)   # reset spiking neuron
        push!(spiketimes, ϕshift) # store spike times
        push!(spikeidx, j) # store spiking neuron index
    end
    nstep / ϕshift / n / τ * ω, spikeidx, spiketimes * τ / ω# output: rate, spike times & indices 
end

function ptc!(ϕ, postid, ϕshift, ω, c) # phase transition curve of QIF 
    for i in postid
        ϕ[i] = 2atan(tan((ϕ[i] + ϕshift) / 2) + c) - ϕshift #(Eq. 12) 
    end
end

# set parameters:
#n: # of neurons, k: synapses/neuron, j0: syn. strength, τ: membr. time const.
n, nstep, k, j0, ratewnt, τ, seedic, seednet = 10^5, 10^5, 100, 1, 1.0, 0.01, 1, 1
# quick run to compile code
@time qifnet(100, 1, 10, j0, ratewnt, τ, seedic, seednet);

# run & benchmark network with specified parameters
GC.gc()
@time rate, sidx, stimes = qifnet(n, nstep, k, j0, ratewnt, τ, seedic, seednet);

# plot spike raster
plot(stimes, sidx, ",k", ms = 0.1)
ylabel("Neuron Index", fontsize = 20)
xlabel("Time (s)", fontsize = 20)
tight_layout()
