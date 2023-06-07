using DataStructures, RandomNumbers.Xorshifts, StatsBase, Random

function thetanet(n, nstep, k, j0, ratewnt, τ, seedic, seedtopo, eta)
    iext = τ * sqrt(k) * j0 * ratewnt* (ones(n) + eta * sin.(2 * (1:n) ./ n)) # iext given by balance equation

    ω = 2 * .√iext
    c = -j0 ./ √k .* 2 ./ ω
    Dt, timeth = 0.0, 2π ./ ω
    rng = Xoroshiro128Star(seedic) # init. random number generator (Xorshift)
    nextspiketime = MutableBinaryHeap{Float64,DataStructures.FasterForward}(2π * (1 .- rand(rng, n)) ./ ω) # initialize state vector
    spikeidx = Int64[] #initialize time
    spiketimes = Float64[] # spike raster
    postidx = Array{Int64,1}(undef, k)

    @time for s = 1:nstep
        tmin, j = top_with_handle(nextspiketime) # get phase of next spiking neuron
        dt = tmin - Dt  # calculate next spike time in time representation
        Dt += dt # global backshift instead of shifting all phases
        Random.seed!(rng, j + seedtopo) # spiking neuron index is seed of rng to reduce memory
        sample!(rng, 1:n-1, postidx) # get receiving neuron index
        @inbounds for i = 1:k # avoid autapses
            postidx[i] >= j && (postidx[i] += 1)
        end
        updatenextspiketime!(nextspiketime, postidx, Dt, c, ω) # evaluate phase transition curve
        update!(nextspiketime, j, Dt + timeth[j])    # reset spiking neuron
        push!(spiketimes, Dt) # store spiketimes
        push!(spikeidx, j) # store spiking neuron
    end
    nstep / Dt / n / τ, spikeidx, spiketimes * τ, Dt
end

PRCqif(ϕ, cin) = 2atan(tan(ϕ / 2) + cin) # PRC or theta neuron

function updatenextspiketime!(nextspiketime, postid, Dt, cin, ω)
    for i in postid
        ϕ = -(nextspiketime.nodes[nextspiketime.node_map[i]].value - Dt) * ω[i] + π
        ϕ = PRCqif(ϕ, cin[i])
        update!(nextspiketime, i, (π - ϕ) / ω[i] + Dt)
    end
end

#n: # of neurons, k: synapses/neuron, j0: syn. strength, τ: membr. time const. r: rapidness, eta: input tuning
n, nstep, k, j0, ratewnt, τ, seedic, seedtopo, eta = 100000, 100000, 100, 1, 1.0, 0.01, 1, 1, 0.4
@time thetanet(100, 1, 10, j0, ratewnt, τ, seedic, seedtopo, eta) # for precompilation
GC.gc()
@time rate, sidx, stimes, DtOut = thetanet(n, nstep, k, j0, ratewnt, τ, seedic, seedtopo, eta)
@show rate
using PyPlot
plot(stimes, sidx, ",k")
ylabel("Neuron Index")
xlabel("Time (s)")
tight_layout()
