# Fitting the Term-Structure of Zero Bond Prices in the Black-Derman-Toy Model
# Week 5 of the FE course
# phuonglh

module BDT

using Optim

mutable struct Params
    S0   # initial value
    n    # number of periods
    a    # a[i] parameters
    b    # the volatility parameter, which is kept fixed (not good!) 
    q    # risk-neutral prob. of an up-move
end

"Price short-term rate in the BDT model."
function ratePricing(params)
    n, a, b = params.n, params.a, params.b
    rates = zeros(n+1, n+1)
    rates[1,1] = a[1]
    for t=1:n
        for j=0:t
            rates[j+1,t+1] = a[j+1]*exp(b*j)
        end
    end
    return rates
end

"The forward equation for the BDT model: similar to the Bond.jl but with percentage of rates (./ 100). "
function elementaryPricing(params)
    n, S0, q = params.n + 1, params.S0, params.q
    rates = ratePricing(params) ./ 100 # note that we need to get percentage
    result = zeros(n+1, n+1)
    result[1,1] = 1
    for t=1:n
        result[1,t+1] = q*result[1,t]/(1 + rates[1,t])
        result[t+1,t+1] = (1-q)*result[t,t]/(1 + rates[t,t])
    end
    for t=2:n
        for j=1:t-1
            result[j+1,t+1] = q*result[j,t]/(1+rates[j,t]) + (1-q)*result[j+1,t]/(1+rates[j+1,t])
        end
    end
    zcb = zeros(n+1)
    spotRates = zeros(n+1)
    for t=2:n+1
        zcb[t] = S0*sum(result[:,t])
        spotRates[t] = (S0/zcb[t])^(1/(t-1))-1 
    end
    # note to scale back spot rates to base 100
    return Dict(:zcb => round.(zcb, digits=4), :spotRates => round.(100 .* spotRates, digits=2), 
        :prices => round.(result, digits=4), :rates => round.(rates, digits=4))
end

"Objective function of the example in the lecture."
function J1(a)
    b = 0.005
    params = Params(1.0, 13, a, b, 0.5)
    dict = elementaryPricing(params)
    marketSpotRates = [7.3, 7.62, 8.1, 8.45, 9.2, 9.64, 10.12, 10.45, 10.75, 11.22, 11.55, 11.92, 12.2, 12.32]
    predictedSpotRates = dict[:spotRates][2:end]
    difference = marketSpotRates - predictedSpotRates
    return difference'*difference
end

"Optimize the function to compute the best `a` parameters."
function test1()
    # a = fill(5.0, 14)
    a = [7.332489105428948,
        8.029676325394634,
        13.73286083383041,
        9.890320815836553,
        15.92697134077938,
        16.0,
        15.837510161796507,
        16.0,
        16.0,
        8.081104180797322,
        6.434309482154262,
        6.771005509870793,
        5.0,
        15.042022199387727]
    # result = optimize(J1, a, NelderMead())
    # result = optimize(J1, a, SimulatedAnnealing(), Optim.Options(iterations=10^6))
    # particle swarm optimization algorithm with box contraints
    @time result = optimize(J1, a, ParticleSwarm(; lower = fill(5.0, 14), upper = fill(16., 14)), Optim.Options(iterations=10^6))
    @info result
    return Optim.minimizer(result)
end

end # module

# 1485.232648 seconds (32.05 G allocations: 766.493 GiB, 6.09% gc time)
# ┌ Info:  * Status: failure (reached maximum number of iterations)
# │ 
# │  * Candidate solution
# │     Final objective value:     7.850000e-02
# │ 
# │  * Found with
# │     Algorithm:     Particle Swarm
# │ 
# │  * Convergence measures
# │     |x - x'|               = NaN ≰ 0.0e+00
# │     |x - x'|/|x'|          = NaN ≰ 0.0e+00
# │     |f(x) - f(x')|         = NaN ≰ 0.0e+00
# │     |f(x) - f(x')|/|f(x')| = NaN ≰ 0.0e+00
# │     |g(x)|                 = NaN ≰ 1.0e-08
# │ 
# │  * Work counters
# │     Seconds run:   1485  (vs limit Inf)
# │     Iterations:    1000000
# │     f(x) calls:    15000014
# └     ∇f(x) calls:   0
# 14-element Array{Float64,1}:
#   7.332489105428948
#   8.029676325394634
#  13.73286083383041
#   9.890320815836553
#  15.92697134077938
#  16.0
#  15.837510161796507
#  16.0
#  16.0
#   8.081104180797322
#   6.434309482154262
#   6.771005509870793
#   5.0
#  15.042022199387727
