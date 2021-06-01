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
    rates = ratePricing(params)
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
    return Dict(:zcb => zcb, :spotRates => spotRates, :prices => result, :rates => rates)
end

"Objective function of the example in the lecture."
function J1(a)
    b = 0.005
    params = Params(1.0, 13, a, b, 0.5)
    dict = elementaryPricing(params)
    marketSpotRates = [7.3, 7.62, 8.1, 8.45, 9.2, 9.64, 10.12, 10.45, 10.75, 11.22, 11.55, 11.92, 12.2, 12.32] ./ 100
    predictedSpotRates = dict[:spotRates][2:end]
    difference = marketSpotRates - predictedSpotRates
    return difference'*difference
end

"Optimize the function to compute the best `a` parameters."
function test1()
    # a = fill(0.05, 14)
    a = [0.0730056578106027,
    0.08564292779180635,
    0.11075759106687774,
    0.15236406555233667,
    0.11775454536360405,
    0.16,
    0.16,
    0.16,
    0.16,
    0.16,
    0.05,
    0.05,
    0.05,
    0.05]
   
    # result = optimize(J1, a, NelderMead(), Optim.Options(iterations=10^4))
    # result = optimize(J1, a, SimulatedAnnealing(), Optim.Options(iterations=10^6))
    # particle swarm optimization algorithm with box contraints
    @time result = optimize(J1, a, ParticleSwarm(; lower = fill(0.05, 14), upper = fill(0.16, 14)), Optim.Options(iterations=10^6))
    @info result
    return Optim.minimizer(result)
end

mutable struct ParamsW5
    S0   # initial value
    n    # number of periods
    r_00 # initial short-term rate
    u    # rate of growth of short rate
    d    # rate of decrease of short rate in each period
    q    # risk-neutral prob. of an up-move
end

"Price short-term rate as in W5."
function ratePricingW5(params)
    n, u, d = params.n, params.u, params.d
    rates = zeros(n+1, n+1)
    rates[1,1] = params.r_00
    for t=1:n
        for j=0:t
            rates[j+1,t+1] = rates[1,1]*u^(t-j)*d^(j)
        end
    end
    return rates
end

function hazardRates(n, a, b)
    rates = zeros(n+1, n+1)
    rates[1,1] = a
    for t=1:n
        for i=0:t
            rates[i+1,t+1] = a*b^(i - t/2)
        end
    end
    return rates
end

"""
    Price a zero-coupon bond with recovery at the present time given a target value vector.

    Thre recovery rate R is on percentage, for example R=20 (do not use 0.2).
"""
function zcbWithRecovery(params, targetValues, a, b, R)
    n, q = params.n, params.q
    zcb = zeros(n+1, n+1)
    # compute the short-term rate lattice
    rates = ratePricingW5(params)
    h = hazardRates(n, a, b)
    # fill the last column
    zcb[:, n+1] = targetValues
    f(u, v, h_ti) = q*(1 - h_ti)*u + (1-q)*(1-h_ti)*v
    # fill columns n, n-1,...,1 backwards
    for t=n:-1:1
        for i=1:t
            h_it = h[i,t]
            first = f(zcb[i,t+1], zcb[i+1,t+1], h_it)
            second = h_it*R # q*h_it*R + (1-q)*h_it*R 
            zcb[i,t] = (first + second)/(1+rates[i,t])
        end
    end
    return zcb
end

function q3()
    params = ParamsW5(100, 10, 0.05, 1.1, 0.9, 0.5)
    targetValues = fill(100, params.n+1)
    zcbWithRecovery(params, targetValues, 0.01, 1.01, 20) # 57.2103
end

end # module
