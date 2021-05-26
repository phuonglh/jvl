# Fitting the Term-Structure of Zero Bond Prices in the Black-Derman-Toy Model
# Week 5 of the FE course
# phuonglh

module BDT

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


function test()
    a = fill(5, 14)
    b = 0.005
    params = Params(1.0, 13, a, b, 0.5)
    dict = elementaryPricing(params)

    marketSpotRates = [7.3, 7.62, 8.1, 8.45, 9.2, 9.64, 10.12, 10.45, 10.75, 11.22, 11.55, 11.92, 12.2, 12.32]
    spotRates = dict[:spotRates][2:end]
    # minimize the sum of square loss function...
end

end # module