# phuonglh@gmail.com
# Zero-coupon bond pricing and coupon-bearing bond procing
# May 2021
# Week 5: Bond pricing

module Bond

mutable struct Params
    S0   # initial value
    n    # number of periods
    r_00 # initial short-term rate
    u    # rate of growth of short rate
    d    # rate of decrease of short rate in each period
    q    # risk-neutral prob. of an up-move
end

"Price short-term rate."
function ratePricing(params)
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

"Price a zero-coupon bond at the present time given a target value vector."
function zcbPricing(params, targetValues, discounted=true)
    n, q = params.n, params.q
    zcb = zeros(n+1, n+1)
    # compute the short-term rate lattice
    rates = ratePricing(params)
    # fill the last column
    zcb[:, n+1] = targetValues
    f(u, v) = q*u + (1-q)*v
    # fill columns n, n-1,...,1 backwards
    if (discounted)
        for t=n:-1:1
            for i=1:t
                zcb[i,t] = f(zcb[i,t+1], zcb[i+1,t+1])/(1+rates[i,t])
            end
        end
    else
        for t=n:-1:1
            for i=1:t
                zcb[i,t] = f(zcb[i,t+1], zcb[i+1,t+1])
            end
        end
    end
    return zcb
end

"Call=1, put=-1"
function americanZeroOptionPricing(params, targetValues, strike, expiration, call=1)
    n, q = params.n, params.q
    @assert expiration <= n
    rates = ratePricing(params)
    zcb = zcbPricing(params, targetValues)
    lastValues = max.(0, call .* (zcb[:, expiration+1][1:expiration+1]) .- strike)
    n = expiration
    result = zeros(n+1, n+1)
    result[:, n+1] = lastValues
    f(u, v) = q*u + (1-q)*v
    # fill columns n, n-1,...,1 backwards
    for t=n:-1:1
        for i=1:t
            result[i,t] = max(call * (zcb[i,t] - strike), f(result[i,t+1], result[i+1,t+1])/(1+rates[i,t]))
        end
    end
    return result
end

"Price a n-periods coupon-bearing bond given a coupon rate."
function couponBearingBondPricing(params, couponRate)
    S0, n, q = params.S0, params.n, params.q
    result = zeros(n+1, n+1)
    # compute the short-term rate lattice
    rates = ratePricing(params)
    # fill the last column
    result[:, n+1] .= S0 * (1 + couponRate)
    f(u, v) = q*u + (1-q)*v
    # fill columns n, n-1,...,1 backwards
    for t=n:-1:1
        for i=1:t
            result[i,t] = S0*couponRate + f(result[i,t+1], result[i+1,t+1])/(1+rates[i,t])
        end
    end
    return result
end

"Price a forward on a coupon-bearing bond."
function forwardPricing(params, couponRate, maturity)
    # the maturity should be less than the number of periods
    S0, n = params.S0, params.n
    @assert maturity <= n
    # compute the price latice of this bond
    prices = couponBearingBondPricing(params, couponRate)
    # fill the last column of result
    targetValues = prices[:, maturity + 1] .- S0*couponRate
    # update the number of periods of the params to maturity
    params.n = maturity
    result = zcbPricing(params, targetValues[1:maturity+1], true)
    # compute the zcb
    zcb = zcbPricing(params, fill(S0, maturity+1), true)
    @info zcb
    return S0*result[1,1]/zcb[1,1]
end

"Price a bond future. Note that there is not discount as in the forward."
function futurePricing(params, couponRate, maturity)
    # the maturity should be less than the number of periods
    S0, n = params.S0, params.n
    @assert maturity <= n
    # compute the price latice of this bond
    prices = couponBearingBondPricing(params, couponRate)
    # fill the last column of result
    targetValues = prices[:, maturity + 1] .- S0*couponRate
    # update the number of periods of the params to maturity
    params.n = maturity
    result = zcbPricing(params, targetValues[1:maturity+1], false)
    return result
end

"Price caplets with a strike (fixed rate, %) and an expiration period (integer)"
function capletPricing(params, strike, expiration)
    n = expiration - 1
    params.n = n
    rates = ratePricing(params)
    targetValues = max.(0, (rates[:,n+1] .- strike) ./ (1 .+ rates[:,n+1]))
    @info targetValues
    return zcbPricing(params, targetValues, true)
end

"Price swaps with a strike (fixed rate, %) and an expiration period (integer)."
function swapPricing(params, strike, expiration)
    n = expiration - 1
    params.n = n
    rates = ratePricing(params)
    targetValues = (rates[:,n+1] .- strike) ./ (1 .+ rates[:,n+1])
    result = zeros(n+1, n+1)
    result[:, n+1] = targetValues
    q = params.q
    f(u, v) = q*u + (1-q)*v
    # fill columns n, n-1,...,1 backwards
    for t=n:-1:1
        for i=1:t
            result[i,t] = ((rates[i,t] - strike) + f(result[i,t+1], result[i+1,t+1]))/(1+rates[i,t])
        end
    end
    return result
end

"Price swaptions."
function swaptionPricing(params, swapStrike, swapExpiration, swaptionStrike, swaptionExpiration)
    swaps = swapPricing(params, swapStrike, swapExpiration)
    targetValues = max.(swaptionStrike, swaps[:, swaptionExpiration+1])
    params.n = swaptionExpiration
    return zcbPricing(params, targetValues[1:swaptionExpiration+1], true)
end

"The forward equation"
function elementaryPricing(params)
    n, S0, q = params.n, params.S0, params.q
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
    return zip(round.(zcb, digits=4), round.(spotRates, digits=4))
end


# 4-year zero-coupon bond pricing
function test1()
    params = Params(1, 4, 0.06, 1.25, 0.9, 0.5)
    targetValues = fill(100, params.n+1)
    zcbPricing(params, targetValues)
end

# 6-year 10% coupon bond pricing with initial value $100
function test2()
    params = Params(100, 6, 0.06, 1.25, 0.9, 0.5)
    couponBearingBondPricing(params, 0.1)
end

# forward pricing
function test3()
    params = Params(100, 6, 0.06, 1.25, 0.9, 0.5)
    forwardPricing(params, 0.1, 4)
end

# future pricing
function test4()
    params = Params(100, 6, 0.06, 1.25, 0.9, 0.5)
    futurePricing(params, 0.1, 4)
end

# caplet pricing
function test5()
    params = Params(1, 5, 0.06, 1.25, 0.9, 0.5)
    capletPricing(params, 0.02, 6)
end

# swap pricing
function test6()
    params = Params(1, 5, 0.06, 1.25, 0.9, 0.5)
    swapPricing(params, 0.05, 6)
end

# swaption pricing
function test7()
    params = Params(1, 5, 0.06, 1.25, 0.9, 0.5)
    swaptionPricing(params, 0.05, 6, 0., 3)
end

function test8()
    params = Params(100, 6, 0.06, 1.25, 0.9, 0.5)
    elementaryPricing(params)
end

function test9()
    params = Params(100, 4, 0.06, 1.25, 0.9, 0.5)
    targetValues = fill(100, params.n+1)
    americanZeroOptionPricing(params, targetValues, 88., 3, -1)
end 

# Week 5 answers
function q1()
    params = Params(100, 10, 0.05, 1.1, 0.9, 0.5)
    targetValues = fill(100, params.n+1)
    zcbPricing(params, targetValues)
end

function q2()
    params = Params(100, 10, 0.05, 1.1, 0.9, 0.5)
    forwardPricing(params, 0., 4)
end

function q3()
    params = Params(100, 10, 0.05, 1.1, 0.9, 0.5)
    futurePricing(params, 0., 4)
end

function q4()
    params = Params(100, 10, 0.05, 1.1, 0.9, 0.5)
    targetValues = fill(100, params.n+1)
    americanZeroOptionPricing(params, targetValues, 80., 6, 1)
end

function q5()
    params = Params(1, 10, 0.05, 1.1, 0.9, 0.5)
    swaps = swapPricing(params, 0.045, 11)
    # at time t = 1, we have a = 0.0723544, b = -0.00226851
    # therefore, the answer is value = [q*a + (1-q)*a]/(1 + 0.05)
    # value * 10^6
    swaps
end

function q6()
    params = Params(100, 10, 0.05, 1.1, 0.9, 0.5)
    swaptions = swaptionPricing(params, 0.045, 11, 0., 5)
    swaptions[1,1]*10^6
end

end # module