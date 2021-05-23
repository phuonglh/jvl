# phuonglh@gmail.com
# Zero-coupon bond pricing and coupon-bearing bond procing
# May 2021

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
function zcbPricing(params, targetValues)
    n, q = params.n, params.q
    zcb = zeros(n+1, n+1)
    # compute the short-term rate lattice
    rates = ratePricing(params)
    # fill the last column
    zcb[:, n+1] = targetValues
    f(u, v) = q*u + (1-q)*v
    # fill columns n, n-1,...,1 backwards
    for t=n:-1:1
        for i=1:t
            zcb[i,t] = f(zcb[i,t+1], zcb[i+1,t+1])/(1+rates[i,t])
        end
    end
    return zcb
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
function bondForwardPricing(params, couponRate, maturity)
    # the maturity should be less than the number of periods
    S0, n = params.S0, params.n
    @assert maturity <= n
    # compute the price latice of this bond
    prices = couponBearingBondPricing(params, couponRate)
    # fill the last column of result
    targetValues = prices[:, maturity + 1] .- S0*couponRate
    # update the number of periods of the params to maturity
    params.n = maturity
    result = zcbPricing(params, targetValues[1:maturity+1])
    # compute the zcb
    zcb = zcbPricing(params, fill(S0, maturity+1))
    return S0*result[1,1]/zcb[1,1]
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

function test3()
    params = Params(100, 6, 0.06, 1.25, 0.9, 0.5)
    bondForwardPricing(params, 0.1, 4)
end

end # module