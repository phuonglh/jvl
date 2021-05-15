# Multi-period Binomial Model for Option Pricing
# phuonglh@gmail.com

struct Params
    T   # years
    S0  # initial price
    σ   # volatility
    r   # 
    c   # dividend yield
    n   # number of periods
    K   # strike price
    position # use `-1` for put option, `+1` for call option
    u
    d
    q
    function Params(T, S0, σ, r, c, n, K, position)
        u = exp(σ * sqrt(T/n))
        d = 1/u
        q = (exp((r - c)*T/n) - d)/(u - d)
        return new(T, S0, σ, r, c, n, K, position, u, d, q)
    end
end 

# The following parameters are for testing, the prices C0 must be 5.21
paramsTest = Params(0.50, 100, 0.2, 0.02, 0.01, 10, 100, -1)
paramsQ1 = Params(0.25, 100, 0.3, 0.02, 0.01, 15, 110, 1)
paramsQ2Q5 = Params(0.25, 100, 0.3, 0.02, 0.01, 15, 110, -1)
paramsQ6Q7 = Params(0.25*10/15, 100, 0.3, 0.02, 0.01, 10, 110, +1)
# Put/Call option of Q8
paramsQ8_C = Params(0.25*15/10, 100, 0.3, 0.02, 0.01, 10, 100, +1)
paramsQ8_P = Params(0.25*15/10, 100, 0.3, 0.02, 0.01, 10, 100, -1)

function stockPricing(params::Params)
    n, S0, u, d = params.n, params.S0, params.u, params.d
    stocks = zeros(n+1, n+1)
    stocks[1,1] = S0
    for t=1:n
        for i=0:t
            stocks[i+1,t+1] = S0*u^(t-i)*d^(i)
        end
    end
    return stocks
end

# compute an European option price at time t using two known values (u, v) at time t+1.
# u is a upper value, v is a lower value
function optionPricing(params::Params, stocks::Array{Float64,2}, type::Char='E')
    n, S0, u, d = params.n, params.S0, params.u, params.d
    r, T, q, K = params.r, params.T, params.q, params.K
    position = params.position

    f(u, v) = exp(-r*T/n)*(q*u + (1-q)*v)
    #f(u, v) = (q*u + (1-q)*v)/(1+r)
    options = zeros(n+1, n+1)
    # fill the last column
    for i=1:n+1
        options[i, n+1] = max(position * (stocks[i,n+1] - K), 0)
    end
    # fill columns n, n-1,...,1 backwards
    for t=n:-1:1
        for i=1:t
            options[i,t] = f(options[i,t+1], options[i+1,t+1])
        end
    end
    if type == 'E' # European
        return options
    else # American option pricing
        payoff = max.(position * (stocks .- K), 0)
        return map((a, b) -> max(a, b), options, payoff)
    end
end

function futurePricing(params::Params, stocks::Array{Float64,2})
    n, q = params.n, params.q
    g(u, v) = q*u + (1-q)*v
    # compute the future lattice
    futures = zeros(n+1, n+1)
    # fill the last column
    futures[:, n+1] = stocks[:, n]
    # fill columns n, n-1,...,1 backwards
    for t=n:-1:1
        for i=1:t
            futures[i,t] = g(futures[i,t+1], futures[i+1,t+1])
        end
    end
    return futures
end

function q1()
    stocks = stockPricing(paramsQ1)
    options = optionPricing(paramsQ1, stocks, 'A')
    # round.(stocks, digits=2)
    # round.(options, digits=2)
    return (stocks, options)
end

function q2()
    stocks = stockPricing(paramsQ2Q5)
    options = optionPricing(paramsQ2Q5, stocks, 'A')
    return (stocks, options)
end

function q4()
    stocks = stockPricing(paramsQ2Q5)
    options = optionPricing(paramsQ2Q5, stocks, 'A')
    payoff = max.(stocks .- paramsQ2Q5.K, 0)
    Δ = max.(payoff - options, 0)
end

function q6()
    stocks = stockPricing(paramsQ6Q7)
    futures = futurePricing(paramsQ6Q7, stocks)
    options = optionPricing(paramsQ6Q7, futures, 'A')
    payoff = max.(futures .- paramsQ6Q7.K, 0)
    Δ = max.(payoff - options, 0)
end


function q8()
    stocksC = stockPricing(paramsQ8_C)
    optionsC = optionPricing(paramsQ8_C, stocksC)
    stocksP = stockPricing(paramsQ8_P)
    optionsP = optionPricing(paramsQ8_P, stocksP)
end

# ## From the result tables:
# P0 = 12.3051 
# C0 = 2.60408

# a = P0 + S0 * exp(-c*T/n) 
# b = C0 + K * exp(-r*T/n)

# @info a, b


