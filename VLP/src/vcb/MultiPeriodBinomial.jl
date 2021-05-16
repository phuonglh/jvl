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
        u = round(u, digits=4)
        d = 1/u
        d = round(d, digits=4)
        q = (exp((r - c)*T/n) - d)/(u - d)
        q = round(q, digits=4)
        return new(T, S0, σ, r, c, n, K, position, u, d, q)
    end
end 

# The following parameters are for testing, the prices C0 must be 5.21
paramsTest0 = Params(0.50, 100, 0.2, 0.02, 0.01, 10, 100, -1)
# Another test case: C0=100, F0=100.25, O0 (European) = 2.417, O0 (American) = 2.417
paramsTest1 = Params(0.25, 100, 0.3, 0.02, 0.01, 10, 110, 1)
# Yet another test case: C0=100, F0=100.25, O0 (European) = 12.118, O0 (American) = 12.118
paramsTest2 = Params(0.25, 100, 0.3, 0.02, 0.01, 10, 110, -1)

paramsQ1 = Params(0.25, 100, 0.3, 0.02, 0.01, 15, 110, 1)
paramsQ2Q5 = Params(0.25, 100, 0.3, 0.02, 0.01, 15, 110, -1)
paramsQ6Q7 = Params(0.25*10/15, 100, 0.3, 0.02, 0.01, 15, 110, +1)

# Put/Call option of Q8
paramsQ8_C = Params(0.25, 100, 0.3, 0.02, 0.01, 15, 100, +1)
paramsQ8_P = Params(0.25, 100, 0.3, 0.02, 0.01, 15, 100, -1)

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

    options = zeros(n+1, n+1)
    # fill the last column
    for i=1:n+1
        options[i, n+1] = max(position*(stocks[i,n+1] - K), 0)
    end
    f(u, v) = exp(-r*T/n)*(q*u + (1-q)*v)
    # fill columns n, n-1,...,1 backwards
    for t=n:-1:1
        for i=1:t
            options[i,t] = f(options[i,t+1], options[i+1,t+1])
        end
    end
    if type == 'E' # European
        return options
    else # American
        optionsA = zeros(n+1,n+1)
        optionsA[:,n+1] = options[:,n+1]
        for t=n:-1:1
            for i=1:t
                optionsA[i,t] = max(max(position*(stocks[i,t] - K), 0), options[i,t])
            end
        end
        return optionsA
    end
end

function futurePricing(params::Params, stocks::Array{Float64,2})
    n, q = params.n, params.q
    r, T = params.r, params.T
    # compute the future lattice
    futures = zeros(n+1, n+1)
    # fill the last column
    futures[:, n+1] = stocks[:, n+1]
    f(u, v) = q*u + (1-q)*v
    # fill columns n, n-1,...,1 backwards
    for t=n:-1:1
        for i=1:t
            futures[i,t] = f(futures[i,t+1], futures[i+1,t+1])
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
    position, K = paramsQ2Q5.position, paramsQ2Q5.K
    payoff = max.(position*(K .- stocks), 0)
    Δ = max.(payoff - options, 0)
end

function q5()
    # Test the quality: a == b    
    # a = P0 + S0 * exp(-c*T/n) 
    # b = C0 + K * exp(-r*T/n)    
end

function q6()
    stocks = stockPricing(paramsQ6Q7)
    futures = futurePricing(paramsQ6Q7, stocks)
    params = Params(0.25, 100, 0.3, 0.02, 0.01, 15, 100, +1)
    options = optionPricing(params, futures, 'A')
    return options
end

function q7()
    stocks = stockPricing(paramsQ6Q7)
    futures = futurePricing(paramsQ6Q7, stocks)
    options = optionPricing(paramsQ6Q7, futures, 'A')
    position, K = paramsQ6Q7.position, paramsQ6Q7.K
    payoff = max.(position*(K .- stocks), 0)
    n = paramsQ6Q7.n
    for t=1:n
        payoff[t+1:n+1,t] .= 0
    end
    Δ = max.(payoff - options, 0)
end

function q8()
    stocks = stockPricing(paramsQ8_C)
    futures = futurePricing(paramsQ8_C, stocks)

    optionsC = optionPricing(paramsQ8_C, futures)
    optionsP = optionPricing(paramsQ8_P, futures)

    optionsMax = max.(optionsC, optionsP)[:,1:11]
    # adjust the period from 15 to 10
    paramsQ8 = Params(0.25, 100, 0.3, 0.02, 0.01, 10, 100, +1)
    options = optionPricing(paramsQ8, optionsMax)
end



