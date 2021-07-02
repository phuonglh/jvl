# phuonglh
# Interest Rate Models 

"""
    duration(N, yield, n, rate)

    Compute the price and duration of a bond of principle `N`, which pays `rate` percent (annually), 
    assuming a flat yield curve `y(0,T) = yield` in `n` periods.
"""
function duration(N::Float64, yield::Float64, n::Int, rate::Float64)
    # compute price 
    p = sum(rate*exp(-yield/100*i) for i=1:n) + N*exp(-yield/100*n)
    # compute duration
    Δp = sum(i*rate*exp(-yield/100*i) for i=1:n) + n*N*exp(-yield/100*n)
    return (p, Δp/p)
end

function convexity(N::Float64, yield::Float64, n::Int, rate::Float64)
    # compute price 
    p = sum(rate*exp(-yield/100*i) for i=1:n) + N*exp(-yield/100*n)
    # compute nominator
    Δp = sum(i*i*rate*exp(-yield/100*i) for i=1:n) + n*n*N*exp(-yield/100*n)
    return (p, Δp/p)
end

"""
    duration(yields, cashFlows)

    Compute the price and duration of a bond portfolio given yields and cash flow. 
    Suppose that the principal is included in the cash flow.
"""
function duration(yields::Array{Float64}, cashFlows::Array{Float64})
    n = length(cashFlows)
    # compute price
    p = sum(cashFlows[i]*exp(-yields[i]/100*i) for i=1:n)
    # compute duration
    Δp = sum(i*cashFlows[i]*exp(-yields[i]/100*i) for i=1:n)
    return (p, Δp/p)
end

function convexity(yields::Array{Float64}, cashFlows::Array{Float64})
    n = length(cashFlows)
    # compute price
    p = sum(cashFlows[i]*exp(-yields[i]/100*i) for i=1:n)
    # compute nominator
    Δp = sum(i*i*cashFlows[i]*exp(-yields[i]/100*i) for i=1:n)
    return (p, Δp/p)
end


"""
    duration(N, yields, n, rate)

    Compute the price and duration of a bond.
"""
function duration(N::Float64, yields::Array{Float64}, n::Int, rate::Float64)
    # compute price 
    p = sum(rate*exp(-yields[i]/100*i) for i=1:n) + N*exp(-yields[n]/100*n)
    # compute duration
    Δp = sum(i*rate*exp(-yields[i]/100*i) for i=1:n) + n*N*exp(-yields[n]/100*n)
    return (p, Δp/p)
end

function convexity(N::Float64, yields::Array{Float64}, n::Int, rate::Float64)
    # compute price 
    p = sum(rate*exp(-yields[i]/100*i) for i=1:n) + N*exp(-yields[n]/100*n)
    # compute nominator
    Δp = sum(i*i*rate*exp(-yields[i]/100*i) for i=1:n) + n*n*N*exp(-yields[n]/100*n)
    return (p, Δp/p)
end

function convexityHedging()
    y = [6., 5.8, 5.62, 5.46]
    c = [6., 8, 106, 7, 9]
    p, D = duration(y, c)
    p1, D1 = duration(100., y, 2, 4.)
    p2, D2 = duration(100., y, 4, 10.)
    A = inv([-D1*p1 -D2*p2; C1*p1 C2*p2])
    B = [D*p; -C*p]
    A*B
end