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

# duration hedging
function Q6()
    y = [6., 5.8, 5.62, 5.46, 5.33]
    c = [6., 8, 106, 7, 9]
    p, D = duration(y, c)
    p2, D2 = duration(100., y, 4, 10.)
    return -D*p/(D2*p2)
end

# relative price change of the duration hedged portfolio
function Q7()
    y = [6., 5.8, 5.62, 5.46, 5.33]
    c = [6., 8, 106, 7, 9]
    p, D = duration(y, c)
    p2, D2 = duration(100., y, 4, 10.)
    q = -D*p/(D2*p2)
    # the price of the duration hedged portfolio is 
    U = p + q*p2
    # now the yield curve moves up by 2%, we need to compute the new price 
    y = y .+ 2
    pNew, _ = duration(y, c)
    p2New, _ = duration(100., y, 4, 10.)
    V = pNew + q*p2New
    Δ = (V-U)/U
    return (U, V, Δ)
end

# convexity hedging
function Q8()
    y = [6., 5.8, 5.62, 5.46, 5.33]
    c = [6., 8, 106, 7, 9]
    p, D = duration(y, c)
    C = convexity(y, c)[2]
    p1, D1 = duration(100., y, 2, 4.)
    C1 = convexity(100., y, 2, 4.)[2]
    p2, D2 = duration(100., y, 4, 10.)
    C2 = convexity(100., y, 4, 10.)[2]
    A = inv([-D1*p1 -D2*p2; C1*p1 C2*p2])
    B = [D*p; -C*p]
    A*B
end

# relative price change of the convexity hedged portfolio
function Q9()
    y = [6., 5.8, 5.62, 5.46, 5.33]
    c = [6., 8, 106, 7, 9]
    p, D = duration(y, c)
    C = convexity(y, c)[2]
    p1, D1 = duration(100., y, 2, 4.)
    C1 = convexity(100., y, 2, 4.)[2]
    p2, D2 = duration(100., y, 4, 10.)
    C2 = convexity(100., y, 4, 10.)[2]
    A = inv([-D1*p1 -D2*p2; C1*p1 C2*p2])
    B = [D*p; -C*p]
    q = A*B
    # the price of the convexity hedged portfolio is
    U = p + q'*[p1, p2]
    # now the yield curve moves up by 2%, we need to compute the new price 
    y = y .+ 2
    pNew, _ = duration(y, c)
    p1New, _ = duration(100., y, 2, 4.)
    p2New, _ = duration(100., y, 4, 10.)
    V = pNew + q'*[p1New, p2New]
    Δ = (V-U)/U
    return (U, V, Δ)
end
