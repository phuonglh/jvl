# Interest Rate Derivatives
# (C) phuonglh@gmail.com, July 2021.

using Distributions
using NLsolve
using Optim

W = Normal(0, 1)

""" zcb(Fs, δ=0.5)

    Compute zero-coupon bond price from forward rates of common time step `δ`.
    If there are 8 time steps in forward rates `Fs`, there will be 9 values for bond 
    prices. The first value is always 1.
"""
function zcb(Fs::Array{Float64,1}, δ::Float64=0.5)::Array{Float64,1}
    n = length(Fs) + 1
    prices = ones(n)
    for t=2:n
        prices[t] = prices[t-1]/(1 + δ*Fs[t-1])
    end
    return prices
end

"""
    kappas(maturities, prices, δ=0.5)

    Compute swap rates (κ) at given maturities from zcb prices. If the maturities are specified 
    in 1-year interval, we know that `length(prices) = 2*length(maturity) + 1`.
"""
function kappas(maturities::Array{Int,1}, prices::Array{Float64,1}, δ::Float64=0.5)::Array{Float64,1}
    n = length(maturities)
    κs = zeros(n)
    for t in maturities
        κs[t] = (prices[2] - prices[2*t+1])/(δ*sum(prices[3:2*t+1]))
    end
    return κs
end

# for the 1-factor HJM model 
function integral(ν, β, t0, t1)
    return (ν/β)^2 * (exp(-β*t0) - exp(-β*t1))^2 * (exp(2*β*t0) - 1)/(2*β)
end
    
"""
    capPrice(m, κs, prices, Ts, ν, β,  δ=0.5)

    Compute cap price in Gaussian HJM model. Here `m` is a maturity and `kappas[m]` is the corresponding 
    cap rate. `T` contains time legs. `ν` and `β` are two parameters. 
"""
function capGaussianHJM(m::Int, κs::Array{Float64,1}, prices::Array{Float64,1}, Ts::Array{Float64,1}, ν, β, δ::Float64=0.5)
    value = 0.
    κ = κs[m]
    for i=2:(2*m)
        I = integral(ν, β, Ts[i], Ts[i+1])
        d1 = (log(prices[i+1]/prices[i] * (1 + δ*κ)) + 0.5*I)/sqrt(I)
        d2 = (log(prices[i+1]/prices[i] * (1 + δ*κ)) - 0.5*I)/sqrt(I)
        cpl = prices[i]*cdf(W, -d2) - (1 + δ*κ)*prices[i+1]*cdf(W, -d1)
        value = value + cpl
    end
    return value
end

"""
    capBlack(m, kappas, prices, Ts, Fs, δ, σ)

    Compute cap price using the Black formula. Here, only `σ` parameter is unknown.
"""
function capBlack(m::Int, κs::Array{Float64,1}, prices::Array{Float64,1}, Ts::Array{Float64,1}, Fs::Array{Float64,1}, δ::Float64, σ)
    value = 0.
    κ = κs[m]
    for i=2:(2*m)
        d1 = (log(Fs[i]/κ) + 0.5*σ^2*Ts[i])/(σ*sqrt(Ts[i]))
        d2 = d1 - σ*sqrt(Ts[i])
        cpl = δ*prices[i+1]*(Fs[i]*cdf(W, d1) - κ*cdf(W, d2))
        value = value + cpl
    end
    return value
end

"""
    vegaBlack(m, κs, prices, Ts, Fs, δ, σ)

    Compute Black cap vega.
"""
function vegaBlack(m::Int, κs::Array{Float64,1}, prices::Array{Float64,1}, Ts::Array{Float64,1}, Fs::Array{Float64,1}, δ::Float64, σ)
    value = 0.
    κ = κs[m]
    for i=2:(2*m)
        d1 = (log(Fs[i]/κ) + 0.5*σ^2*Ts[i])/(σ*sqrt(Ts[i]))
        cplVega = δ*prices[i+1]Fs[i]*sqrt(Ts[i])*pdf(W, d1)
        value = value + cplVega
    end
    return value
end

# observed forward rates
Fs = [6, 8, 9, 10, 10, 10, 9, 9] ./ 100

# time interval 
δ = 0.5

# 1. Compute zero-coupon bond prices
zs = zcb(Fs, δ) # [1.0, 0.9709, 0.9335, 0.8933, 0.8508, 0.8103, 0.7717, 0.7385, 0.7067]

# 2. Compute cap rates κ values at each maturity
maturities = [1, 2, 3, 4]
κs = kappas(maturities, zs, δ)  # [0.08, 0.089691, 0.09352, 0.092628]

# 3. Find implied volatility σ for each maturity by matching observed cap prices to Black cap prices

# observed time-0 prices of ATM caps with semi-annual cash flows
capPrices = [0.2, 0.8, 1.2, 1.6] ./ 100
# time legs
Ts = collect(0:δ:4)
impliedVolatilities = Array{Float64,1}()
for m in maturities
    function f!(F, x)
        F[1] = capBlack(m, κs, zs, Ts, Fs, δ, x[1]) - capPrices[m]
    end
    # solve for σ (the root of a univariable function)
    solution = nlsolve(f!, [0.005], autodiff = :forward)
    @info solution
    push!(impliedVolatilities, solution.zero[1])
    @info impliedVolatilities
end

# found Black implied volatilities = [0.19000807, 0.13909835, 0.09679878, 0.09194901]

# 4. Compute Black cap vega
vegas = Array{Float64,1}()
for m in maturities
    push!(vegas, vegaBlack(m, κs, zs, Ts, Fs, δ, impliedVolatilities[m]))
end

# The result should have vegas = [0.01051004, 0.03762907, 0.07393034, 0.11751275]

# 5. Compute a weighted least squared loss function
function loss(x)
    ν, β = x[1], x[2]
    J = 0.
    for m in maturities
        C_hat = capGaussianHJM(m, κs, zs, Ts, ν, β, δ)
        C = capPrices[m]
        J = J + (1/vegas[m]^2)*(C_hat - C)^2
    end
    return J
end

# 6. Optimize the loss function to find the best parameters x*
x0 = [0.01, 0.03]
result = optimize(loss, x0, LBFGS())
ν, β = Optim.minimizer(result)

# ν = 0.029439011333051955, β = 1.5275477707084084 (Nelder-Mead)
# ν = 0.029348957478484473, β = 1.5210014564075842 (LBFGS)

