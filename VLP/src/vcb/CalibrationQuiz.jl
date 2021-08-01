# Interest Rate Derivatives, Quiz 4
# (C) phuonglh@gmail.com, July 2021.

using Distributions
using NLsolve
using Optim

W = Normal(0, 1)

""" zcb(Fs, δ)

    Compute zero-coupon bond price from forward rates of common time step `δ`.
    If there are 8 time steps in forward rates `Fs`, there will be 9 values for bond 
    prices. The first value is always 1.
"""
function zcb(Fs::Array{Float64,1}, δ::Float64)::Array{Float64,1}
    n = length(Fs) + 1
    prices = ones(n)
    for t=2:n
        prices[t] = prices[t-1]/(1 + δ*Fs[t-1])
    end
    return prices
end

"""
    kappas(maturities, prices, δ)

    Compute swap rates (κ) at given maturities from zcb prices. If the maturities are specified 
    in 1-year interval, we know that `length(prices) = 2*length(maturity) + 1`.
"""
function kappas(maturities::Array{Int,1}, prices::Array{Float64,1}, δ::Float64)::Array{Float64,1}
    n = length(maturities)
    κs = zeros(n)
    steps = Int(1/δ)
    for t in maturities
        κs[t] = (prices[2] - prices[steps*t+1])/(δ*sum(prices[3:steps*t+1]))
    end
    return κs
end

"""
    capBlack(m, kappas, prices, Ts, Fs, δ, σ)

    Compute cap price using the Black formula. Here, only `σ` parameter is unknown.
"""
function capBlack(m::Int, κs::Array{Float64,1}, prices::Array{Float64,1}, Ts::Array{Float64,1}, Fs::Array{Float64,1}, δ::Float64, σ)
    value = 0.
    κ = κs[m]
    steps = Int(1/δ)
    for i=2:(steps*m)
        d1 = (log(Fs[i]/κ) + 0.5*σ^2*Ts[i])/(σ*sqrt(Ts[i]))
        d2 = d1 - σ*sqrt(Ts[i])
        cpl = δ*prices[i+1]*(Fs[i]*cdf(W, d1) - κ*cdf(W, d2))
        value = value + cpl
    end
    return value
end

"""
    capBachelier(m, kappas, prices, Ts, Fs, δ, σ)

    Compute cap price using the Bachelier formula. Here, only `σ` parameter is unknown.    
"""
function capBachelier(m::Int, κs::Array{Float64,1}, prices::Array{Float64,1}, Ts::Array{Float64,1}, Fs::Array{Float64,1}, δ::Float64, σ)
    value = 0.
    κ = κs[m]
    steps = Int(1/δ)
    for i=2:(steps*m)
        D = (Fs[i] - κ)/(σ*sqrt(Ts[i]))        
        cpl = δ*prices[i+1]*σ*sqrt(Ts[i])*(D*cdf(W, D) + pdf(W, D))
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
    steps = Int(1/δ)
    for i=2:(steps*m)
        d1 = (log(Fs[i]/κ) + 0.5*σ^2*Ts[i])/(σ*sqrt(Ts[i]))
        cplVega = δ*prices[i+1]Fs[i]*sqrt(Ts[i])*pdf(W, d1)
        value = value + cplVega
    end
    return value
end

# observed forward rates
Fs = [6, 8, 9, 10, 10, 10, 9, 9] ./ 100

# time interval 
δ = 0.25

# 1. Compute zero-coupon bond prices
zs = zcb(Fs, δ) 
@info zs # [1.0, 0.985222, 0.965904, 0.944649, 0.921609, 0.899131, 0.877201, 0.857898, 0.83902]

# 2. Compute cap rates κ values at each maturity
maturities = [1, 2]
κs = kappas(maturities, zs, δ)  
@info κs # [0.08984360064268108, 0.09274690493079601]

# 3. Find implied volatility σ for each maturity by matching observed cap prices to Black cap prices

# observed time-0 prices of ATM caps with quarter cash flows
capPrices = [1, 1] ./ 100
# time legs
Ts = collect(0:δ:2)

# consider maturity m = 2
m = 2
function f!(F, x)
    F[1] = capBlack(m, κs, zs, Ts, Fs, δ, x[1]) - capPrices[m]
end

# solve for Black implied σ (the root of a univariable function)
solutionBlack = nlsolve(f!, [0.005], autodiff = :forward)
@info solutionBlack
sigmaBlack = solutionBlack.zero[1]
@info sigmaBlack
@info round(sigmaBlack*100, digits=2)

# Q5: solve for Bachelier implied σ (the root of a univariable function)
# in basis points
function g!(F, x)
    F[1] = capBachelier(m, κs, zs, Ts, Fs, δ, x[1]) - capPrices[m]
end
solutionBachelier = nlsolve(g!, [0.005], autodiff = :forward)
@info solutionBachelier
sigmaBachelier = solutionBachelier.zero[1]
@info sigmaBachelier
@info round(sigmaBachelier*10_000, digits=2)

# Q6: Compute price given sigmaBlack = 14.1%
blackPrice = capBlack(m, κs, zs, Ts, Fs, δ, 0.141)
@info "price = $(blackPrice)"
@info "price = $(round(blackPrice*100, digits=2))"

# Q8: Vasicek model for cap price

# for the 1-factor HJM model 
function integral(ν, β, t0, t1)
    return (ν/β)^2 * (exp(-β*t0) - exp(-β*t1))^2 * (exp(2*β*t0) - 1)/(2*β)
end
    
## Set model parameters
r0 = 0.08
θ = 0.09 # mean reversion level
κ = 0.86 # speed of mean reversion (kappa)
σ = 0.0148 # volatility
t = 0.    # valuation date
T = 30.   #

# Zero-coupon bond prices in Vasicek model 
function zcPrice(r0, κ, θ, σ, T, t)
    u = (1/κ)*(1-exp(-κ*(T-t))) 
    v = exp((θ - 0.5*σ^2/κ^2)*(u - T + t) - (σ^2) / (4*κ) * (u^2))
    return (v*exp(-u*r0))
end

# time legs
δ = 0.25
Ts = collect(0:δ:30)
# compute prices using the Vasicek formula
prices = [zcPrice(r0, κ, θ, σ, Ts[i], 0) for i=2:length(Ts)]
prepend!(prices, 1.0)
@info prices

# compute strike rates at the maturities
maturities = collect(1:30)
κs = kappas(maturities, prices, δ)
@info κs 

# # derive forward rates from zc prices
# Fs = zeros(length(prices)-1)
# for t=1:length(Fs)
#     Fs[t] = (prices[t]/prices[t+1] - 1)/δ
# end
# @info Fs

# for the 1-factor HJM model 
function integral(ν, β, t0, t1)
    return (ν/β)^2 * (exp(-β*t0) - exp(-β*t1))^2 * (exp(2*β*t0) - 1)/(2*β)
end
    
"""
    capGaussianHJM(m, κs, prices, Ts, ν, β,  δ=0.5)

    Compute cap price in Gaussian HJM model. Here `m` is a maturity and `kappas[m]` is the corresponding 
    cap rate. `T` contains time legs. `ν` and `β` are two parameters. 
"""
function capGaussianHJM(m::Int, κs::Array{Float64,1}, prices::Array{Float64,1}, Ts::Array{Float64,1}, ν, β, δ)
    value = 0.
    κ = κs[m]
    steps = Int(1/δ)
    for i=2:(steps*m)
        I = integral(ν, β, Ts[i], Ts[i+1])
        d1 = (log(prices[i+1]/prices[i] * (1 + δ*κ)) + 0.5*I)/sqrt(I)
        d2 = (log(prices[i+1]/prices[i] * (1 + δ*κ)) - 0.5*I)/sqrt(I)
        cpl = prices[i]*cdf(W, -d2) - (1 + δ*κ)*prices[i+1]*cdf(W, -d1)
        value = value + cpl
    end
    return value
end

# ν is σ, β is κ
capPrice = capGaussianHJM(30, κs, prices, Ts, σ, κ, δ)
@info capPrice
@info "cap price = $(round(capPrice*100, digits=2))"

