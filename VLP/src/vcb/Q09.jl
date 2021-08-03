# Final Quiz 9, 10, 11
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
    Compute Black swaption price with notional N=1.
"""
function swaptionBlack(κ, prices, δ, σ)
    value = 0.
    for i=5:8
        d1 = 0.5*σ*sqrt(1)
        d2 = -d1
        swt = prices[i+1]*κ*(cdf(W, d1) - cdf(W, d2)) # payer swaption R_swap(t) = κ
        value = value + swt
    end
    return value*δ
end

"""
    Compute Bachelier swaption price with notional N=1.
"""
function swaptionBachelier(prices, δ, σ)
    value = 0.
    for i=5:8
        swt = prices[i+1]*σ*sqrt(1)*(0. + pdf(W, 0.))
        value = value + swt
    end
    return value*δ
end

# observed forward rates
Fs = [6, 8, 9, 10, 10, 10, 9, 9] ./ 100

# time interval 
δ = 0.25

# 1. Compute zero-coupon bond prices
zs = zcb(Fs, δ) 
@info zs # [1.0, 0.985222, 0.965904, 0.944649, 0.921609, 0.899131, 0.877201, 0.857898, 0.83902]

# 2. Compute R_swap
swapRate = (zs[5] - zs[9])/(δ*sum(zs[6:9]))

# observed time-0 prices of ATM swaptions with quarter cash flows
swaptionPrice = 1 ./ 100
# time legs
Ts = collect(0:δ:2)

# Q09: find Black implied σ (the root of a univariable function)
function f!(F, x)
    F[1] = swaptionBlack(swapRate, zs, δ, x[1]) - swaptionPrice
end
solutionBlack = nlsolve(f!, [0.005], autodiff = :forward)
@info solutionBlack
sigmaBlack = solutionBlack.zero[1]
@info sigmaBlack
@info round(sigmaBlack*100, digits=2)

# Q10: solve for Bachelier implied σ (the root of a univariable function)
# in basis points
function g!(F, x)
    F[1] = swaptionBachelier(zs, δ, x[1]) - swaptionPrice
end
solutionBachelier = nlsolve(g!, [0.006], autodiff = :forward)
@info solutionBachelier
sigmaBachelier = solutionBachelier.zero[1]
@info sigmaBachelier
@info round(sigmaBachelier*10_000, digits=2) # 288.68, correct.

# Q11: Compute price given sigmaBlack = 50%
blackPrice = swaptionBlack(swapRate, zs, δ, 0.5)
@info "price = $(blackPrice)"
@info "price = $(round(blackPrice*100, digits=2))"

