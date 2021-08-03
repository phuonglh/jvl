# Final Quiz, 12, 13, 14.
# phuonglh@gmail.com, July 2021

using Dates
using LinearAlgebra

t0 = Date(2000, 1, 1)
swaps = [
    (0.36, Date(2001, 1, 1)), 
    (0.52, Date(2002, 1, 1)), 
    (0.93, Date(2003, 1, 1)), 
    (1.21, Date(2004, 1, 1)), 
    (1.46, Date(2005, 1, 1)), 
    (1.66, Date(2006, 1, 1)), 
    (1.84, Date(2007, 1, 1)), 
    (1.99, Date(2008, 1, 1)), 
    (2.13, Date(2009, 1, 1)),
    (2.21, Date(2010, 1, 1)),
    (2.63, Date(2015, 1, 1)),
    (2.73, Date(2020, 1, 1)),
    (2.71, Date(2030, 1, 1))]

struct Instrument
    kind
    rate
    date
    cash
end

function cashSwap(rate, Tn)
    flow = Array{Tuple{Date,Float64},1}()
    range = collect(t0:Dates.Month(6):Tn) # semianual fixed leg
    for j=1:length(range)-1
        push!(flow, (range[j+1], rate/2/100))
    end
    # at the maturity, we need to add 1 to the cash value
    flow[end] = (flow[end][1], 1 + flow[end][2])
    @info flow[end]
    return flow
end

# build all cash flows, each corresponds to a row of the cash flow matrix C.
elements = Array{Instrument,1}()
# swaps
swapDates = map(d -> d[2], swaps)
for d in swaps
    flow = cashSwap(d[1], d[2])
    push!(elements, Instrument(:s, d[1], d[2], flow))
end


function getDates(instrument)
    map(pair -> pair[1], instrument.cash)
end

# time marks (columns of the cash flow matrix C)
dates = Iterators.flatten(map(instrument -> getDates(instrument), elements))
dates = sort(unique(collect(dates)))

# buid a map from dates to column index
dateIndex = Dict{Date,Int}(date => i for (i,date) in enumerate(dates))
# numCols
N = length(dates)

"""
    makeRow(instrument)
"""
function makeRow(instrument)
    v = zeros(N)
    for pair in instrument.cash
        v[dateIndex[pair[1]]] = pair[2]
    end
    return v
end

rows = map(instrument -> makeRow(instrument), elements)
C = hcat(rows...)'

# market price vector p
p = Array{Float64,1}()
push!(p, map(x -> 1., swaps)...)

# construct matrices
δ = zeros(N)
δ[1] = 1/sqrt(Dates.value(dates[1] - t0)/360)
for t=1:N-1
    δ[t+1] = 1/sqrt(Dates.value(dates[t+1] - dates[t])/360)
end
W = Diagonal(δ)
M = Bidiagonal(ones(N), zeros(N-1) .- 1, :L)

# compute the best Δ*
a = zeros(N); a[1] = 1.
A = C*inv(M)*inv(W)
Δ = A'*inv(A*A')*(p - C*inv(M)*a)

# infer price vector d from Δ
function pricing(Δ, δ)
    prices = zeros(N)
    p0 = 1.0
    for i=1:N
        prices[i] = p0 + Δ[i]/δ[i]
        p0 = prices[i]
    end
    return prices
end

# Estimate discounted prices
prices = pricing(Δ, δ)

@info prices

# compute the forward swap rate for the last time leg
# u, v = prices[end-1], prices[end] # 
# F = (1/(Dates.value(dates[end]-dates[end-1])/360))*(u/v - 1)*100


using Distributions
using NLsolve
using Optim

W = Normal(0, 1)

"""
    kappas(maturities, prices, δ=0.5)

    Compute swap rates (κ) at given maturities from zcb prices. If the maturities are specified 
    in 1-year interval, we know that `length(prices) = 2*length(maturity) + 1`.
"""
function kappas(maturities::Array{Int,1}, prices::Array{Float64,1}, δ::Float64=0.5)::Array{Float64,1}
    n = length(maturities)
    κs = zeros(n)
    steps = Int(1/δ)
    for t in maturities
        κs[t] = (prices[2] - prices[steps*t+1])/(δ*sum(prices[3:steps*t+1]))
    end
    return κs
end

# Compute kappa values
maturities = collect(1:30)
δ = 0.5
prepend!(prices, 1.0)
κs = kappas(maturities, prices, δ)
@info κs

# for the 2-factor HJM model 
function integral(ν1, β1, ν2, β2, t0, t1)
    first = (ν1/β1)^2 * (exp(-β1*t0) - exp(-β1*t1))^2 * (exp(2*β1*t0) - 1)/(2*β1) 
    second = (ν2/β2)^2 * (exp(-β2*t0) - exp(-β2*t1))^2 * (exp(2*β2*t0) - 1)/(2*β2)
    return first + second
end

"""
    capGaussianHJM(m, κs, prices, Ts, ν, β,  δ=0.5)

    Compute cap price in a two-factor Gaussian HJM model. Here `m` is a maturity and `κs[m]` is the corresponding 
    cap rate. `T` contains time legs. `ν` and `β` are two parameters. 
"""
function capGaussianHJM2(m::Int, κs::Array{Float64,1}, prices::Array{Float64,1}, Ts::Array{Float64,1}, ν1, β1, ν2, β2, δ=0.5)
    value = 0.
    κ = κs[m]
    steps = Int(1/δ)
    for i=2:(steps*m)
        I = integral(ν1, β1, ν2, β2, Ts[i], Ts[i+1])
        d1 = (log(prices[i+1]/prices[i] * (1 + δ*κ)) + 0.5*I)/sqrt(I)
        d2 = (log(prices[i+1]/prices[i] * (1 + δ*κ)) - 0.5*I)/sqrt(I)
        cpl = prices[i]*cdf(W, -d2) - (1 + δ*κ)*prices[i+1]*cdf(W, -d1)
        value = value + cpl
    end
    return value
end


# Q12: price of an ATM cap with maturity in 30 years, semi-anual cash flows and first reset date in 6 months.
ν1 = 0.01
β1 = 0.3
ν2 = 0.02
β2 = 0.5
Ts = collect(0:δ:30)
m = 30 # maturity

capPrice30 = capGaussianHJM2(m, κs, prices, Ts, ν1, β1, ν2, β2, δ)
@info "capPrices30 = $(capPrice30)"
@info "capPrices30 = $(round(capPrice30*100, digits=2))"


# Compute forward rates from prices
Fs = [(prices[t]/prices[t+1] - 1)/δ for t=1:length(prices)-1]
@info Fs

"""
    capBlack(m, κs, prices, Ts, Fs, δ, σ)

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

function f!(F, x)
    F[1] = capBlack(m, κs, prices, Ts, Fs, δ, x[1]) - capPrice30
end

# solve for Black implied σ (the root of a univariable function)
solutionBlack = nlsolve(f!, [0.005], autodiff = :forward)
@info solutionBlack
sigmaBlack = solutionBlack.zero[1]
@info sigmaBlack
@info round(sigmaBlack*100, digits=2)


"""
    capBachelier(m, κs, prices, Ts, Fs, δ, σ)

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

# solve for Bachelier implied σ (the root of a univariable function)
# in basis points
function g!(F, x)
    F[1] = capBachelier(m, κs, prices, Ts, Fs, δ, x[1]) - capPrice30
end
solutionBachelier = nlsolve(g!, [0.005], autodiff = :forward)
@info solutionBachelier
sigmaBachelier = solutionBachelier.zero[1]
@info sigmaBachelier
@info round(sigmaBachelier*10_000, digits=2)
