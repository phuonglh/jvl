## Set model parameters
r0 = 0.06
θ = 0.08 # mean reversion level
κ = 0.86 # speed of mean reversion (kappa)
σ = 0.01 # volatility
t = 0    # valuation date
T0 = 1.   #
T1 = 1.25 

# Zero-coupon bond prices in Vasicek model 
function zcPrice(r0, κ, θ, σ, T, t)
    u = (1/κ)*(1-exp(-κ*(T-t))) 
    v = exp((θ - 0.5*σ^2/a^2)*(u - T + t) - (σ^2) / (4*κ) * (u^2))
    return (v*exp(-u*r0))
end

p0 = zcPrice(r0, κ, θ, σ, T0, t)
p1 = zcPrice(r0, κ, θ, σ, T1, t)

@info p0
@info p1

# Future derivatives calibration
# for the 1-factor HJM model 
function integral(ν, β, t0, t1)
    return (ν/β)^2 * (exp(-β*t0) - exp(-β*t1))^2 * (exp(2*β*t0) - 1)/(2*β)
end
