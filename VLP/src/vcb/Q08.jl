r0 = 0.06
θ = 0.08 # mean reversion level
κ = 0.86 # speed of mean reversion (kappa)
σ = 0.01 # volatility
T0 = 1   #
T1 = 2


# variance = int_0^t | v(s,T0) - v(s,T1)|^2 ds
# = 

v2 = (σ/κ)^2*(exp(-κ*T0) - exp(-κ*T1))^2*(exp(2*κ*T0) - 1)/(2*κ)
bps = sqrt(v2)*10^4
@info bps
@info round(bps, digits=2)