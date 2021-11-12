
κ = 0.2
σ = 0.1
r0 = 0.05
γ = sqrt(κ^2 + 2*σ^2)

function B(t)
    return 2*(exp(γ*t) - 1)/((γ + κ)*(exp(γ*t) - 1) + 2*γ)
end

function A(t, θ)
    u = 2*γ*exp((γ+κ)*t/2)/((γ + κ)*(exp(γ*t) - 1) + 2γ)
    return (-2*κ*θ/σ^2)*log(u)
end

function solve()
    right = log(1.05) - B(1)*r0
    u = 2*γ*exp((γ+κ)/2)/((γ + κ)*(exp(γ) - 1) + 2γ)
    θ = -right*σ^2/(2*κ*log(u))
    return θ
end

θ = solve()
@info θ
@info round(θ*100, digits=2)
price = exp(-A(1, θ) - B(1)*r0)
@info price