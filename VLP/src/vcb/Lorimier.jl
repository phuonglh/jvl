# Lorimier's method for estimating a smooth forward curve.
# Interest Rate Models, w3
# phuonglh@gmail.com, July 2021.

"""
    scalarProduct(x, y)

    The scalar product in the Hilbert space
"""
function scalarProduct(x, y)
    return -min(x,y)^3/6 + x*y*(1 + min(x,y)/2)
end

function scalarProduct(Ts::Array{Int})
    N = length(Ts)
    H = zeros(N, N)
    for i=1:N, j=i:N
        H[i,j] = scalarProduct(Ts[i], Ts[j])
        H[j,i] = H[i,j]
    end
    return H
end

Ts = [2, 3, 4, 5, 7, 10, 20, 30]
H = scalarProduct(Ts)

ys = [-0.79, -0.73, -0.65, -0.55, -0.33, -0.04, 0.54, 0.73]

# build matrix A
α = 0.1 
N = length(Ts)+1
A = zeros(N,N)
A[1,1] = 0
A[1,2:N] = Ts
A[2:N,1] = Ts
A[2:N,2:N] = H
for i=2:N
    A[i,i] = A[i,i] + 1/α
end

v = map((x, y) -> x*y, Ts, ys)
prepend!(v, 0)

β = inv(A)*v

# quadratic spline function
function h(T, u)
    return T + T*min(T,u) - (min(T, u))^2/2
end

# estimated forward curve at time u
function f(u)
    hs = map(T -> h(T,u), Ts)
    return β[1] + β[2:N]'*hs
end

# compute the estimated yield for maturity date u
function yield(u)
    Is = map(Ti -> scalarProduct(Ti, u), Ts)
    return β[1] + β[2:N]'*Is/u
end

fs = [f(u) for u=1:30]
estimated_yields = [yield(T) for T=1:length(fs)]

using Plots
plot(estimated_yields, marker=:+, label="estimated", xlabel="Time to maturity T", ylabel="Yield [%]", legend=:bottomright)
plot!(Ts, ys, marker=:o, label="real")
#Y6 = estimated_yields[6] # -0.4137772123302091
