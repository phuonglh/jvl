# Lorimier's method for estimating a smooth forward curve.
# Interest Rate Models, w3
# phuonglh@gmail.com, July 2021.

"""
    scalarProduct(x, y)

    The scalar product in the Hilbert space
"""
function scalarProduct(x, y)
    return -min(x,y)^3/6 + x*y*min(x,y)/2 + x*y
end

function scalarProduct(T::Array{Int})
    N = length(T)
    H = zeros(N, N)
    for i=1:N, j=1:N
        H[i,j] = scalarProduct(T[i], T[j])
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
A[2:end,2:end] = H
for i=2:N
    A[i,i] = A[i,i] + 1/α
end

v = map((x, y) -> x*y, Ts, ys)
prepend!(v, 0)

β = inv(A)*v

