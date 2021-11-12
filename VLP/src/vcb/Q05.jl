# Final Quiz, 5, 6.
# phuonglh@gmail.com, July 2021

using Dates
using LinearAlgebra

t0 = Date(1996, 9, 4)
LIBORs = [
    (10.0, Date(1996, 11, 15), Date(1996, 11, 15), 103.82),
    (9.75, Date(1997, 1, 19), Date(1998, 1, 19), 106.04),
    (12.25, Date(1996, 9, 26), Date(1999, 3, 26), 118.44),
    (9., Date(1997, 3, 3), Date(2000, 3, 3), 106.28),
    (7., Date(1996, 11, 6), Date(2001, 11, 6), 101.15),
    (9.75, Date(1997, 2, 27), Date(2002, 8, 27), 111.06),
    (8.5, Date(1996, 12, 7), Date(2005, 12, 7), 106.24),
    (7.75, Date(1997, 3, 8), Date(2006, 9, 8), 98.49),
    (9., Date(1996, 10, 13), Date(2008, 10, 13), 110.87)
]

struct Instrument
    kind
    rate
    date
    cash
end

# 30/360 day count convention
function delta(T0, T1)
    d1, m1, y1 = Dates.value.([Dates.Day(T0), Dates.Month(T0), Dates.Year(T0)])
    d2, m2, y2 = Dates.value.([Dates.Day(T1), Dates.Month(T1), Dates.Year(T1)])
    return (min(d2, 30) + max(0, 30-d1))/360 + (m2 - m1 - 1)/12 + y2 - y1
    # return Dates.value(T1-T0)/365
end

function cashLIBOR(rate, T0, T1)
    flow = Array{Tuple{Date,Float64},1}()
    range = T0:Dates.Month(6):T1
    for t=1:length(range)-1
        push!(flow, (range[t], delta(range[t], range[t+1])*rate))
    end
    push!(flow, (T1, 100 + delta(T1, T1+Dates.Month(6))*rate))
    @info flow
    return flow
end


# determine cashflow dates 
dates = Array{Date,1}()
for tuple in LIBORs
    for d in tuple[2]:Dates.Month(6):tuple[3]
        push!(dates, d)
    end
end
unique!(dates)
sort!(dates)

# build all cash flows, each corresponds to a row of the cash flow matrix C.
elements = Array{Instrument,1}()
for tuple in LIBORs
    rate = tuple[1]
    start, maturity = tuple[2], tuple[3]
    push!(elements, Instrument(:l, rate, maturity, cashLIBOR(rate, start, maturity)))
end

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

# market price vector p, use the dirty prices
p = Array{Float64,1}()
# push!(p, map(x -> 100., LIBORs)...)
push!(p, map(x -> x[4], LIBORs)...)

# construct matrices
δ = zeros(N)
δ[1] = 1/sqrt(delta(t0, dates[1]))
for t=1:N-1
    δ[t+1] = 1/sqrt(delta(dates[t], dates[t+1]))
end

W = Diagonal(δ)
M = Bidiagonal(ones(N), zeros(N-1) .- 1, :L)

# compute the best Δ*
a = zeros(N); a[1] = 1.
A = C*inv(M)*inv(W)
Δ = A'*inv(A*A')*(p - C*inv(M)*a)

# infer price vector d from Δ
# function pricing(Δ, δ)
#     prices = zeros(N)
#     p0 = 1.0
#     for i=1:N
#         prices[i] = p0 + Δ[i]/δ[i]
#         p0 = prices[i]
#     end
#     return prices
# end

# prices = pricing(Δ, δ)

# the price vector d can be computed as follows:
prices = inv(M)*(inv(W)*Δ + a)

portfolioDates = [Date(2002, 8, 27), Date(2005, 12, 7), Date(2006, 9, 8), Date(2008, 10, 13)]
cashflow = [80, 100, 60, 250]

# prices at portfolio dates
is = map(date -> dateIndex[date], portfolioDates)
ps = map(i -> prices[i], is)
portforlioValue = sum(map((a, b) -> a * b, ps, cashflow))
@info portforlioValue
@info round(portforlioValue, digits=2)

# Q6
# compute the derivative of discount price w.r.t bond price ∂d/∂p
# this is a 104 x 9 matrix 
U = inv(M)*inv(W)*A'*inv(A*A')

# The change of cash flow of the porfolio when there is not Bond 1 
s = sum(U[is,1] .* cashflow)
# Bond 1 has cash flow at time 4, if it is included in the portfolio
# then its contribution to the cash flows must be equal s, so the unit n1 
# is computed simply as follows:
n1 = -s/U[4,1]/105
@info n1
@info round(n1, digits=2)
