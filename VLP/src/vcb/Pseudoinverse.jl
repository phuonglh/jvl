# Interest Rate Models, w3
# Pseudoinverse method
# phuonglh, July 2021

using Dates
using LinearAlgebra

t0 = Date(2012, 10, 1)

LIBORs = [(0.15, Date(2012, 10, 2)), (0.21, Date(2012, 11, 5)), (0.36, Date(2013, 1, 3))]
futures = [(99.68, Date(2013, 3, 20)), (99.67, Date(2013, 6, 19)), (99.65, Date(2013, 9, 18)), (99.64, Date(2013, 12, 18)), (99.62, Date(2014, 3, 19))]
swaps = [(0.36, Date(2014, 10, 3)), (0.43, Date(2015, 10, 5)), (0.56, Date(2016, 10, 3)), (0.75, Date(2017, 10, 3)), 
    (1.17, Date(2019, 10, 3)), (1.68, Date(2022, 10, 3)), (2.19, Date(2027, 10, 4)), (2.40, Date(2032, 10, 4)), (2.58, Date(2042, 10, 3))]


# create a map month/year to date: this is necessary to create time legs for futures
month2Date = Dict{Tuple{Year, Month}, Date}()
month2Date[(Dates.Year(2012), Dates.Month(12))] = Date(2012, 12, 19)
month2Date[(Dates.Year(2013), Dates.Month(3))] = Date(2013, 3, 20)
month2Date[(Dates.Year(2013), Dates.Month(6))] = Date(2013, 6, 19)
month2Date[(Dates.Year(2013), Dates.Month(9))] = Date(2013, 9, 18)
month2Date[(Dates.Year(2013), Dates.Month(12))] = Date(2013, 12, 18)
month2Date[(Dates.Year(2014), Dates.Month(3))] = Date(2014, 3, 19)

# create a map to map a year to a date: this is necessary to create time legs for swap contracts
year2Date = Dict{Year, Date}(Dates.Year(p[2]) => p[2] for p in swaps)
year2Date[Dates.Year(2018)] = Date(2018, 10, 3)
year2Date[Dates.Year(2020)] = Date(2020, 10, 3)
year2Date[Dates.Year(2021)] = Date(2021, 10, 3)
for y=2023:2026
    year2Date[Dates.Year(y)] = Date(y, 10, 3)
end
for y=2028:2041
    year2Date[Dates.Year(y)] = Date(y, 10, 4)
end

struct Instrument
    kind
    rate
    date
    cash
end

function cashLIBOR(rate, T)
    flow = Array{Tuple{Date,Float64},1}()
    push!(flow, (T, 1 + Dates.value(T - t0)/360*rate/100))
    return flow
end

function cashFutures(price, Tn)
    flow = Array{Tuple{Date,Float64},1}()
    rate = 1 - price/100
    date1 = Tn-Dates.Month(3) # futures are defined in quarter
    (y1, m1) = (Dates.Year(date1), Dates.Month(date1))
    date1 = get(month2Date, (y1, m1), date1)
    push!(flow, (date1, -1))
    (y2, m2) = (Dates.Year(Tn), Dates.Month(Tn))
    date2 = get(month2Date, (y2, m2), Tn)
    push!(flow, (date2, 1 + Dates.value(date2 - t0)/360*rate))
    return flow
end

function cashSwap(rate, Tn)
    flow = Array{Tuple{Date,Float64},1}()
    range = collect(t0:Dates.Year(1):Tn)
    for j=1:length(range)-1
        y1 = Dates.Year(range[j])
        y2 = Dates.Year(range[j+1])
        date1 = get(year2Date, y1, range[j])
        date2 = get(year2Date, y2, range[j+1])
        push!(flow, (date2, Dates.value(date2 - date1)/360*rate/100))
    end
    # at the maturity, we need to add 1 to the cash value
    flow[end] = (flow[end][1], 1 + flow[end][2])
    @info flow[end]
    return flow
end

# build all cash flows, each corresponds to a row of the cash flow matrix C.
elements = Array{Instrument,1}()
# LIBORs
for d in LIBORs
    push!(elements, Instrument(:l, d[1], d[2], cashLIBOR(d[1], d[2])))
end

# futures
futureDates = map(d -> d[2], futures)
for d in futures
    flow = cashFutures(d[1], d[2])
    push!(elements, Instrument(:f, d[1], d[2], flow))
end


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
# this includes only LIBORs and futures time legs
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
push!(p, map(x -> 1., LIBORs)...)
push!(p, map(x -> 0., futures)...)
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
prices = zeros(N)
p0 = 1.0
for i=1:N
    prices[i] = p0 + Δ[i]/δ[i]
    p0 = prices[i]
end
prices
