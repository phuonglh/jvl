# Interest Rate Models, w3
# Pseudoinverse method
# phuonglh, July 2021

using Dates

t0 = Date(2012, 10, 1)

LIBORs = [(0.15, Date(2012, 10, 2)), (0.21, Date(2012, 11, 5)), (0.36, Date(2013, 1, 3))]
futures = [(99.68, Date(2013, 3, 20)), (99.67, Date(2013, 6, 19)), (99.65, Date(2013, 9, 18)), (99.64, Date(2013, 12, 18)), (99.62, Date(2014, 3, 19))]
swaps = [(0.36, Date(2014, 10, 3)), (0.43, Date(2015, 10, 5)), (0.56, Date(2016, 10, 3)), (0.75, Date(2017, 10, 3)), (1.17, Date(2019, 10, 3)), (1.68, Date(2022, 10, 3)), (2.19, Date(2027, 10, 4)), (2.40, Date(2032, 10, 4)), (2.58, Date(2042, 10, 3))]

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

function cashFutures(price, T1, T2)
    flow = Array{Tuple{Date,Float64},1}()
    rate = 1 - price/100
    if (T1 >= t0)
        push!(flow, (T1, -1))
    end
    if (T2 >= t0)
        push!(flow, (T2, 1 + Dates.value(T2 - t0)/360*rate))
    end
    return flow
end

function cashSwap(rate, T0, Tn)
    flow = Array{Tuple{Date,Float64},1}()
    range = collect(T0:Dates.Year(1):Tn+Dates.Year(1))
    for j=1:length(range)-1
        push!(flow, (range[j], Dates.value(range[j+1] - range[j])/360*rate/100))
    end
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
    flow = cashFutures(d[1], d[2] - Dates.Year(1) + Dates.Day(1), d[2])
    push!(elements, Instrument(:f, d[1], d[2], flow))
end

# swaps
# swapDates = map(d -> d[2], swaps)
# Tn = swapDates[end]
# for d in swaps
#     flow = cashSwap(d[1], d[2], Tn)
#     push!(elements, Instrument(:s, d[1], d[2], flow))
# end

function getDates(instrument)
    map(pair -> pair[1], instrument.cash)
end

# time marks (columns of the cash flow matrix C)
dates = Iterators.flatten(map(instrument -> getDates(instrument), elements))

