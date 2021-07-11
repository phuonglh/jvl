# Interest Rate Models
# July 2021
# Estimating Term Structure
# Graded quiz: Q1 

using Dates


# time slots, 31 elements
dateSt = [
    "10/3/2012",
    "10/3/2013", "10/3/2014", "10/5/2015", "10/3/2016", "10/3/2017", "10/3/2018",
    "10/3/2019", "10/3/2020", "10/3/2021", "10/3/2022", "10/3/2023", "10/3/2024",
    "10/3/2025", "10/3/2026", "10/4/2027", "10/4/2028", "10/4/2029", "10/4/2030",
    "10/4/2031", "10/4/2032", "10/4/2033", "10/4/2034", "10/4/2035", "10/4/2036",
    "10/4/2037", "10/4/2038", "10/4/2039", "10/4/2040", "10/4/2041", "10/3/2042"
]

df = DateFormat("m/d/y")
dates = map(s -> Date(s, df), dateSt)

# swap rates (28 elements)
rates = [
    0.586000, 0.752000, 0.942000, 1.133000, 1.324000,
    1.462586, 1.600793, 1.739000, 1.824107, 1.909447, 1.994553, 2.079660,
    2.165000, 2.188038, 2.211013, 2.233987, 2.256962, 2.280000, 2.285199,
    2.290397, 2.295596, 2.300809, 2.306007, 2.311206, 2.316404, 2.321617,
    2.326816, 2.332000
]

prices = zeros(length(rates) + 2) # 30 elements
prices[1] = 0.997528314 # 10/3/2013
prices[2] = 0.990426067 # 10/3/2014

for n=4:length(prices)+1
    p = 0
    for i=2:n-1
        δ = Dates.value(dates[i] - dates[i-1])/360
        p = p + δ*prices[i-1]
    end
    r = rates[n-3]/100
    d = Dates.value(dates[n] - dates[n-1])/360
    prices[n-1] = (1-r*p)/(1 + r*d)
    @info rates[n-3], dates[n], prices[n-1]
end

# prices
u, v = prices[end-1], prices[end]
@info dates[end-1], dates[end]
δ = Dates.value(dates[end] - dates[end-1])/360
F = (1/δ)*(u/v - 1)*100
@info prices
@info string("F = ", F) 
 
# compute the resulting forward curve
N = length(prices) - 1
forwardRates = zeros(N)
for n=1:N
    d = Dates.value(dates[n+2] - dates[n+1])/360
    F = (1/d)*(prices[n]/prices[n+1] - 1)*100
    forwardRates[n] = F
    @info dates[n+1], prices[n], forwardRates[n]
end

forwardRates
