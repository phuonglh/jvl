# Multi-period Binomial Model for Option Pricing
# phuonglh@gmail.com

# The following parameters are for testing, the prices C0 must be 5.21
# T = 0.5 # years
# S0 = 100 # initial price
# σ = 0.2  # volatility
# r = 0.02 # 
# c = 0.01 # dividend yield
# n = 10   # number of periods
# K = 100  # strike price
# call = -1 # use `-1` for put option

# For the questions from 1 to 5. 
T = 0.25 # years
S0 = 100 # initial price
σ = 0.3  # volatility
r = 0.02 # 
c = 0.01 # dividend yield
n = 15   # number of periods
K = 110  # strike price
call = -1 # use `-1` for put option

# For the questions from 6 to 7
# T = 0.25*10/15 # years
# S0 = 100 # initial price
# σ = 0.3  # volatility
# r = 0.02 # 
# c = 0.01 # dividend yield
# n = 10   # number of periods
# K = 110  # strike price
# call = 1 # use `-1` for put option

# For question 8: Need to run two times, setting call = 1 or call = -1
# T = 0.25*15/10 # years
# S0 = 100 # initial price
# σ = 0.3  # volatility
# r = 0.02 # 
# c = 0.01 # dividend yield
# n = 10   # number of periods
# K = 100  # strike price
# call = -1 # use `-1` for put option


# compute up and down prob.
u = exp(σ * sqrt(T/n))
d = 1/u
q = (exp((r - c)*T/n) - d)/(u - d)

# compute the stock lattice of shape nxn 
stocks = zeros(n+1, n+1)
for t=1:n
    for i=0:t
        stocks[i+1,t] = S0*u^(t-i)*d^(i)
    end
end

# compute an option price at time t using two known values (u, v) at time t+1.
# u is a upper value, v is a lower value
f(u, v) = exp(-r*T/n)*(q*u + (1-q)*v)

options = zeros(n+1, n+1)
# fill the last column
for i=1:n+1
    options[i, n+1] = max(call * (stocks[i,n] - K), 0)
end
# fill columns n, n-1,...,1 backwards
for t=n:-1:1
    for i=1:t
        options[i,t] = f(options[i,t+1], options[i+1,t+1])
    end
end

round.(stocks, digits=2)
round.(options, digits=2)

# compare payoff values to find the earliest period
payoff0 = max.(stocks .- K, 0)
delta0 = max.(payoff0 - options, 0)


## From the result tables:
P0 = 12.3051 
C0 = 2.60408

a = P0 + S0 * exp(-c*T/n) 
b = C0 + K * exp(-r*T/n)

@info a, b

# u is a upper value, v is a lower value
g(u, v) = q*u + (1-q)*v
# compute the future lattice
futures = zeros(n+1, n+1)
# fill the last column
futures[:, n+1] = stocks[:, n]
# fill columns n, n-1,...,1 backwards
for t=n:-1:1
    for i=1:t
        futures[i,t] = g(futures[i,t+1], futures[i+1,t+1])
    end
end

# compare payoff values to find the earliest period
payoff1 = max.(futures .- K, 0)
delta1 = max.(payoff1 - options, 0)

