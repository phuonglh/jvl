# wdbc.txt
# read data from "wdbc.txt" into matrix X and column vector y.

using DelimitedFiles
using Statistics

function readData(path)
    A = readdlm(path, ',')
    X = A[:,3:end]
    y = Int.(A[:,2]) .+ 1
    return (X, y)
end

function train(X, y)
    K = length(unique(y))
    N, D = size(X)
    θ = zeros(K)
    μ = zeros(K,D)
    σ = zeros(K,D)
    for k=1:K
        N_k = sum(y .== k)
        θ[k] = N_k/N
        X_k = X[y .== k,:]
        μ[k,:] = mean(X_k, dims=1)
        σ[k,:] = std(X_k, dims=1)
    end
    return (θ, μ, σ)
end

# prediction/classification
function classify(x::Array{Float64,1}, θ, μ, σ)
    K, D = size(μ)
    score = zeros(K)
    for k=1:K
        score[k] = log(θ[k]) + sum(-log(σ[k,j]*sqrt(2π)) - (x[j] - μ[k,j])^2/(2σ[k,j]^2) for j=1:D) 
    end
    return argmax(score)
end

function classify(X::Array{Float64,2}, θ, μ, σ)
    return [classify(X[i,:], θ, μ, σ) for i=1:size(X,1)]
    # map(i -> classify(X[i,:], θ, μ, σ), collect(1:size(X,1))
end 

function evaluate(X::Array{Float64,2}, y::Array{Int,1}, θ, μ, σ)
    z = classify(X, θ, μ, σ)
    return sum(z .== y)/length(y)
end

# main program

# Step 0: read data 
(X, y) = readData("/home/phuonglh/jvl/VLP/src/iml/wdbc.txt")
# Step 1: parameter estimation
(θ, μ, σ) = train(X, y)
# Step 2: evaluate the classification accuracy of the trained model
accuracy = evaluate(X, y, θ, μ, σ)
@info accuracy