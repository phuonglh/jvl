using DelimitedFiles
using Plots
using Optim

"""
	readData(path)
	path: path to a file (wdbc.txt)
	return X: matrix of shape N x D, y: vector of length N.
"""
function readData(path)
	A = readdlm(path, ',')
	y = Int.(A[:,2])
	X = hcat(ones(length(y)), A[:,3:12])
	return (X, y)
end

"Sigmoid function"
σ(z) = 1 ./ (1 .+ exp.(-z))

function plotSigmoid()
	z = -5:0.01:5
	g = σ(z)
	plot(z, g, legend=false, xlabel="z", ylabel="σ", title="Logistic Function")
end

"""
	J(X, y, θ): cost function. 
	X: matrix of size N x (D+1), y: vector of length N.
	θ: vector of length (D+1)
	return a real number: J(θ): R^{D+1} -> R
"""
function J(X::Array{Float64,2}, y::Array{Int}, θ::Array{Float64}, λ=0.0)::Float64
	N = length(y)
	u = θ[2:end]
	return (-1/N)*(y'*log.(σ(X*θ)) + (1 .- y)'*log.(1 .- σ(X*θ))) + λ*u'*u/(2*N)
end

function ∇J(X::Array{Float64,2}, y::Array{Int}, θ::Array{Float64}, λ=0.0)::Array{Float64}
	# return vector gradient of length (D+1)
	N = length(y)
	u = copy(θ)
	u[1] = 0.
	return X'*(σ(X*θ) - y)/N + λ*u/N
end

# For using with Optim.jl 

function J2(θ)
	N = length(y)
	u = θ[2:end]
	return (-1/N)*(y'*log.(σ(X*θ)) + (1 .- y)'*log.(1 .- σ(X*θ))) + λ*u'*u/(2*N)
end

function ∇J2!(G, θ)
	N = length(y)
	u = copy(θ)
	u[1] = 0
	g = X'*(σ(X*θ) - y)/N + λ*u/N	
	for j=1:length(G)
		G[j] = g[j]
	end
end

X, y = readData("wdbc.txt")
θ_0 = zeros(size(X, 2))
λ = 0.1

# Nelder-Mead algorithm
# result_nm = optimize(J2, θ_0)
# θ_best_nm = Optim.minimizer(result_nm)

# Gradient Descent algorithm
# result_gd = optimize(J2, ∇J2!, θ_0, GradientDescent())
# θ_best_gd = Optim.minimizer(result_gd)

# L-BFGS algorithm
result_lbfgs = optimize(J2, ∇J2!, θ_0, LBFGS())
# θ_best_lbfgs = Optim.minimizer(result_lbfgs)

