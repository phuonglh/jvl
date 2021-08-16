### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 5b620dc0-fb45-11eb-2cf5-79621d564f14
using DelimitedFiles

# ╔═╡ b35223d7-eac2-46ac-8c08-8dc072b7fdd4
using Optim

# ╔═╡ 64ef33ef-d583-45cd-818f-2d754c4eba15
function read(path)
	A = readdlm(path)
	y = Int.(A[:,1])
	X = hcat(ones(length(y)), A[:,2:end])
	return (X, y)
end

# ╔═╡ 18d43619-07f5-4d6c-a50f-7458286d7ef8
X, y = read("C:\\Users\\phuonglh\\courses\\iml\\dat\\iris.train")

# ╔═╡ e11bf725-c58b-4d1f-aadb-0c904880af8a
N, D = size(X)

# ╔═╡ f35db049-21ad-43cf-a356-0a174955cee4
K = length(unique(y))

# ╔═╡ d3e1ada4-4fb6-4214-bb18-ec7a2553481c
θ_0 = zeros(K,D)

# ╔═╡ dc0570b1-c666-4dc6-b948-6d59b281341c
θ_1 = rand(K,D)

# ╔═╡ 05bf9bb1-7ee8-438d-860b-6514c13c0f8c
function costSimple(θ) # unvectorized version
	ℓ = 0
	for i=1:N
		u = θ[y[i],:]' * X[i,:]
		v = 0
		for k=1:K
			v = v + exp(θ[k,:]' * X[i,:])
		end
		ℓ = ℓ + (u - log(v))
	end
	return -ℓ/N
end

# ╔═╡ 9f09c4b8-54ab-4020-9717-4b3fcb8267cd
costSimple(θ_0)

# ╔═╡ 1a672b95-59f4-4276-8645-fc7084d09687
costSimple(θ_1)

# ╔═╡ 11141f49-51e7-4c51-b6fc-db78dcccbb03
function cost(θ) # vectorized version of the cost function
	u = sum(θ[y,:] .* X, dims=2) 	# column vector of length N
	v = sum(exp.(θ * X'), dims=1)   # row vector of length N
	ℓ = sum(u' - log.(v))
	return -ℓ/N
end

# ╔═╡ dbde9b91-e02e-47be-9032-0f493af60037
cost(θ_0)

# ╔═╡ 120bc566-6e63-4d62-b4ca-ae95bb97d50d
cost(θ_1)

# ╔═╡ 6d7ef1f5-21c7-4e7a-bf9e-81f132593f1e
function grad(θ)
	∇J = zeros(K,D)
	δ = zeros(N,K)
	p = zeros(N,K)
	for k=1:K
		δ[:,k] = (y .== k)
		θk = repeat(θ[k,:]', N, 1)
		u = sum(θk .* X, dims=2)
		p[:,k] = exp.(u)
	end
	sp = sum(p, dims=2)
	p = p./repeat(sp,1,K)
	w = δ - p
	for k=1:K
		W = repeat(w[:,k],1,D)
		∇J[k,:] = sum(W .* X, dims=1)
		∇J[k,:] = -∇J[k,:] .+ 0.
	end
	return ∇J/N
end

# ╔═╡ e5461b6a-6b1a-46fb-b59a-acd8669f2052
grad(θ_0)

# ╔═╡ 1732d169-7287-470c-a7f4-aa9c2d13831f
grad(θ_1)

# ╔═╡ 8f525bb1-060f-4a34-9367-b70c8a800804
θ_1

# ╔═╡ 74a1bf75-36b8-412d-8824-ed7905e06b8d
vec(θ_1)

# ╔═╡ 828f9c66-ba89-49f1-bc28-a447a72608ef
v = vec(θ_1)

# ╔═╡ 35176229-120c-42e0-a6d8-961cd040f34e
reshape(v, 3, 5)

# ╔═╡ 43c11303-c2fa-46f5-b3f9-cda468446f28
θ_1

# ╔═╡ df2cbfdd-2b5a-45c9-9d64-ea976ff89986
function f(v)
	θ = reshape(v, K, D)
	return cost(θ)
end

# ╔═╡ 04663eb3-9f17-4df4-ad28-1c5ea375d38b
function g!(G, v)
	θ = reshape(v, K, D)
	∇J = grad(θ) # matrix 
	u = vec(∇J)
	for j=1:length(G) # note: we need to keep the same G memory
		G[j] = u[j]
	end
end

# ╔═╡ bc31be29-872a-47d6-9115-6f4861dda9af
function train(v0)
	optimize(f, g!, v0, LBFGS())
end

# ╔═╡ 21502009-b3f5-42e1-b1be-75095138c1fd
v0 = vec(θ_0)

# ╔═╡ 3655dda9-d1e5-4abf-b400-f5528111a68a
result_bfgs = train(v0)

# ╔═╡ 78693067-d481-47c7-b399-15f88b37dfe6
v_best = Optim.minimizer(result_bfgs)

# ╔═╡ aae4f1fc-f86f-4f4e-abd7-8abf66d78dd6
θ_best = reshape(v_best, K, D)

# ╔═╡ 99bca4a6-74fa-4145-8888-a1ce7882a8c6
grad(θ_best)

# ╔═╡ dd88657a-b9e3-4882-9e52-004642d36f30
sum(grad(θ_best))

# ╔═╡ c12adfbc-ff37-402d-a736-a9a1e537ad5b
function classify(θ, x)
	score = θ * x
	return argmax(score)
end

# ╔═╡ ed588beb-70b5-4a83-abf4-1b05d1b0e813
classify(θ_best, X[1,:])

# ╔═╡ dd20a320-ba3c-45d0-a0c2-86f65d33921b
classify(θ_best, X[end,:])

# ╔═╡ 843ba191-5e12-47c3-ac85-beeb9e52bbdd
function evaluate(θ, X, y)
	N = length(y)
	z = [classify(θ, X[i,:]) for i=1:N]
	return sum(z .== y)/N * 100
end

# ╔═╡ 0c82e293-31ff-4be0-b1d7-f42f9e106dfc
accuracy_train = evaluate(θ_best, X, y)

# ╔═╡ e11ddc75-7ca8-455a-a1c4-dc6610ed56ab
X_test, y_test = read("C:\\Users\\phuonglh\\courses\\iml\\dat\\iris.test")

# ╔═╡ 4fffe878-df8b-460c-a608-9e63f1794c0c
accuracy_test = evaluate(θ_best, X_test, y_test)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"

[compat]
Optim = "~1.4.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArrayInterface]]
deps = ["IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "cdb00a6fb50762255021e5571cf95df3e1797a51"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.23"

[[Artifacts]]
deps = ["Pkg"]
git-tree-sha1 = "c30985d8821e0cd73870b17b0ed0ce6dc44cb744"
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.3.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bdc0937269321858ab2a4f288486cb258b9a0af7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.3.0"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "344f143fa0ec67e47917848795ab19c6a455f32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.32.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8e695f735fca77e9708e795eda62afdb869cbb70"
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.3.4+0"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "85d2d9e2524da988bffaf2a381864e20d2dae08d"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.2.1"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8c8eac2af06ce35973c3eadb4ab3243076a408e7"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.1"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "8b3c09b56acaf3c0e581c66638b85c8650ee9dca"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.8.1"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "b5e930ac60b613ef3406da6d4f42c35d8dc51419"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.19"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[LibGit2]]
deps = ["Printf"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "3d682c07e6dd250ed082f883dc88aee7996bf2cc"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.0"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "0fb723cd8c45858c22169b2e42269e53271a6df7"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.7"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "4ea90bd5d3985ae1f9a908bd4500ae88921c5ce7"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "144bab5b1443545bc4e791536c9f1eacb4eed06a"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.1"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9db77584158d0ab52307f8c04f8e7c08ca76b5b3"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.3+4"

[[Optim]]
deps = ["Compat", "FillArrays", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "7863df65dbb2a0fa8f85fcaf0a41167640d2ebed"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.4.1"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "2276ac65f1e236e0a6ea70baff3f62ad4c625345"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.2"

[[Pkg]]
deps = ["Dates", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "UUIDs"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "62701892d172a2fa41a1f829f66d2b0db94a9a63"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.3.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "fed1ec1e65749c4d96fc20dd13bea72b55457e62"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.9"

[[TOML]]
deps = ["Dates"]
git-tree-sha1 = "44aaac2d2aec4a850302f9aa69127c74f0c3787e"
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[Test]]
deps = ["Distributed", "InteractiveUtils", "Logging", "Random"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
"""

# ╔═╡ Cell order:
# ╠═5b620dc0-fb45-11eb-2cf5-79621d564f14
# ╠═64ef33ef-d583-45cd-818f-2d754c4eba15
# ╠═18d43619-07f5-4d6c-a50f-7458286d7ef8
# ╠═e11bf725-c58b-4d1f-aadb-0c904880af8a
# ╠═f35db049-21ad-43cf-a356-0a174955cee4
# ╠═d3e1ada4-4fb6-4214-bb18-ec7a2553481c
# ╠═dc0570b1-c666-4dc6-b948-6d59b281341c
# ╠═05bf9bb1-7ee8-438d-860b-6514c13c0f8c
# ╠═9f09c4b8-54ab-4020-9717-4b3fcb8267cd
# ╠═1a672b95-59f4-4276-8645-fc7084d09687
# ╠═11141f49-51e7-4c51-b6fc-db78dcccbb03
# ╠═dbde9b91-e02e-47be-9032-0f493af60037
# ╠═120bc566-6e63-4d62-b4ca-ae95bb97d50d
# ╠═6d7ef1f5-21c7-4e7a-bf9e-81f132593f1e
# ╠═e5461b6a-6b1a-46fb-b59a-acd8669f2052
# ╠═1732d169-7287-470c-a7f4-aa9c2d13831f
# ╠═b35223d7-eac2-46ac-8c08-8dc072b7fdd4
# ╠═8f525bb1-060f-4a34-9367-b70c8a800804
# ╠═74a1bf75-36b8-412d-8824-ed7905e06b8d
# ╠═828f9c66-ba89-49f1-bc28-a447a72608ef
# ╠═35176229-120c-42e0-a6d8-961cd040f34e
# ╠═43c11303-c2fa-46f5-b3f9-cda468446f28
# ╠═df2cbfdd-2b5a-45c9-9d64-ea976ff89986
# ╠═04663eb3-9f17-4df4-ad28-1c5ea375d38b
# ╠═bc31be29-872a-47d6-9115-6f4861dda9af
# ╠═21502009-b3f5-42e1-b1be-75095138c1fd
# ╠═3655dda9-d1e5-4abf-b400-f5528111a68a
# ╠═78693067-d481-47c7-b399-15f88b37dfe6
# ╠═aae4f1fc-f86f-4f4e-abd7-8abf66d78dd6
# ╠═99bca4a6-74fa-4145-8888-a1ce7882a8c6
# ╠═dd88657a-b9e3-4882-9e52-004642d36f30
# ╠═c12adfbc-ff37-402d-a736-a9a1e537ad5b
# ╠═ed588beb-70b5-4a83-abf4-1b05d1b0e813
# ╠═dd20a320-ba3c-45d0-a0c2-86f65d33921b
# ╠═843ba191-5e12-47c3-ac85-beeb9e52bbdd
# ╠═0c82e293-31ff-4be0-b1d7-f42f9e106dfc
# ╠═e11ddc75-7ca8-455a-a1c4-dc6610ed56ab
# ╠═4fffe878-df8b-460c-a608-9e63f1794c0c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
