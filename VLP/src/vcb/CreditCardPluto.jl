### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ efb0a844-aa8a-11eb-0e6f-9fcec40ab449
begin
	using CSV
	using DataFrames
	using Plots
	using StatsPlots
end

# ╔═╡ 85ebde02-61ec-47f0-8958-96b222423411
using Statistics

# ╔═╡ 9d201c71-604e-48e8-8292-0f766c3622bc
using Flux

# ╔═╡ bda456b4-ef24-4105-8323-67c46f8d3ec2
using Flux: @epochs

# ╔═╡ 92c6091b-51dd-478a-afe0-4b1f286875ac
using PlutoUI

# ╔═╡ 9cd0aae8-d2f7-43ec-95ef-a2057c5855b9
df = DataFrame(CSV.File("creditcard.csv"))

# ╔═╡ 706ba380-79b1-4210-ba54-2c0cd33f3f6b
describe(df)

# ╔═╡ 5ff9d873-8a0d-49c1-ad09-86127a33c2b5
nrow(df)

# ╔═╡ 02a73d16-3464-4a05-8647-8845a023ff81
ncol(df)

# ╔═╡ e63c626e-cf6a-44f1-aad7-894dc9b59053
names(df)

# ╔═╡ 3173ff6c-a386-48ef-8b41-70654f3d66ea
unique(df[:, :Class])

# ╔═╡ 0051c204-f3ee-4bbc-8393-d0a5d99672c0
numOnes = nrow(df[df.Class .== 0, :])

# ╔═╡ 54a7f804-d991-40a0-8341-f5fac74b4fd6
numZeros = nrow(df[df.Class .== 1, :])

# ╔═╡ 6c67a759-e6b1-4dc4-95ac-64a7700ac79d
numZeros / (numZeros + numOnes) * 100

# ╔═╡ 30351e2c-0c89-41fb-98b1-e7f301e5fe1b
@df df scatter(:V1, :V2, group=:Class, xlabel="V1", ylabel="V2", legend=:bottomright)

# ╔═╡ 62a34bfa-5054-41b1-945b-dd96af5980a6
@df df scatter(:V5, :V6, group=:Class, xlabel="V5", ylabel="V6", legend=:topright)

# ╔═╡ 5d2d2c8d-acad-48d5-8e79-21eb23be146c
@df df scatter(:V15, :V16, group=:Class, xlabel="V15", ylabel="V16", legend=:bottomright)

# ╔═╡ 6663a361-bc1c-4740-9aa8-01018380538d
ef = transform(df, :Amount => x -> log.(x .+ 1E-3))

# ╔═╡ 82a39e4f-f43d-4d22-a3a3-cc518db01457
@df ef histogram(:Amount_function, group=:Class)

# ╔═╡ d1258064-0f53-4baa-8f36-02617392d196
md"### Select target column"

# ╔═╡ 4d7ea712-ae15-420e-90d8-b1c2e75a3c6d
target = ef[:, :Class]

# ╔═╡ 6060e76a-99bf-4a6d-9f27-14c15504c9d7
md"### Select all columns other than `:Time`, `:Amount` and `:Class`"

# ╔═╡ 10275690-3f7d-470b-9ecc-4427f054a6ea
Xdf = df[:, setdiff(names(df), ["Time", "Amount", "Class"])]

# ╔═╡ 949440fc-80c9-4d35-9a08-ffb1f34a990d
md"### Normalize the feature data frame"

# ╔═╡ 67c48a3a-0690-4e2a-ac89-b99ad1856719
begin
	μ⃗ = mean.(eachcol(Xdf))
	σ⃗ = std.(eachcol(Xdf))
	combine(x -> (x .- μ⃗') ./ σ⃗', Xdf)
end

# ╔═╡ d12a527b-4584-445a-9903-6e9877cc2080
md"### Convert Xdf to design matrix of size `DxN`"

# ╔═╡ 14d7869f-6eb1-4165-a0d6-5b1222245bf5
X = Matrix(Xdf)'

# ╔═╡ 18372f42-d9e4-4cd4-ab3d-b6397efe0112
loader = Flux.Data.DataLoader((X, target), batchsize=64)

# ╔═╡ 2870b1b5-8a5e-4ca6-b875-fdb4ae93b2eb
first(loader)

# ╔═╡ a1b45eec-409f-4cc4-940b-940ae752caf6
md"### Build a MLP model for classification"

# ╔═╡ f091f40a-6e69-466b-b97e-0833b247f1a6
model = Chain(Dense(28, 16, relu), Dropout(0.5), Dense(16, 1))

# ╔═╡ a6f60712-deeb-4d14-94dd-879b736f9752
loss(x, y) = Flux.Losses.logitbinarycrossentropy(model(x), y)

# ╔═╡ 31e8088b-c48a-4cb6-9d1e-6bb575d6c049
optimizer = ADAM(1E-3)

# ╔═╡ 423f865e-1751-418b-80d2-edd9aaea1b0c
function cbf()
	ℓ= sum(loss(b...) for b in loader)
	println("ℓ = $ℓ")
end

# ╔═╡ 11e929e4-2836-4f48-884b-d2bcb984327a
with_terminal() do
	@epochs 2 Flux.train!(loss, Flux.params(model), loader, optimizer, cb = cbf)
end

# ╔═╡ Cell order:
# ╠═efb0a844-aa8a-11eb-0e6f-9fcec40ab449
# ╠═9cd0aae8-d2f7-43ec-95ef-a2057c5855b9
# ╠═706ba380-79b1-4210-ba54-2c0cd33f3f6b
# ╠═5ff9d873-8a0d-49c1-ad09-86127a33c2b5
# ╠═02a73d16-3464-4a05-8647-8845a023ff81
# ╠═e63c626e-cf6a-44f1-aad7-894dc9b59053
# ╠═3173ff6c-a386-48ef-8b41-70654f3d66ea
# ╠═0051c204-f3ee-4bbc-8393-d0a5d99672c0
# ╠═54a7f804-d991-40a0-8341-f5fac74b4fd6
# ╠═6c67a759-e6b1-4dc4-95ac-64a7700ac79d
# ╠═30351e2c-0c89-41fb-98b1-e7f301e5fe1b
# ╠═62a34bfa-5054-41b1-945b-dd96af5980a6
# ╠═5d2d2c8d-acad-48d5-8e79-21eb23be146c
# ╠═6663a361-bc1c-4740-9aa8-01018380538d
# ╠═82a39e4f-f43d-4d22-a3a3-cc518db01457
# ╠═d1258064-0f53-4baa-8f36-02617392d196
# ╠═4d7ea712-ae15-420e-90d8-b1c2e75a3c6d
# ╠═6060e76a-99bf-4a6d-9f27-14c15504c9d7
# ╠═10275690-3f7d-470b-9ecc-4427f054a6ea
# ╠═85ebde02-61ec-47f0-8958-96b222423411
# ╠═949440fc-80c9-4d35-9a08-ffb1f34a990d
# ╠═67c48a3a-0690-4e2a-ac89-b99ad1856719
# ╠═d12a527b-4584-445a-9903-6e9877cc2080
# ╠═14d7869f-6eb1-4165-a0d6-5b1222245bf5
# ╠═9d201c71-604e-48e8-8292-0f766c3622bc
# ╠═18372f42-d9e4-4cd4-ab3d-b6397efe0112
# ╠═2870b1b5-8a5e-4ca6-b875-fdb4ae93b2eb
# ╠═a1b45eec-409f-4cc4-940b-940ae752caf6
# ╠═f091f40a-6e69-466b-b97e-0833b247f1a6
# ╠═a6f60712-deeb-4d14-94dd-879b736f9752
# ╠═31e8088b-c48a-4cb6-9d1e-6bb575d6c049
# ╠═bda456b4-ef24-4105-8323-67c46f8d3ec2
# ╠═92c6091b-51dd-478a-afe0-4b1f286875ac
# ╠═423f865e-1751-418b-80d2-edd9aaea1b0c
# ╠═11e929e4-2836-4f48-884b-d2bcb984327a
