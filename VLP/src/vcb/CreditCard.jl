using CSV
using DataFrames
using Statistics
using Flux
using Flux: @epochs
using Random
using EvalMetrics


df = DataFrame(CSV.File("dat/vcb/creditcard.csv"))
y = df[:, :Class]
Xdf = df[:, setdiff(names(df), ["Time", "Amount", "Class"])]

μ⃗ = mean.(eachcol(Xdf))
σ⃗ = std.(eachcol(Xdf))
A = combine(x -> (x .- μ⃗') ./ σ⃗', Xdf)
X = Matrix(A)'

# randomly permute the data set
Random.seed!(1234)
js = shuffle(1:length(y))
X = X[:, js]
y = y[js]

# training/test split
N_train = 100_000
N_test = 10_000
X_train, X_test = X[:, 1:N_train], X[:, (N_train + 1):(N_train + N_test)]
y_train, y_test = y[1:N_train], y[(N_train + 1):(N_train + N_test)]

# build data loaders; we use a large batch size to ensure that 
# each batch has several positive samples to learn from
dataset_train = Flux.Data.DataLoader((X_train, y_train), batchsize=2048)
dataset_test = Flux.Data.DataLoader((X_test, y_test), batchsize=2048)

# build model
model = Chain(Dense(28, 16, relu), Dropout(0.5), Dense(16, 1, σ))
loss(x, y) = Flux.Losses.binarycrossentropy(model(x), y)


function cbf()
	ℓ = sum(loss(b...) for b in dataset_train)
	@info "ℓ = $ℓ"
    ŷ_train = Iterators.flatten(map(x -> model(x), map(b -> first(b), dataset_train)))
    @info binary_eval_report(y_train, collect(ŷ_train))
    ŷ_test = Iterators.flatten(map(x -> model(x), map(b -> first(b), dataset_test)))
    @info binary_eval_report(y_test, collect(ŷ_test))
end

# perform optimization
optimizer = ADAM(1E-3)
@epochs 30 Flux.train!(loss, Flux.params(model), dataset_train, optimizer, cb = Flux.throttle(cbf, 60))

@info "Final evaluation on the training set: "
ŷ_train = Iterators.flatten(map(x -> model(x), map(b -> first(b), dataset_train)))
binary_eval_report(y_train, collect(ŷ_train))
cm_train = ConfusionMatrix(y_train, collect(ŷ_train) .>= 0.5)

@info "Final evaluation on the test set: "
ŷ_test = Iterators.flatten(map(x -> model(x), map(b -> first(b), dataset_test)))
binary_eval_report(y_test, collect(ŷ_test))
cm_test = ConfusionMatrix(y_test, collect(ŷ_test) .>= 0.5)

# plot the Precision-Recall and ROC plots
#using Plots
#prplot(y_train, ŷ_train)
#rocplot(y_train, ŷ_train)


# struct ConfusionMatrix{T<:Real}
#     p::T    # positive in target
#     n::T    # negative in target
#     tp::T   # correct positive prediction
#     tn::T   # correct negative prediction
#     fp::T   # (incorrect) positive prediction when target is negative
#     fn::T   # (incorrect) negative prediction when target is positive
# end
