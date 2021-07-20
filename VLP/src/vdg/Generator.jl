# Vietnamese diacritics generation
# (C) phuonglh@gmail.com, 2021

module VDG

using Base.Iterators: sort!
using Base.Iterators: push!
using Base.Iterators: isempty
using FLoops: Iterators
using CSV
using DataFrames
using Flux
using Flux: @epochs
using BSON: @save, @load

using FLoops
using Random

include("DiacriticsRemoval.jl")
include("Utils.jl")

options = Dict{Symbol,Any}(
    :dataPath => string(pwd(), "/dat/qas/monre.csv"),
    :labelPath => string(pwd(), "/dat/vdg/label.txt"),
    :alphabetPath => string(pwd(), "/dat/vdg/alphabet.txt"),
    :modelPath => string(pwd(), "/dat/vdg/vdg.bson"),
    :maxSequenceLength => 64,
    :batchSize => 32,
    :numEpochs => 10,
    :unkChar => 'X',
    :padChar => 'P',
    :gpu => false,
    :hiddenSize => 64,
    :split => [0.8, 0.2]
)

"""
  vectorize(text, alphabet, label=Array{Char,1}())

  Transforms a sentence into an array of vectors based on an alphabet and 
  a label set for training. In training mode, `label` is given and this function 
  returns a pair of matrix `(x, y)` where `x` is a matrix of size `|alphabet| x maxSequenceLength` 
  and `y` is a matrix of size `|label| x maxSequenceLength`. 
  In test mode, this function returns only `x` matrix.
"""
function vectorize(text::String, alphabet::Array{Char}, label=Array{Char,1}())
    # truncate or pad input sequences to have the same maxSequenceLength
    n = options[:maxSequenceLength]
    x = removeDiacritics(text)
    # slice x into subarrays of equal length
    xs = collect(Iterators.partition(x, n))
    # pad the last subarray with the pad character
    ps = fill(options[:padChar], n - length(xs[end]))
    xs[end] = vcat(xs[end], ps)
    # now all the subarrays in xs is of the same length, 
    # convert them into one-hot matrices of size (|alphabet| x maxSeqLen)
    Xs = map(x -> Float32.(Flux.onehotbatch(x, alphabet)), xs)
    if (!isempty(label)) # training mode
        texte = map(c -> c ∈ keys(charMap) ? c : options[:unkChar], text)
        ys = collect(Iterators.partition(texte, n))
        ys[end] = vcat(ys[end], ps)
        Ys = map(y -> Float32.(Flux.onehotbatch(y, label)), ys)
        return (Xs, Ys)
    else
        return Xs
    end
end

"""
    evaluate(model, Xb, Yb)

    Evaluates the accuracy of a mini-batch, return the total labels in the batch and the number 
    of correctly predicted labels. This function is used in training for showing performance score.
"""
function evaluate(model, Xb, Yb)::Tuple{Int,Int}
    batchSize = length(Xb)
    as = map(X -> Flux.onecold(model(X)), Xb)
    bs = map(Y -> Flux.onecold(Y), Yb)
    # find the real length of target sequence (without padding symbol of index 1)
    total = 0
    correct = 0
    padding = 1
    for i = 1:batchSize
        t = options[:maxSequenceLength]
        while t > 0 && bs[i][t] == padding
            t = t - 1
        end
        total = total + t
        correct = correct + sum(as[i][1:t] .== bs[i][1:t])
    end
    Flux.reset!(model)
    return (total, correct)
end

function train(options)
    # read input data frame and create inp/out sequences
    df = CSV.File(options[:dataPath]) |> DataFrame
    ys = map(y -> lowercase(y), df[:, :question])
    # create and save alphabet index
    alphabetY = unique(join(ys))
    alphabet = unique(join(alphabetY, values(charMap)))
    prepend!(alphabet, options[:padChar])
    charIndex = Dict{Char,Int}(c => i for (i, c) in enumerate(alphabet))
    saveIndex(charIndex, options[:alphabetPath])
    # create and save label index
    label = collect(keys(charMap))
    prepend!(label, options[:unkChar])
    prepend!(label, options[:padChar])
    labelIndex = Dict{Char,Int}(c => i for (i, c) in enumerate(label))
    saveIndex(labelIndex, options[:labelPath])
    # create training sequences
    XYs = map(y -> vectorize(y, alphabet, label), ys)
    Xs = collect(Iterators.flatten(map(xy -> xy[1], XYs)))
    Ys = collect(Iterators.flatten(map(xy -> xy[2], XYs)))
    # batch inp/out sequences
    Xbs = collect(Iterators.partition(Xs, options[:batchSize]))
    Ybs = collect(Iterators.partition(Ys, options[:batchSize]))    
    if options[:gpu] 
        @info "Bringing data to GPU..."
        Xbs = map(t -> gpu.(t), Xbs)
        Ybs = map(t -> gpu.(t), Ybs)
    end
    dataset = collect(zip(Xbs, Ybs))
    @info "#(batches) = $(length(dataset))"
    @info "typeof(X1) = $(typeof(Xbs[1]))" 
    @info "typeof(Y1) = $(typeof(Ybs[1]))" 
    # split training/test parts
    Random.seed!(220712)
    n = length(Xbs)
    is = shuffle(1:n)
    dataset = dataset[is]
    j = Int(round(n*options[:split][2]))
    dataset_test = dataset[1:j]       # test part
    dataset_train = dataset[j+1:end]  # training part

    # define a model
    model = Chain(
        GRU(length(alphabet), options[:hiddenSize]), 
        Dense(options[:hiddenSize], length(label))
    )
    @info model
    # compute the loss of the model on a batch
    function loss(Xb, Yb)
        batchSize = length(Xb)
        value = sum(Flux.logitcrossentropy(model(Xb[i]), Yb[i]) for i=1:batchSize)
        Flux.reset!(model)
        return value
    end
    optimizer = ADAM()
    
    accuracy_test, accuracy_train = Array{Float64,1}(), Array{Float64,1}()
    Js = Array{Float64,1}()
    function evalcb() 
        J = sum(loss(dataset_train[i]...) for i=1:length(dataset_train))
        push!(Js, J)
        pairs_test = [evaluate(model, dataset_test[i]...) for i=1:length(dataset_test)]
        u, v = reduce(((a, b), (c, d)) -> (a + c, b + d), pairs_test)
        push!(accuracy_test, v/u)
        pairs_train = [evaluate(model, dataset_train[i]...) for i=1:length(dataset_train)]
        u, v = reduce(((a, b), (c, d)) -> (a + c, b + d), pairs_train)
        push!(accuracy_train, v/u)
        @info "loss = $J, accuracy_test = $(accuracy_test[end]), accuracy_train = $(accuracy_train[end])"
    end
    @elapsed @epochs options[:numEpochs] Flux.train!(loss, params(model), dataset_train, optimizer, cb = Flux.throttle(evalcb, 60))
    if (options[:gpu])
        model = model |> cpu
    end
    @save options[:modelPath] model
    # using Plots
    # plot(1:length(Js), accuracy_test, accuracy_train, xlabel="iteration", ylabel="accuracy", label=["test", "train"])
    return model
end

function predict(text::String, model, alphabet::Array{Char}, labelMap::Dict{Int,Char})::String
    Xs = vectorize(text, alphabet)
    zs = map(X -> Flux.onecold(model(X)), Xs)
    cs = map(z -> join(map(i -> labelMap[i], z)), zs)
    @info cs
    texte = join(cs)
    is = findall(c -> c == options[:unkChar], texte)
    texte[is] .= text[is]
    return texte
end

end # module

