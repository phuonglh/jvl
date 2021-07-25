# Vietnamese diacritics generation
# (C) phuonglh@gmail.com, 2021

module VDG

using Flux
using Flux: @epochs
using BSON: @save, @load

using FLoops
using Random

# using CSV
# using DataFrames
# using Plots

include("DiacriticsRemoval.jl")
include("Utils.jl")
include("BiRNN.jl")

options = Dict{Symbol,Any}(
    :sampleSize => 5_000,
    :dataPath => string(pwd(), "/dat/vdg/010K.txt"),
    :alphabetPath => string(pwd(), "/dat/vdg/alphabet.txt"),
    :modelPath => string(pwd(), "/dat/vdg/vdg.bson"),
    :maxSequenceLength => 64,
    :batchSize => 32,
    :numEpochs => 20,
    :unkChar => 'S',
    :padChar => 'P',
    :gpu => false,
    :hiddenSize => 64,
    :split => [0.8, 0.2],
    :η => 1E-3 # learning rate for Adam optimizer
)

"""
  vectorize(text, alphabet)

  Transforms a sentence into an array of vectors based on an alphabet for training. 
  In training mode, this function 
  returns a pair of matrix `(x, y)` where `x` is a matrix of size `|alphabet| x maxSequenceLength` 
  and `y` is also a matrix of size `|alphabet| x maxSequenceLength`. 
  In test mode, this function returns only `x` matrix.
"""
function vectorize(text::String, alphabet::Array{Char}, training::Bool=true)
    # truncate or pad input sequences to have the same maxSequenceLength
    n = options[:maxSequenceLength]
    x = removeDiacritics(text)
    # slice x into subarrays of equal length
    xs = collect(Iterators.partition(x, n))
    # pad the last subarray with the pad character
    ps = fill(options[:padChar], n - length(xs[end]))
    xs[end] = vcat(xs[end], ps)
    # now all the subarrays in xs are of the same length, 
    # convert them into one-hot matrices of size (|alphabet| x maxSeqLen)
    Xs = map(x -> Float32.(Flux.onehotbatch(x, alphabet)), xs)
    if (training) 
        # texte = map(c -> c ∈ keys(charMap) ? c : options[:unkChar], text)
        ys = collect(Iterators.partition(text, n))
        ys[end] = vcat(ys[end], ps)
        Ys = map(y -> Float32.(Flux.onehotbatch(y, alphabet)), ys)
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
    padding = 1
    total = 0
    correct = 0
    for i = 1:batchSize
        t = options[:maxSequenceLength]
        while t > 0 && bs[i][t] == padding
            t = t - 1
        end
        total = total + t
        correct = correct + sum(as[i][1:t] .== bs[i][1:t])
    end
    return (total, correct)
end

function train(options)
    # read input data frame and create inp/out sequences
    # df = CSV.File(options[:dataPath]) |> DataFrame
    # ys = map(y -> lowercase(y), df[:, :question])
    lines = readlines(options[:dataPath])
    N = min(options[:sampleSize], length(lines))
    @info "N = $(N)"
    ys = map(y -> lowercase(y), lines[1:N])
    # create and save alphabet index    
    alphabet = unique(join(ys))
    sort!(alphabet)
    prepend!(alphabet, options[:padChar])
    @info "alphabet = $(join(alphabet))"
    charIndex = Dict{Char,Int}(c => i for (i, c) in enumerate(alphabet))
    saveIndex(charIndex, options[:alphabetPath])

    # create training sequences
    XYs = map(y -> vectorize(y, alphabet), ys)
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
        Dense(options[:hiddenSize], length(alphabet))
    )
    @info model
    # compute the loss of the model on a batch
    function loss(Xb, Yb)
        function g(X, Y)
            ys = Flux.onecold(Y)
            t = options[:maxSequenceLength]
            while (ys[t] == 1) t = t - 1; end
            Z = model(X)
            return Flux.logitcrossentropy(Z[:,1:t], Y[:,1:t])
        end
        Flux.reset!(model)
        return sum(g(Xb[i], Yb[i]) for i=1:length(Xb))
    end
    optimizer = ADAM(options[:η])
    
    # accuracy_test, accuracy_train = Array{Float64,1}(), Array{Float64,1}()
    Js = Array{Float64,1}()
    function evalcb() 
        J = sum(loss(dataset_train[i]...) for i=1:length(dataset_train))
        push!(Js, J)
        @info "J(θ) = $J"
        # pairs_test = [evaluate(model, dataset_test[i]...) for i=1:length(dataset_test)]
        # u, v = reduce(((a1, b1), (a2, b2)) -> (a1 + a2, b1 + b2), pairs_test)
        # push!(accuracy_test, v/u)
        # pairs_train = [evaluate(model, dataset_train[i]...) for i=1:length(dataset_train)]
        # u, v = reduce(((a1, b1), (a2, b2)) -> (a1 + a2, b1 + b2), pairs_train)
        # push!(accuracy_train, v/u)
        # @info "loss = $J, test accuracy = $(accuracy_test[end]), training accuracy = $(accuracy_train[end])"
    end
    for _=1:options[:numEpochs]
        @time Flux.train!(loss, params(model), dataset_train, optimizer, cb = Flux.throttle(evalcb, 60))
    end
    if (options[:gpu])
        model = model |> cpu
    end
    @save options[:modelPath] model
    # report test accuracy and training accuracy
    pairs_test = [evaluate(model, dataset_test[i]...) for i=1:length(dataset_test)]
    u, v = reduce(((a1, b1), (a2, b2)) -> (a1 + a2, b1 + b2), pairs_test)
    push!(accuracy_test, v/u)
    pairs_train = [evaluate(model, dataset_train[i]...) for i=1:length(dataset_train)]
    u, v = reduce(((a1, b1), (a2, b2)) -> (a1 + a2, b1 + b2), pairs_train)
    push!(accuracy_train, v/u)
    @info "test accuracy = $(accuracy_test[end]), training accuracy = $(accuracy_train[end])"
    # plot(1:length(Js), xlabel="iteration", ylabel="loss", label=["J"])
    return model, Js
end

function predict(text::String, model, alphabet::Array{Char}, labelMap::Dict{Int,Char})::String
    Xs = vectorize(text, alphabet, false)
    zs = map(X -> Flux.onecold(model(X)), Xs)
    cs = map(z -> join(map(i -> labelMap[i], z)), zs)
    texte = collect(join(cs))[1:length(text)]
    is = findall(c -> c == options[:unkChar], texte)
    texte[is] .= collect(text)[is]
    return join(texte)
end

function test(text, model)
    alphabetIndex = loadAlphabet(options[:alphabetPath])
    alphabetMap = Dict{Int,Char}(i => c for (c,i) in alphabetIndex)
    test(text, model, alphabet, alphabetMap)
end

function test(text, model, alphabet, alphabetMap)
    Xs, Ys = vectorize(text, alphabet)
    xs = Flux.onecold(Xs[1])
    @info alphabet[xs]
    ys = Flux.onecold(Ys[1])
    vs = join(map(y -> alphabetMap[y], ys))
    @info vs
    zs = Flux.onecold(model(Xs[1]))
    ws = join(map(y -> alphabetMap[y], zs))
    @info ws
end

end # module

