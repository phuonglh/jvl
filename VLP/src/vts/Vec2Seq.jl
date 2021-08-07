# A simple character-based Vec2Seq implementation in Julia
# (C) phuonglh@gmail.com

module Vec2Seq

using Flux
using Flux: @epochs, onehot
using BSON: @save, @load

using FLoops
using Random
using StatsBase

# using CSV
# using DataFrames
# using Plots

include("Utils.jl")
include("BiRNN.jl")

options = Dict{Symbol,Any}(
    :sampleSize => 2_000,
    :dataPath => string(pwd(), "/dat/vdg/010K.txt"),
    :alphabetPath => string(pwd(), "/dat/vts/alphabet.txt"),
    :modelPath => string(pwd(), "/dat/vts/vts.bson"),
    :maxSequenceLength => 64,
    :batchSize => 32,
    :numEpochs => 100,
    :bosChar => 'B',
    :eosChar => 'E',
    :padChar => 'P',
    :hiddenSize => 128,
    :gpu => false,
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
    x = string(options[:bosChar], text)
    # slice x into subarrays of equal length
    xs = collect(Iterators.partition(x, n))
    # pad the last subarray with one EOS character and then pad characters
    ps = fill(options[:padChar], n - length(xs[end]))
    xs[end] = vcat(xs[end], ps)
    # now all the subarrays in xs are of the same length, 
    # convert them into one-hot matrices of size (|alphabet| x maxSeqLen)
    Xs = map(x -> Float32.(Flux.onehotbatch(x, alphabet)), xs)
    if (training) 
        y = string(text, options[:eosChar])
        ys = collect(Iterators.partition(y, n))
        ys[end] = vcat(ys[end], ps)
        Ys = map(y -> Float32.(Flux.onehotbatch(y, alphabet)), ys)
        return (Xs, Ys)
    else
        return Xs
    end
end

function train(options)
    lines = readlines(options[:dataPath])
    N = min(options[:sampleSize], length(lines))
    @info "N = $(N)"
    ys = map(y -> lowercase(y), lines[1:N])
    # create and save alphabet index    
    alphabet = unique(join(ys))
    sort!(alphabet)
    prepend!(alphabet, options[:eosChar])
    prepend!(alphabet, options[:bosChar])
    prepend!(alphabet, options[:padChar]) # should be the first (index=1)
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
    # define a model
    model = Chain(
        GRU(length(alphabet), options[:hiddenSize]),
        Dense(options[:hiddenSize], length(alphabet), relu)
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
    
    Js = Array{Float64,1}()
    function evalcb() 
        J = sum(loss(dataset[i]...) for i=1:length(dataset))
        push!(Js, J)
        @info "J(θ) = $J"
    end
    for _=1:options[:numEpochs]
        @time Flux.train!(loss, params(model), dataset, optimizer, cb = Flux.throttle(evalcb, 60))
    end
    if (options[:gpu])
        model = model |> cpu
    end
    @save options[:modelPath] model
    return model, Js
end

function generate(prefix::String, model, alphabet::Array{Char}, numChars::Int=100, sampling::Bool=false)::String
    text = string(options[:bosChar], prefix)
    n = length(text)
    X = vectorize(text, alphabet, false)[1]
    Y = model(X)
    y = Y[:,n] # the last vector
    prediction = Array{Int,1}()
    Flux.reset!(model)
    for _=1:numChars
        z = softmax(y)
        ŷ = sampling ? wsample(alphabet, z) : alphabet[Flux.onecold(z)]
        ỹ = Flux.onehot(ŷ, alphabet)
        append!(prediction, Flux.onecold(ỹ))
        y = model(ỹ)
    end
    return string(text, join(alphabet[prediction]))
end

end # module