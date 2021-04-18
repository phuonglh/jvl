# Intent Detection module in Julia
# We use the dataset at https://github.com/xliuhw/NLU-Evaluation-Data
# April, 2021 for a demonstration purpose.
# phuonglh@gmail.com

#module Intents

using CSV
using DataFrames
using Flux
using Flux: @epochs
using BSON: @save, @load
using FLoops
using Random
using Plots

include("Embedding.jl")
include("GRU3.jl")
include("Utils.jl")

options = Dict{Symbol,Any}(
    :minFreq => 1,
    :vocabSize => 2^16,
    :wordSize => 20,
    :hiddenSize => 64,
    :maxSequenceLength => 10,
    :batchSize => 64,
    :numEpochs => 50,
    # :corpusPath => string(pwd(), "/dat/nlu/xliuhw/AnnotatedData/NLU-Data-Home-Domain-Annotated-All.csv"),
    :corpusPath => string(pwd(), "/dat/nlu/xliuhw/AnnotatedData/NLU-Data-Home-Domain-Annotated-All.csv.sample"),
    :modelPath => string(pwd(), "/dat/nlu/model.bson"),
    :wordPath => string(pwd(), "/dat/nlu/word.txt"),
    :labelPath => string(pwd(), "/dat/nlu/label.txt"),
    :numCores => 4,
    :verbose => false,
    :unknown => "[UNK]",
    :paddingX => "[PAD_X]",
    :delimiters => r"[-@…–~`'“”’‘|\/$.,:;!?'\u0022\s_]",
    :split => [0.8, 0.2]
)

"""
    readCorpus(path, sampling=true)

    Reads an intent corpus given in a path and return a dataframe of two columns (`intent` and `text`).
"""
function readCorpus(path::String, sampling::Bool=true)::DataFrame
    df = DataFrame(CSV.File(path))
    ef = if !sampling select(df, :intent => :intent, :answer => :text) else df end
    dropmissing!(ef)
end

"""
    vocab(df, options)

    Builds a vocabulary of words. The word vocabulary is sorted by frequency.
    Only words whose count is greater than `minFreq` are kept. The corpus is in `df` with two columns: `intent` and `text`.
"""    
function vocab(df::DataFrame, options)::Array{String}
    sentences = df[:, :text]
    tokens = Iterators.flatten(map(sentence -> string.(split(sentence, options[:delimiters])), sentences))
    wordFrequency = Dict{String, Int}()
    for token in tokens
        word = lowercase(strip(token))
        haskey(wordFrequency, word) ? wordFrequency[word] += 1 : wordFrequency[word] = 1
    end
    filter!(p -> p.second >= options[:minFreq], wordFrequency)
    return collect(keys(wordFrequency))
end

"""
    batch(df, wordIndex, labelIndex, options)

    Create batches of data for training or evaluating. Each batch contains a pair (Xb, Yb) where 
    Xb is an array of matrices of size (d x maxSequenceLength). Each column of Xb is a vector representing a token.
    If a sentence is shorter than maxSequenceLength, it is padded. To speed up the computation, Xb and Yb 
    are stacked as 3-d matrices where the 3-rd dimention is the batch one.
"""
function batch(df::DataFrame, wordIndex::Dict{String,Int}, labelIndex::Dict{String,Int}, options)
    X, Y = Array{Array{Int,1},1}(), Array{Array{Int,1},1}()
    paddingX = wordIndex[options[:paddingX]]
    sentences = map(sentence -> string.(split(sentence, options[:delimiters])), df[:, :text])
    labels = df[:, :intent]
    for i = 1:length(sentences)
        sentence = sentences[i]
        xs = map(token -> get(wordIndex, lowercase(token), 1), sentence)
        ys = Flux.onehot(get(labelIndex, labels[i], 1), 1:length(labelIndex), 1)
        # truncate or pad the sample to have maxSequenceLength
        if length(xs) > options[:maxSequenceLength]
            xs = xs[1:options[:maxSequenceLength]]
        end
        for t=length(xs)+1:options[:maxSequenceLength]
            push!(xs, paddingX)
        end
        push!(X, xs)
        push!(Y, ys)
    end
    # build batches of data for training
    Xb = Iterators.partition(X, options[:batchSize])
    Yb = Iterators.partition(Y, options[:batchSize])
    # stack each input batch as a 3-d matrix
    Xs = map(b -> Int.(Flux.batch(b)), Xb)
    # stack each output batch as a 2-d matrix
    Ys = map(b -> Int.(Flux.batch(b)), Yb)
    (Xs, Ys)
end

function train(options)
    df = readCorpus(options[:corpusPath])
    # random split df for training/test data
    n = nrow(df)
    xs = shuffle(1:n)
    j = Int(round(n*options[:split][2]))
    dfV = df[1:j, :]    # test part
    dfU = df[j+1:n, :]  # training part
    labels = unique(dfU[:, :intent])
    labelIndex = Dict{String,Int}(x => i for (i, x) in enumerate(labels))
    vocabulary = vocab(dfU, options)
    prepend!(vocabulary, [options[:unknown]])
    append!(vocabulary, [options[:paddingX]])
    wordIndex = Dict{String,Int}(x => i for (i, x) in enumerate(vocabulary))
    saveIndex(labelIndex, options[:labelPath])
    saveIndex(wordIndex, options[:wordPath])
    @info "#(vocab)  = $(length(vocabulary))"
    @info "#(labels) = $(length(labels))"

    # define a model for sentence encoding
    encoder = Chain(
        Embedding(min(length(wordIndex), options[:vocabSize]), options[:wordSize]),
        GRU3(options[:wordSize], options[:hiddenSize]),
        Dense(options[:hiddenSize], length(labels))
    )
    @info "Total weight of initial word embeddings = $(sum(encoder[1].W))"

    # the loss function on a batch
    function loss(Xb, Yb)
        Ŷb = encoder(Xb)
        Flux.reset!(encoder)
        return Flux.logitcrossentropy(Ŷb, Yb)
    end

    Xs, Ys = batch(dfU, wordIndex, labelIndex, options)    
    trainingData = collect(zip(Xs, Ys))
    Xv, Yv = batch(dfV, wordIndex, labelIndex, options)    
    testData = collect(zip(Xv, Yv))

    @info string("Number of batches = ", length(trainingData))
    optimizer = ADAM()
    accuracy = Array{Tuple{Float64,Float64},1}()
    evalcb = function()
        ℓ = sum(loss(trainingData[i]...) for i=1:length(trainingData))
        a = evaluate(encoder, Xs, Ys, options)
        b = evaluate(encoder, Xv, Yv, options)
        @info string("loss = ", ℓ, ", training accuracy = ", a, ", test accuracy = ", b)
        push!(accuracy, (a, b))
    end
    @epochs options[:numEpochs] Flux.train!(loss, params(encoder), trainingData, optimizer, cb = Flux.throttle(evalcb, 60))
    # save the model to a BSON file
    @save options[:modelPath] encoder

    @info "Total weight of final word embeddings = $(sum(encoder[1].W))"
    @info "Evaluating the model on the training set..."
    a = evaluate(encoder, Xs, Ys, options)
    b = evaluate(encoder, Xv, Yv, options)
    @info "Training accuracy = $a, test accuracy = $b"
    push!(accuracy, (a, b))
    # plot the accuracy scores
    # @info "Plotting score figure..."
    # as, bs = map(p -> p[1], accuracy), map(p -> p[2], accuracy)
    # plot(1:length(accuracy), [as, bs], xlabel="iterations", ylabel="accuracy", label=["train." "test"], legend=:bottomright, lw=2)
    return encoder, accuracy
end

function evaluate(encoder, Xs, Ys, options)
    numBatches = length(Xs)
    @floop ThreadedEx(basesize=numBatches÷options[:numCores]) for i=1:numBatches
        Ŷb = Flux.onecold(encoder(Xs[i]))
        Yb = Flux.onecold(Ys[i])
        matches = sum(Ŷb .== Yb)
        @reduce(numMatches += matches, numSents += length(Yb))
    end
    @info "Total matches = $(numMatches)/$(numSents)"
    return 100 * (numMatches/numSents)
end


"""
    sampling(df)

    Takes a random subset of a given number of samples and save to an output file.
"""
function sampling(df, numSamples::Int=10000)
    n = nrow(df)
    x = shuffle(1:n)
    sample = df[x[1:numSamples], :]
    CSV.write(string(options[:corpusPath], ".sample"), sample)
    return sample
end

#end # module

