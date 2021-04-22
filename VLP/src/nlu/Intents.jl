# Intent Detection module in Julia
# We use the dataset at https://github.com/xliuhw/NLU-Evaluation-Data
# April, 2021 for a demonstration purpose.
# phuonglh@gmail.com

module Intents

using CSV
using DataFrames
using Flux
using Flux: @epochs
using BSON: @save, @load
using FLoops
using Random
using Plots
using CUDA

include("Embedding.jl")
include("GRU3.jl")
include("Utils.jl")

include("../tok/VietnameseTokenizer.jl")
using .VietnameseTokenizer


options = Dict{Symbol,Any}(
    :minFreq => 1,
    :vocabSize => 2^16,
    :wordSize => 20,
    :hiddenSize => 128,
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
    :split => [0.8, 0.2],
    :gpu => false
)

"""
    tokenize(utterance)

    Tokenizes an utterance into tokens, replacing common tokens by their shape, including 
    email, number, url, date, time, currency, punct.
"""
function tokenize(utterance::String)::Array{String}
    function transform(token::VietnameseTokenizer.Token)::String
        s = VietnameseTokenizer.shape(token.text)
        if s == "phrase" || s == "allcap" || s == "capital" || s == "name" || s == "UNK"
            return lowercase(token.text)
        else
            return string("{", s, "}")
        end
    end
    tokens = VietnameseTokenizer.tokenize(utterance)
    map(token -> transform(token), tokens)
end

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
    sentences = df[:,:text]
    # tokens = Iterators.flatten(map(sentence -> string.(split(lowercase(sentence), options[:delimiters])), sentences))
    tokens = Iterators.flatten(map(sentence -> tokenize(sentence), sentences))
    wordFrequency = Dict{String, Int}()
    for token in tokens
        word = strip(token)
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
    # sentences = map(sentence -> string.(split(lowercase(sentence), options[:delimiters])), df[:, :text])
    sentences = map(sentence -> tokenize(sentence), df[:,:text])
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
    Random.seed!(220712)
    n = nrow(df)
    xs = shuffle(1:n)
    j = Int(round(n*options[:split][2]))
    dfV = df[1:j,:]    # test part
    dfU = df[j+1:n,:]  # training part
    labels = unique(dfU[:,:intent])
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
    @info string("Number of training batches = ", length(trainingData))
    @info string("Number of test batches = ", length(testData))
    # bring data and model to GPU if set
    if options[:gpu]
        @info "Moving data to GPU..."
        trainingData = map(p -> p |> gpu, trainingData)
        testData = map(p -> p |> gpu, testData)
        @info "Moving model to GPU..."
        encoder = encoder |> gpu
    end
    optimizer = ADAM()
    accuracy = Array{Tuple{Float64,Float64},1}()
    evalcb = function()
        ℓ = sum(loss(trainingData[i]...) for i=1:length(trainingData))
        a = evaluate(encoder, Xs, Ys, options)
        b = evaluate(encoder, Xv, Yv, options)
        @info string("loss = ", ℓ, ", training accuracy = ", a, ", test accuracy = ", b)
        push!(accuracy, (a, b))
    end
    # train the model until the validation accuracy decreases 2 consecutive times
    t, k = 1, 0
    bestDevAccuracy = 0
    @time while (t <= options[:numEpochs]) 
        @info "Epoch $t, k = $k"
        Flux.train!(loss, params(encoder), trainingData, optimizer, cb = Flux.throttle(evalcb, 60))
        devAccuracy = evaluate(encoder, Xv, Yv, options)
        if bestDevAccuracy < devAccuracy
            bestDevAccuracy = devAccuracy
            k = 0
        else
            k = k + 1
            if (k == 3)
                @info "Stop training because current accuracy is smaller than the best accuracy: $(devAccuracy) < $(bestDevAccuracy)."
                break
            end
        end
        @info "bestDevAccuracy = $bestDevAccuracy"
        t = t + 1
    end
    # save the model to a BSON file
    if options[:gpu]
        encoder = encoder |> cpu
    end
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

"""
    evaluate(encoder, Xs, Ys, options)

    Evaluates the accuracy of the classifier w/o using threaded execution.
"""
function evaluate(encoder, Xs, Ys, options)
    numBatches = length(Xs)
    numMatches = 0
    numSents = 0
    for i=1:numBatches
        Ŷb = Flux.onecold(encoder(Xs[i]) |> cpu) # fix an issue of Julia 1.5
        Yb = Flux.onecold(Ys[i] |> cpu) 
        matches = sum(Ŷb .== Yb)
        numMatches += matches
        numSents += length(Yb)
    end
    @info "Total matches = $(numMatches)/$(numSents)"
    return 100 * (numMatches/numSents)
end

# function evaluate(encoder, Xs, Ys, options)
#     numBatches = length(Xs)
#     @floop ThreadedEx(basesize=numBatches÷options[:numCores]) for i=1:numBatches
#         Ŷb = Flux.onecold(encoder(Xs[i]))
#         Yb = Flux.onecold(Ys[i])  
#         matches = sum(Ŷb .== Yb)
#         @reduce(numMatches += matches, numSents += length(Yb))
#     end
#     @info "Total matches = $(numMatches)/$(numSents)"
#     return 100 * (numMatches/numSents)
# end


"""
    predict(encoder, Xs, labelMap)

    Predicts the labels of input utterances and returns batches of predicted intents.
    The `labelMap` is derived from a `labelIndex`, which maps an integer into an intent string. 
"""
function predict(encoder, Xs, labelMap::Dict{Int,String})
    numBatches = length(Xs)
    result = Array{Array{String,1},1}()
    for i=1:numBatches
        Ŷb = Flux.onecold(encoder(Xs[i]))
        Lb = map(ys -> map(y -> labelMap[y], ys), Ŷb) 
        push!(result, Lb)
    end
    return result
end

"""
    predict(encoder, utterances, options)

    Predicts the intents of given utterances using a trained model.
"""
function predict(encoder, utterances::Array{String,1}, options)
    df = DataFrame(:text => utterances)
    predict(encoder, df, options)
end

"""
    predict(encoder, df, options)

    Predicts the intents of utterances given in a data frame (with column :text).
"""
function predict(encoder, df, options)
    wordIndex = loadIndex(options[:wordPath])
    X = Array{Array{Int,1},1}()
    paddingX = wordIndex[options[:paddingX]]
    sentences = map(sentence -> tokenize(sentence), df[:,:text])
    for i = 1:length(sentences)
        sentence = sentences[i]
        xs = map(token -> get(wordIndex, lowercase(token), 1), sentence)
        if length(xs) > options[:maxSequenceLength]
            xs = xs[1:options[:maxSequenceLength]]
        end
        for t=length(xs)+1:options[:maxSequenceLength]
            push!(xs, paddingX)
        end
        push!(X, xs)
    end
    # build batches of data for evaluating
    Xb = Iterators.partition(X, options[:batchSize])
    # stack each input batch as a 3-d matrix
    Xs = map(b -> Int.(Flux.batch(b)), Xb)
    # build a label map from the loaded label index
    labelIndex = loadIndex(options[:labelPath])
    labelMap = Dict{Int,String}(labelIndex[label] => label for label in keys(labelIndex))
    return predict(encoder, Xs, labelMap)
end

"""
    predict(encoder, utterance, wordIndex, labelMap)

    Predicts the intent of an utterance. This function is useful for building web service API.
"""
function predict(encoder, utterance::String, wordIndex::Dict{String,Int}, labelMap::Dict{Int,String})
    sentence = tokenize(utterance)
    xs = map(token -> get(wordIndex, lowercase(token), 1), sentence)
    if length(xs) > options[:maxSequenceLength]
        xs = xs[1:options[:maxSequenceLength]]
    end
    for t=length(xs)+1:options[:maxSequenceLength]
        push!(xs, wordIndex[options[:paddingX]])
    end
    Xs = Int.(Flux.batch([xs]))
    j = Flux.onecold(encoder(Xs))[1]
    return labelMap[j]
end

end # module

