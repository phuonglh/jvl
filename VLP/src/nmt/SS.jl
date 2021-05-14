using Flux
using Flux: @epochs
using BSON: @save, @load
using FLoops
using BangBang
using MicroCollections
using StatsBase


include("../tok/VietnameseTokenizer.jl")
using .VietnameseTokenizer

include("Embedding.jl")
include("Options.jl")
include("Corpus.jl")


struct Machine
    sourceEmbedding
    encoder
    targetEmbedding
    decoder
    linear
end

"""
    vocab(sentences, options)

    Builds vocabularies of words, shapes, parts-of-speech, and labels. The word vocabulary is sorted by frequency.
    Only words whose count is greater than `minFreq` are kept.
"""
function vocab(sentences::Array{String}, options)::Dict{String,Int}
    tokens = Iterators.flatten(map(sentence -> string.(split(sentence, options[:delimiters])), sentences))
    wordFrequency = Dict{String, Int}()
    for token in tokens
        word = lowercase(strip(token))
        if VietnameseTokenizer.shape(word) == "number" word = "[NUM]"; end
        haskey(wordFrequency, word) ? wordFrequency[word] += 1 : wordFrequency[word] = 1
    end
    # filter out infrequent words
    filter!(p -> p.second >= options[:minFreq], wordFrequency)
    words = collect(keys(wordFrequency))
    Dict{String,Int}(word => i for (i, word) in enumerate(words))
end

function batch(pairs::Array{Tuple{String,String}}, sourceDict, targetDict, options)
    @info "Create batches for training. Please wait..."
    Xs, Ys = Array{Array{Int},1}(), Array{Array{Int},1}()
    b = options[:batchSize]
    m = options[:maxSequenceLength]
    transform(word) = if VietnameseTokenizer.shape(word) == "number" word = "[NUM]"; else word end
    for pair in pairs
        srcWords = lowercase.(string.(split(pair[1], options[:delimiters])))
        tarWords = lowercase.(string.(split(pair[2], options[:delimiters])))
        X = map(word -> get(sourceDict, transform(word), sourceDict[options[:UNK]]), srcWords)
        Y = map(word -> get(targetDict, transform(word), targetDict[options[:UNK]]), tarWords)
        if (length(X) < m && length(Y) < m) 
            append!(X, sourceDict[options[:EOS]])
            append!(Y, targetDict[options[:EOS]])
            xs = Flux.rpad(X, m, sourceDict[options[:PAD]])
            ys = Flux.rpad(Y, m, targetDict[options[:PAD]])
            push!(Xs, xs)
            push!(Ys, ys)
        end
    end
    Xbs = collect(map(A -> collect(A), Iterators.partition(Xs, b)))
    Ybs = collect(map(A -> collect(A), Iterators.partition(Ys, b)))
    (Xbs, Ybs)
end

function model(X::Array{Int}, Y::Array{Int}, machine)
    U = machine.sourceEmbedding(X)
    H = machine.encoder(U)
    # compute output embeddings
    V = machine.targetEmbedding(Y)

    s = H[:,end]
    function g(j::Int)
        o = vcat(s, V[:,j])
        s = machine.decoder(o)
    end
    os = [g(j) for j=1:size(H,2)]
    O = Flux.stack(os, 2)
    machine.linear(O)
end

function model(Xb, Yb, machine)
    Flux.reset!(machine.encoder)
    Flux.reset!(machine.decoder)
    map((X, Y) -> model(X, Y, machine), Xb, Yb)
end

function train(options)
    pairs = readCorpusEuroparl(options)
    sourceSentences = map(pair -> pair[1], pairs)
    targetSentences = map(pair -> pair[2], pairs)
    sourceDict = vocab(sourceSentences, options); saveIndex(sourceDict, options[:sourceDictPath])
    targetDict = vocab(targetSentences, options); saveIndex(targetDict, options[:targetDictPath])
    m, n = length(sourceDict), length(targetDict)
    sourceDict[options[:UNK]] = m+1; sourceDict[options[:EOS]] = m+2; sourceDict[options[:PAD]] = m+3
    targetDict[options[:UNK]] = n+1; targetDict[options[:EOS]] = n+2; targetDict[options[:PAD]] = n+3

    Xbs, Ybs = batch(pairs, sourceDict, targetDict, options)
    dataset = collect(zip(Xbs, Ybs))
    @info string("source vocab size = ", m)
    @info string("target vocab size = ", n)
    @info string("number of batches = ", length(Xbs))

    sourceEmbedding = Embedding(m+3, options[:inputSize])
    encoder = GRU(options[:inputSize], options[:hiddenSize])
    targetEmbedding = Embedding(n+3, options[:outputSize])
    decoder = GRU(options[:hiddenSize] + options[:outputSize], options[:hiddenSize])
    linear = Dense(options[:hiddenSize], n+3)

    layers = Chain(sourceEmbedding, encoder, targetEmbedding, decoder, linear)
    machine = Machine(sourceEmbedding, encoder, targetEmbedding, decoder, linear)

    # We need to explicitly program a loss function which does not take into account of padding labels.
    paddingY = targetDict[options[:PAD]]
    maxSeqLen = options[:maxSequenceLength]

    function loss(Xb, Yb)
        Ŷb = model(Xb, Yb, machine)
        Zb = map(Y -> Flux.onehotbatch(Y, 1:n+3), Yb)
        J = 0
        # run through the batch and aggregate loss values
        for t=1:length(Yb)
            k = maxSeqLen
            # find the last position of non-padded element
            while (Yb[t][k] == paddingY) k = k - 1; end
            J += Flux.logitcrossentropy(Ŷb[t][1:k], Zb[t][1:k])
        end
        return J
    end

    optimizer = ADAM(options[:α])
    file = open(options[:logPath], "w")
    write(file, "loss,trainingAccuracy\n")
    evalcb = function()
        ℓ = sum(loss(dataset[i]...) for i=1:min(100,length(dataset)))
        @info string("\tloss = ", ℓ)
        trainingAccuracy = evaluate(machine, Xbs, Ybs, paddingY, maxSeqLen)
        @info string("\tloss = ", ℓ, ", training accuracy = ", trainingAccuracy)
        write(file, string(ℓ, ',', trainingAccuracy, "\n"))
    end
    # train the model until the validation accuracy decreases 2 consecutive times
    epoch = 1
    times = 0
    bestAccuracy = 0
    @time while (epoch <= options[:numEpochs]) 
        @info "Epoch $epoch, times = $times"
        Flux.train!(loss, params(layers), dataset, optimizer, cb = Flux.throttle(evalcb, 60))
        accuracy = evaluate(machine, Xbs, Ybs, paddingY, maxSeqLen)
        if bestAccuracy < accuracy
            bestAccuracy = accuracy
            times = 0
        else
            times = times + 1
            if (times >= 3)
                @info "3-consecutive times worse than the best. Stop training: $(accuracy) < $(bestAccuracy)."
                break
            end
        end
        @info "\tbestAccuracy = $bestAccuracy"
        epoch = epoch + 1
    end
    close(file)
    
    # save the model to a BSON file
    @save options[:modelPath] machine

    @info "Evaluating the model on the training set..."
    accuracy = evaluate(machine, Xbs, Ybs, paddingY, maxSeqLen)
    @info "Training accuracy = $accuracy"
    machine
end

function evaluate(machine, Xbs, Ybs, paddingY, maxSeqLen)
    numBatches = length(Xbs)
    @floop ThreadedEx(basesize=numBatches÷options[:numCores]) for i=1:numBatches
        Ŷb = Flux.onecold.(model(Xbs[i], Ybs[i], machine))
        Yb = Ybs[i]
        # number of tokens and number of matches in this batch
        tokens, matches = 0, 0
        for t=1:length(Yb)
            k = maxSeqLen
            # find the last position of non-padded element
            while (Yb[t][k] == paddingY) k = k - 1; end
            tokens += k
            matches += sum(Ŷb[t][1:k] .== Yb[t][1:k])
        end
        @reduce(numTokens += tokens, numMatches += matches)
    end
    @info "\tmatched tokens = $(numMatches)/$(numTokens)"
    100 * numMatches/numTokens
end

