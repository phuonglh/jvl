# phuonglh@gmail.com
# December 2020
# Implementation of PoS tagger using the sequence-to-sequence model in Julia.

using Flux

using Flux: @epochs
using BSON: @save, @load
using FLoops
using BangBang
using MicroCollections
using StatsBase

include("Corpus.jl")
using .Corpus

include("../tok/VietnameseTokenizer.jl")
using .VietnameseTokenizer

include("Embedding.jl")
include("Options.jl")
include("Utils.jl")


"""
    vocab(sentences, minFreq)

    Builds vocabularies of words, shapes, parts-of-speech, and labels. The word vocabulary is sorted by frequency.
    Only words whose count is greater than `minFreq` are kept.
"""    
function vocab(sentences::Array{Sentence}, minFreq::Int = 1)::Vocabularies
    tokens = Iterators.flatten(map(sentence -> sentence.tokens, sentences))
    wordFrequency = Dict{String, Int}()
    shapes = Dict{String,Int}()
    partsOfSpeech = Dict{String,Int}()
    labels = Dict{String,Int}()
    for token in tokens
        word = lowercase(strip(token.word))
        haskey(wordFrequency, word) ? wordFrequency[word] += 1 : wordFrequency[word] = 1
        shapes[shape(token.word)] = 0
        partsOfSpeech[token.annotation[:upos]] = 0
        labels[token.annotation[:pos]] = 0
    end
    # filter out infrequent words
    filter!(p -> p.second >= minFreq, wordFrequency)
    
    Vocabularies(collect(keys(wordFrequency)), collect(keys(shapes)), collect(keys(partsOfSpeech)), collect(keys(labels)))
end

"""
    batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex, options)

    Create batches of data for training or evaluating. Each batch contains a triple (Xb, Yb0, Yb1) where 
     - Xb contains matrices of size (3 x maxSequenceLength); each column is a vector representing a token.
     - Yb0 contains matrices of size (numLabels x maxSequenceLength); each column is an one-hot vector reprsenting [BOS, ys]
     - Yb1 contains matrices of size (numLabels x maxSequenceLength); each column is an one-hot vector reprsenting [ys, EOS]
    If a sentence is shorter than maxSequenceLength, it is padded with vectors of ones.
"""
function batch(sentences::Array{Sentence}, wordIndex::Dict{String,Int}, shapeIndex::Dict{String,Int}, posIndex::Dict{String,Int}, labelIndex::Dict{String,Int}, options)
    X, Y0, Y1 = Array{Array{Int,2},1}(), Array{Array{Int,2},1}(), Array{Array{Int,2},1}()
    paddingX = [wordIndex[options[:paddingX]]; shapeIndex["other"]; posIndex["X"]]
    numLabels = length(labelIndex)
    paddingY = Flux.onehot(labelIndex[options[:paddingY]], 1:numLabels)
    for sentence in sentences
        tokens = sentence.tokens[2:end] # not consider the ROOT token of UD graphs
        xs = map(token -> [ get(wordIndex, lowercase(token.word), wordIndex[options[:unknown]]), 
            get(shapeIndex, shape(token.word), shapeIndex["other"]), 
            get(posIndex, token.annotation[:upos], posIndex["X"])], tokens)
        push!(xs, paddingX)
        ys = map(token -> Flux.onehot(labelIndex[token.annotation[:pos]], 1:numLabels, 1), tokens)
        ys0 = copy(ys); prepend!(ys0, [Flux.onehot(labelIndex["BOS"], 1:length(labelIndex), 1)])
        ys1 = copy(ys); append!(ys1, [Flux.onehot(labelIndex["EOS"], 1:length(labelIndex), 1)])
        # crop the columns of xs and ys to maxSequenceLength
        if length(xs) > options[:maxSequenceLength]
            xs = xs[1:options[:maxSequenceLength]]
            ys0 = ys0[1:options[:maxSequenceLength]]
            ys1 = ys1[1:options[:maxSequenceLength]]
        end
        # pad the sequences to the same maximal length if necessary 
        for t=length(xs)+1:options[:maxSequenceLength]
            push!(xs, paddingX) 
            push!(ys0, paddingY)
            push!(ys1, paddingY)
        end
        push!(X, Flux.batch(xs))
        push!(Y0, Flux.batch(ys0))
        push!(Y1, Flux.batch(ys1))
    end
    # build batches of data for training
    Xb = collect(map(A -> collect(A), Iterators.partition(X, options[:batchSize])))
    Yb0 = collect(map(A -> collect(A), Iterators.partition(Y0, options[:batchSize])))
    Yb1 = collect(map(A -> collect(A), Iterators.partition(Y1, options[:batchSize])))
    (Xb, Yb0, Yb1)
end

"""
    encode(X, embedding, encoder)

    Encodes an index matrix of size (3 x maxSequenceLength) using an embedding layer and an encoder.
    The endoder layer is usually a RNN layer (GRU or LSTM).
"""
encode(X::Array{Int,2}, embedding, encoder) = encoder(embedding(X))

"""
    β(s, H, attention)

    Align a decoder state `s` with hidden states of inputs `H` of size (hiddenSize x m). 
    The decoder state is a column vector `s` of length hiddenSize, it should be repeated to create 
    the same number of columns as `H`, that is of size (hiddenSize x m). 

    This function computes attention scores matrix of size (1 x maxSequenceLength) for a decoder position.
"""
function β(s, H::Array{Float32,2}, attention)
    S = s .* Float32.(ones(1, size(H,2)))
    return attention(tanh.(H + S))
end

"""
    α(β)

    Compute the probabilities (weights) vector by using the softmax function. Return a matrix of the same 
    size as β, that is (1 x maxSequenceLength).
"""
function α(β::Array{Float32,2})
    score = exp.(β)
    s = sum(score, dims=2)
    return score ./ s
end

"""
    decode(H, y0, decoder, attention)

    Decode at a position. `H` contains the hidden states of the encoder, which is a matrix of size (hiddenSize x maxSequenceLength)
    and y0 is an one-hot vector representing a label at position t. `attention` is an attention model.
"""
function decode(H::Array{Float32,2}, y0::Array{Int,1}, decoder, attention)
    w = α(β(decoder.state, H, attention)) # a matrix of size (1 x m)
    c = sum(w .* H, dims=2) # a matrix of size (hiddenSize x 1)
    v = vcat(y0, c)
    return decoder(v)
end

"""
    decode(H, Y0, decoder, attention)

    Decodes a sequence given all components.
"""
function decode(H::Array{Float32,2}, Y0::Array{Int,2}, decoder, attention)
    # find the last non-padded element at position n
    z0 = Flux.onecold(Y0)
    maxLen = size(Y0,2)
    n = maxLen
    while z0[n] == 1 
        n = n-1; 
    end
    # decode positions up to the real sequence length
    y0s = [Y0[:, t] for t=1:n]
    ŷs = [decode(H[:,1:n], y0, decoder, attention) for y0 in y0s]
    # stack the output array into a 2-d matrix of size (hiddenSize x maxSequenceLength)
    return hcat(ŷs...)
end

function model(Xb, Y0b, embedding, encoder, decoder, attention, linear)
    f(X, Y0) = begin
        H = encode(X, embedding, encoder)
        Flux.reset!(encoder)
        # take the last state of the encoder as the initial state of the decoder
        decoder.init = encoder.state[:,end]
        Ŷs = decode(H, Y0, decoder, attention)
        Flux.reset!(decoder)
        linear(Ŷs)
    end
    return map((X, Y0) -> f(X, Y0), Xb, Y0b)
end

"""
    train(options, lr)

    Train the pipeline using a learning rate.
"""
function train(options::Dict{Symbol,Any}, lr=1E-4)
    (sentences, sentencesValidation, sentencesTest) = if (options[:columnFormat])
        (readCorpusUD(options[:trainCorpus]), readCorpusUD(options[:validCorpus]), readCorpusUD(options[:testCorpus]))
    else
        ss = readCorpusVLSP(options[:trainCorpus])
        ss = shuffleSentences(ss)
        splitSentences(ss, [0.8, 0.1, 0.1])
    end
    @info "Number of training sentences = $(length(sentences))"
    @info "Number of validation sentences = $(length(sentencesValidation))"
    @info "Number of test sentences = $(length(sentencesTest))"

    vocabularies = vocab(sentences, options[:minFreq])

    # the PAD_X word will have index 1
    prepend!(vocabularies.words, [options[:paddingX]])
    # the UNK word will have the last index
    append!(vocabularies.words, [options[:unknown]])
    wordIndex = Dict{String,Int}(word => i for (i, word) in enumerate(vocabularies.words))
    shapeIndex = Dict{String,Int}(shape => i for (i, shape) in enumerate(vocabularies.shapes))
    posIndex = Dict{String,Int}(pos => i for (i, pos) in enumerate(vocabularies.partsOfSpeech))
    prepend!(vocabularies.labels, [options[:paddingY]])
    append!(vocabularies.labels, ["BOS", "EOS"])
    labelIndex = Dict{String,Int}(label => i for (i, label) in enumerate(vocabularies.labels))

    # save the vocabulary, shape, part-of-speech and label information to external files
    saveIndex(wordIndex, options[:wordPath])
    saveIndex(shapeIndex, options[:shapePath])
    saveIndex(posIndex, options[:posPath])
    saveIndex(labelIndex, options[:labelPath])

    # create batches of data
    Xbs, Y0bs, Ybs = batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex, options)
    dataset = collect(zip(Xbs, Y0bs, Ybs))

    @info "vocabSize = ", length(wordIndex)
    @info "shapeSize = ", length(shapeIndex)
    @info "posSize = ", length(posIndex)
    @info "numLabels = ", length(labelIndex)
    @info "numBatches  = ", length(dataset)

    # 0. Create an embedding layer
    embedding = EmbeddingWSP(
        min(length(wordIndex), options[:vocabSize]), options[:wordSize], 
        length(shapeIndex), options[:shapeSize], 
        length(posIndex), options[:posSize]
    )
    @info "Total weight of inial word embeddings = $(sum(embedding.word.W))"

    inputSize = options[:wordSize] + options[:shapeSize] + options[:posSize]

    # 1. Create an encoder
    encoder = GRU(inputSize, options[:hiddenSize])

    # 2. Create an attention model which scores the degree of match between 
    #  an output position and an input position. 
    attention = Dense(options[:hiddenSize], 1)

    # 3. Create a decoder
    numLabels = length(labelIndex)
    decoder = GRU(options[:hiddenSize] + numLabels, options[:hiddenSize])
    linear = Dense(options[:hiddenSize], numLabels)
    # The full machinary (WITHOUT embedding for quick testing)
    machine = (encoder, decoder, attention, linear)

    # We need to explicitly program a loss function which does not take into account of padding labels.
    function loss(Xb, Y0b, Yb)
        Ŷb = model(Xb, Y0b, embedding, encoder, decoder, attention, linear)
        Zb = Flux.onecold.(Yb)
        J = 0
        for t=1:length(Yb)
            n = options[:maxSequenceLength]
            # find the last position of non-padded element (1)
            while Zb[t][n] == 1
                n = n - 1
            end
            J += Flux.logitcrossentropy(Ŷb[t][1:n], Yb[t][1:n])
        end
        return J
    end

    Ubs, Vbs, Wbs = batch(sentencesValidation, wordIndex, shapeIndex, posIndex, labelIndex, options)
    datasetValidation = collect(zip(Ubs, Vbs, Wbs))

    optimizer = ADAM(lr)
    file = open(options[:logPath], "w")
    write(file, "loss,trainingAccuracy,validationAccuracy\n")
    evalcb = function()
        ℓ = sum(loss(datasetValidation[i]...) for i=1:length(datasetValidation))
        trainingAccuracy = evaluate(embedding, encoder, decoder, attention, linear, Xbs, Y0bs, Ybs, options)
        validationAccuracy = evaluate(embedding, encoder, decoder, attention, linear, Ubs, Vbs, Wbs, options)
        @info string("\tloss = ", ℓ, ", training accuracy = ", trainingAccuracy, ", validation accuracy = ", validationAccuracy)
        write(file, string(ℓ, ',', trainingAccuracy, ',', validationAccuracy, "\n"))
    end
    # train the model until the validation accuracy decreases 2 consecutive times
    t = 1
    k = 0
    bestDevAccuracy = 0
    @time while (t <= options[:numEpochs]) 
        @info "Epoch $t, k = $k"
        Flux.train!(loss, params(machine), dataset, optimizer, cb = Flux.throttle(evalcb, 60))
        devAccuracy = evaluate(embedding, encoder, decoder, attention, linear, Ubs, Vbs, Wbs, options)
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
        @info "\tbestDevAccuracy = $bestDevAccuracy"
        t = t + 1
    end
    close(file)
    
    # save the model to a BSON file
    @save options[:modelPath] machine

    @info "Total weight of final word embeddings = $(sum(embedding.word.W))"
    @info "Evaluating the model on the training set..."
    accuracy = evaluate(embedding, encoder, decoder, attention, linear, Xbs, Y0bs, Ybs, options)
    @info "Training accuracy = $accuracy"
    accuracyValidation = evaluate(embedding, encoder, decoder, attention, linear, Ubs, Vbs, Wbs, options)
    @info "Validation accuracy = $accuracyValidation"
    return machine
end

"""
    evaluate(embedding, encoder, decoder, attention, linear, Xbs, Y0bs, Ybs, options)

    Evaluate the accuracy of the model on a dataset. 
"""
function evaluate(embedding, encoder, decoder, attention, linear, Xbs, Y0bs, Ybs, options)
    numBatches = length(Xbs)
    @floop ThreadedEx(basesize=numBatches÷options[:numCores]) for i=1:numBatches
        Ŷb = Flux.onecold.(model(Xbs[i], Y0bs[i], embedding, encoder, decoder, attention, linear))
        Yb = Flux.onecold.(Ybs[i])
        # number of tokens and number of matches in this batch
        tokens, matches = 0, 0
        for t=1:length(Yb)
            n = options[:maxSequenceLength]
            # find the last position of non-padded element (1)
            while Yb[t][n] == 1
                n = n - 1
            end
            tokens += n
            matches += sum(Ŷb[t][1:n] .== Yb[t][1:n])
        end
        @reduce(numTokens += tokens, numMatches += matches)
    end
    @info "\tmatched tokens = $(numMatches)/$(numTokens)"
    return 100 * numMatches/numTokens
end

"""
    predict(sentence, embedding, encoder, decoder, attention, linear, wordIndex, shapeIndex, posIndex, labelIndex)

    Find the label sequence for a given sentence.
"""
function predict(sentence, embedding, encoder, decoder, attention, linear, wordIndex, shapeIndex, posIndex, labelIndex, options)
    numLabels = length(labelIndex)
    ps = [labelIndex["BOS"]]
    Xs, Y0s, Ys = batch([sentence], wordIndex, shapeIndex, posIndex, labelIndex, options)
    Xb = first(Xs)
    Y0 = Int.(zeros(numLabels, size(Xb[1],2)))
    m = min(length(sentence.tokens), size(Xb[1],2))
    for t=1:m
        currentY = Flux.onehot(ps[end], 1:numLabels)
        Y0[:,t] = currentY
        Y0b = [ Int.(Y0) ]
        score = model(Xb, Y0b, embedding, encoder, decoder, attention, linear)
        Ŷ = softmax(score[1][:,t])
        # nextY = Flux.onecold(Ŷ)     # use a hard selection approach, always choose the label with the best probability
        nextY = wsample(1:numLabels, Ŷ) # use a soft selection approach to sample a label from the distribution
        push!(ps, nextY)
    end
    return ps[2:end]
end

function diagnose(sentence, embedding, encoder, decoder, attention, linear, wordIndex, shapeIndex, posIndex, labelIndex, options)
    Xs, Y0s, Ys = batch([sentence], wordIndex, shapeIndex, posIndex, labelIndex, options)
    Xb = first(Xs)
    Y0b = first(Y0s)
    score = model(Xb, Y0b, embedding, encoder, decoder, attention, linear)
    return score
end

function loadIndices(options)
    wordIndex = loadIndex(options[:wordPath])
    shapeIndex = loadIndex(options[:shapePath])
    posIndex = loadIndex(options[:posPath])
    labelIndex = loadIndex(options[:labelPath])
    return (wordIndex, shapeIndex, posIndex, labelIndex)
end

# options = optionsVUD
# sentences = readCorpusUD(options[:trainCorpus]);
# machine = train(options)
# embedding, encoder, decoder, attention, linear = machine
# OR: encoder, decoder, attention, linear = machine 
# wordIndex, shapeIndex, posIndex, labelIndex = loadIndices(options)
# predict(sentences[1], embedding, encoder, decoder, attention, linear, wordIndex, shapeIndex, posIndex, labelIndex, options)
# diagnose(tokens, embedding, encoder, decoder, attention, linear, wordIndex, shapeIndex, posIndex, labelIndex, options)
