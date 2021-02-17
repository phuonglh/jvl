# phuonglh@gmail.com
# Sentence encoder which encodes a sequence of tokens into a sequence of 
# dense vectors and perform sequence tagging. This programme performs part-of-speech 
# tagging on a Universal Dependencies treebank data. Here, we use (word, shape, universal PoS) to 
# infer language-specific part-of-speech.

module PoSTagger

export train, evaluate, run, optionsVLSP2010, loadEncoder

using Flux
using Flux: @epochs
using BSON: @save, @load

using FLoops

using ..Corpus

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

    Create batches of data for training or evaluating. Each batch contains a pair (Xb, Yb) where 
    Xb is an array of matrices of size (featuresPerToken x maxSequenceLength). Each column of Xb is a vector representing a token.
    If a sentence is shorter than maxSequenceLength, it is padded with vectors of ones. To speed up the computation, Xb and Yb 
    are stacked as 3-d matrices where the 3-rd dimention is the batch one.
"""
function batch(sentences::Array{Sentence}, wordIndex::Dict{String,Int}, shapeIndex::Dict{String,Int}, posIndex::Dict{String,Int}, labelIndex::Dict{String,Int}, options=optionsVUD)
    X, Y = Array{Array{Int,2},1}(), Array{Array{Int,2},1}()
    paddingX = [wordIndex[options[:paddingX]]; 1; 1]
    paddingY = Flux.onehot(labelIndex[options[:paddingY]], 1:length(labelIndex))
    for sentence in sentences
        xs = map(token -> [get(wordIndex, lowercase(token.word), 1), get(shapeIndex, shape(token.word), 1), get(posIndex, token.annotation[:upos], 1)], sentence.tokens)
        ys = map(token -> Flux.onehot(get(labelIndex, token.annotation[:pos], 1), 1:length(labelIndex), 1), sentence.tokens)
        # pad the columns of xs and ys to maxSequenceLength
        if length(xs) > options[:maxSequenceLength]
            xs = xs[1:options[:maxSequenceLength]]
            ys = ys[1:options[:maxSequenceLength]]
        end
        for t=length(xs)+1:options[:maxSequenceLength]
            push!(xs, paddingX) 
            push!(ys, paddingY)
        end
        push!(X, Flux.batch(xs))
        push!(Y, Flux.batch(ys))
    end
    # build batches of data for training
    Xb = Iterators.partition(X, options[:batchSize])
    Yb = Iterators.partition(Y, options[:batchSize])
    # stack each input batch as a 3-d matrix
    Xs = map(b -> Int.(Flux.batch(b)), Xb)
    # stack each output batch as a 3-d matrix
    Ys = map(b -> Int.(Flux.batch(b)), Yb)
    (Xs, Ys)
end

"""
    train(options)

    Train an encoder.
"""
function train(options::Dict{Symbol,Any})
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
    
    prepend!(vocabularies.words, [options[:unknown]])
    append!(vocabularies.words, [options[:paddingX]])
    wordIndex = Dict{String,Int}(word => i for (i, word) in enumerate(vocabularies.words))
    shapeIndex = Dict{String,Int}(shape => i for (i, shape) in enumerate(vocabularies.shapes))
    posIndex = Dict{String,Int}(pos => i for (i, pos) in enumerate(vocabularies.partsOfSpeech))
    prepend!(vocabularies.labels, [options[:paddingY]])
    labelIndex = Dict{String,Int}(label => i for (i, label) in enumerate(vocabularies.labels))
    
    # create batches of data, each batch is a 3-d matrix of size 3 x maxSequenceLength x batchSize
    Xs, Ys = batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex, options)
    dataset = collect(zip(Xs, Ys))
    @info "vocabSize = ", length(wordIndex)
    @info "shapeSize = ", length(shapeIndex)
    @info "posSize = ", length(posIndex)
    @info "numLabels = ", length(labelIndex)
    @info "numBatches  = ", length(dataset)
    @info size(Xs[1])
    @info size(Ys[1])

    # save the vocabulary, shape, part-of-speech and label information to external files
    saveIndex(wordIndex, options[:wordPath])
    saveIndex(shapeIndex, options[:shapePath])
    saveIndex(posIndex, options[:posPath])
    saveIndex(labelIndex, options[:labelPath])

    # define a model for sentence encoding
    encoder = Chain(
        EmbeddingWSP(min(length(wordIndex), options[:vocabSize]), options[:wordSize], length(shapeIndex), options[:shapeSize], length(posIndex), options[:posSize]),
        GRU(options[:wordSize] + options[:shapeSize] + options[:posSize], options[:hiddenSize]),
        Dense(options[:hiddenSize], length(labelIndex))
    )

    @info "Total weight of initial word embeddings = $(sum(encoder[1].word.W))"

    """
        loss(X, Y)

        Compute the loss on one batch of data where X and Y are 3-d matrices of size (K x maxSequenceLength x batchSize).
        We use the log cross-entropy loss to measure the average distance between prediction and true sequence pairs.
    """
    function loss(X, Y)
        b = size(X,3)
        predictions = [encoder(X[:,:,i]) for i=1:b]
        truths = [Y[:,:,i] for i=1:b]
        value = sum(Flux.logitcrossentropy(predictions[i], truths[i]) for i=1:b)
        Flux.reset!(encoder)
        return value
    end

    Us, Vs = batch(sentencesValidation, wordIndex, shapeIndex, posIndex, labelIndex, options)
    datasetValidation = collect(zip(Us, Vs))

    optimizer = ADAM()
    file = open(options[:logPath], "w")
    write(file, "loss,trainingAccuracy,validationAccuracy\n")
    evalcb = Flux.throttle(30) do
        ℓ = loss(dataset[1]...)
        trainingAccuracy = evaluate(encoder, Xs, Ys, options)
        validationAccuracy = evaluate(encoder, Us, Vs, options)
        @info string("loss = ", ℓ, ", training accuracy = ", trainingAccuracy, ", validation accuracy = ", validationAccuracy)
        write(file, string(ℓ, ',', trainingAccuracy, ',', validationAccuracy, "\n"))
    end
    # train the model
    @time @epochs options[:numEpochs] Flux.train!(loss, params(encoder), dataset, optimizer, cb = evalcb)
    close(file)
    # save the model to a BSON file
    @save options[:modelPath] encoder

    @info "Total weight of final word embeddings = $(sum(encoder[1].word.W))"
    @info "Evaluating the model on the training set..."
    accuracy = evaluate(encoder, Xs, Ys, options)
    @info "Training accuracy = $accuracy"

    accuracy = evaluate(encoder, Us, Vs, options)
    @info "Validation accuracy = $accuracy"

    Us, Vs = batch(sentencesTest, wordIndex, shapeIndex, posIndex, labelIndex, options)
    accuracy = evaluate(encoder, Us, Vs, options)
    @info "Test accuracy = $accuracy"

    encoder
end

"""
    evaluate(encoder, Xs, Ys, options, paddingY)

    Evaluate the accuracy of the encoder on a dataset. `Xs` is a list of 3-d input matrices and `Ys` is a list of 
    3-d ground-truth output matrices. The third dimension is the batch one.
"""
function evaluate(encoder, Xs, Ys, options, paddingY::Int=1)
    numBatches = length(Xs)
    # normally, size(X,3) is the batch size except the last batch
    @floop ThreadedEx(basesize=numBatches÷options[:numCores]) for i=1:numBatches
        b = size(Xs[i],3)
        Flux.reset!(encoder)
        Ŷb = Flux.onecold.(encoder(Xs[i][:,:,t]) for t=1:b)
        Yb = Flux.onecold.(Ys[i][:,:,t] for t=1:b)
        # number of tokens and number of matches in this batch
        tokens, matches = 0, 0
        for t=1:b
            n = options[:maxSequenceLength]
            # find the last position of non-padded element
            while Yb[t][n] == paddingY
                n = n - 1
            end
            tokens += n
            matches += sum(Ŷb[t][1:n] .== Yb[t][1:n])
        end
        @reduce(numTokens += tokens, numMatches += matches)
    end
    @info "Total matched tokens = $(numMatches)/$(numTokens)"
    return numMatches/numTokens
end

"""
    run(encoder, X)

    Tag an input batch and return the output batch. `X` is a 3-d matrix of size `3 x 40 x b` where 
    `b` is a batch size, `40` is the maximum sequence length and `3` is the size of vector representing 
    a token (word index, shape index, UD part-of-speech index).
"""
function run(encoder, X, labelIndex)
    b = size(X,3)
    Flux.reset!(encoder)
    Ŷ = Flux.onecold.(encoder(X[:,:,t]) for t=1:b)
    labels = fill("", length(labelIndex))
    for a in keys(labelIndex)
        labels[labelIndex[a]] = a
    end
    map(ŷ -> labels[ŷ], Ŷ)
end

"""
    run(encoder, sentences, options)

    Tag multiple sentences.
"""
function run(encoder, sentences::Array{Sentence,1}, options::Dict{Symbol,Any})
    wordIndex = loadIndex(options[:wordPath])
    shapeIndex = loadIndex(options[:shapePath])
    posIndex = loadIndex(options[:posPath])
    labelIndex = loadIndex(options[:labelPath])
    run(encoder, sentences, options, wordIndex, shapeIndex, posIndex, labelIndex)
end

"""
    run(encoder, sentences, options, wordIndex, shapeIndex, posIndex, labelIndex)

    Tag multiple sentences.
"""
function run(encoder, sentences::Array{Sentence,1}, options, wordIndex, shapeIndex, posIndex, labelIndex)
    X = Array{Array{Int,2},1}()
    paddingX = [wordIndex[options[:paddingX]]; 1; 1]
    for sentence in sentences
        xs = map(token -> [get(wordIndex, lowercase(token.word), 1), get(shapeIndex, shape(token.word), 1), get(posIndex, token.annotation[:upos], 1)], sentence.tokens)
        # pad the columns of xs to maxSequenceLength
        if length(xs) > options[:maxSequenceLength]
            xs = xs[1:options[:maxSequenceLength]]
        end
        for t=length(xs)+1:options[:maxSequenceLength]
            push!(xs, paddingX) 
        end
        push!(X, Flux.batch(xs))
    end
    # build batches of data
    Xb = Iterators.partition(X, options[:batchSize])
    # stack each input batch as a 3-d matrix
    Xs = map(b -> Int.(Flux.batch(b)), Xb)
    
    @info "Tagging sentences. Please wait..."
    Ŷs = collect(Iterators.flatten([run(encoder, X, labelIndex) for X in Xs]))
    @info Ŷs
    prediction = Array{Array{String},1}()
    for i=1:length(sentences)
        n = length(sentences[i].tokens)
        push!(prediction, Ŷs[i][1:n])
    end
    return prediction
end

"""
    run(sentences, options, wordIndex, shapeIndex, posIndex, labelIndex)

    Load a pre-trained encoder and tag given sentences.
"""
function run(sentences::Array{Sentence,1}, options, wordIndex, shapeIndex, posIndex, labelIndex)
    @info "Loading encoder..."
    @load options[:modelPath] encoder
    run(encoder, sentences, options, wordIndex, shapeIndex, posIndex, labelIndex)
end


"""
    loadEncoder(options)

    Load a pre-trained encoder.
"""
function loadEncoder(options)
    @info "Loading a pre-trained part-of-speech tagger (encoderPoS)..."
    @load options[:modelPath] encoder
    return encoder
end


end # module