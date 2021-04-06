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
function batch(sentences::Array{Sentence}, wordIndex::Dict{String,Int}, shapeIndex::Dict{String,Int}, posIndex::Dict{String,Int}, labelIndex::Dict{String,Int}, options=optionsVLSP2016)
    X, Y0, Y1 = Array{Array{Int,2},1}(), Array{Array{Int,2},1}(), Array{Array{Int,2},1}()
    paddingX = [wordIndex[options[:paddingX]]; 1; 1]
    numLabels = length(labelIndex)
    paddingY = Flux.onehot(labelIndex[options[:paddingY]], 1:numLabels)
    for sentence in sentences
        tokens = sentence.tokens[2:end] # not consider the ROOT token of UD graphs
        xs = map(token -> [get(wordIndex, lowercase(token.word), 1), get(shapeIndex, shape(token.word), 1), get(posIndex, token.annotation[:upos], 1)], tokens)
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
    encode(Xb, embedding, encoder)

    Encodes a batch, each element in the batch is a matrix representing a sequence. 
"""
encode(Xb::Array{Array{Int,2},1}, embedding, encoder) = [encode(X, embedding, encoder) for X in Xb]

"""
    decode(H, y0, decoder, α, β, linear)

    H is the hidden states of the encoder, which is a matrix of size (hiddenSize x maxSequenceLength)
    and y0 is an one-hot vector representing a label at position t.
"""
function decode(H::Array{Float32,2}, y0::Array{Int,1}, decoder, α, β, linear)
    w = α(β(decoder.state, H))
    c = sum(w .* H, dims=2)
    v = vcat(y0, c)
    linear(decoder(v))
end

function decode(H::Array{Float32,2}, Y0::Array{Int,2}, encoder, decoder, α, β, linear)
    # take the last state of the encoder as the init state for the decoder
    decoder.init = encoder.state[:,end] 
    # decode positions, one by one
    y0s = [Y0[:, t] for t=1:size(Y0,2)]
    ŷs = [decode(H, y0, decoder, α, β, linear) for y0 in y0s]
    # stack the output array into a 2-d matrix of size (hiddenSize x maxSequenceLength)
    hcat(ŷs...)
end

function decode(Hb::Array{Array{Float32,2},1}, Y0b::Array{Array{Int,2},1}, encoder, decoder, α, β, linear)    
    [decode(Hb[i], Y0b[i], encoder, decoder, α, β, linear) for i=1:length(Hb)]
end

function model(Xb, Y0b, embedding, encoder, decoder, α, β, linear)
    Ŷb = decode(encode(Xb, embedding, encoder), Y0b, encoder, decoder, α, β, linear)
    return Ŷb
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
    append!(vocabularies.labels, ["BOS", "EOS"])
    labelIndex = Dict{String,Int}(label => i for (i, label) in enumerate(vocabularies.labels))

    # save the vocabulary, shape, part-of-speech and label information to external files
    saveIndex(wordIndex, options[:wordPath])
    saveIndex(shapeIndex, options[:shapePath])
    saveIndex(posIndex, options[:posPath])
    saveIndex(labelIndex, options[:labelPath])

    # create batches of data
    Xbs, Y0bs, Ybs = batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex)
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
    #  an output position and an input position. The attention model that we use here is a simple linear model.
    attention = Dense(2*options[:hiddenSize], 1)

    """
        β(s, H)

        Align a decoder state `s` with hidden states of inputs `h` of size (hiddenSize x m). 
        The decoder state is a column vector `s` of length hiddenSize, it should be repeated to create 
        the same number of columns as `h`, that is of size (hiddenSize x m). 

        This function computes attention scores matrix of size (1 x maxSequenceLength) for a decoder position.
    """
    function β(s, H::Array{Float32,2})
        V = s .* Float32.(ones(1, size(H,2)))
        attention(vcat(H, V))
    end

    """
        α(β)

        Compute the probabilities (weights) vector by using the softmax function. Return a matrix of the same 
        size as β, that is (1 x maxSequenceLength).
    """
    function α(β::Array{Float32,2})
        score = exp.(β)
        s = sum(score, dims=2)
        score ./ s
    end

    # 3. Create a decoder
    numLabels = length(labelIndex)
    decoder = GRU(options[:hiddenSize] + numLabels, options[:hiddenSize])
    linear = Dense(options[:hiddenSize], numLabels)
    # The full machinary
    machine = Chain(embedding, encoder, attention, decoder, linear)

    # define the loss function
    loss(Xb, Y0b, Yb) = sum(Flux.logitcrossentropy.(model(Xb, Y0b, embedding, encoder, decoder, α, β, linear), Yb))

    Ubs, Vbs, Wbs = batch(sentencesValidation, wordIndex, shapeIndex, posIndex, labelIndex)
    datasetValidation = collect(zip(Ubs, Vbs, Wbs))

    # should use a small learning rate
    optimizer = ADAM(1E-5)
    file = open(options[:logPath], "w")
    write(file, "loss,trainingAccuracy,validationAccuracy\n")
    evalcb = function()
        ℓ = sum(loss(datasetValidation[i]...) for i=1:length(datasetValidation))
        trainingAccuracy = evaluate(embedding, encoder, decoder, α, β, linear, Xbs, Y0bs, Ybs, options)
        validationAccuracy = evaluate(embedding, encoder, decoder, α, β, linear, Ubs, Vbs, Wbs, options)
        @info string("loss = ", ℓ, ", training accuracy = ", trainingAccuracy, ", validation accuracy = ", validationAccuracy)
        write(file, string(ℓ, ',', trainingAccuracy, ',', validationAccuracy, "\n"))
        # gs = gradient(() -> loss(Xbs[1], Y0bs[1], Ybs[1]), params(machine))
        # eg = sum(gs[encoder.cell.Wh])
        # dg = sum(gs[decoder.cell.Wh])
        # @info "\tsum(encoder gradient) = $(eg), sum(decoder gradient) = $(dg)"
    end
    # train the model until the validation accuracy decreases 2 consecutive times
    t = 1
    k = 0
    bestDevAccuracy = 0
    @time while (t <= options[:numEpochs]) 
        @info "Epoch $t, k = $k"
        Flux.train!(loss, params(machine), dataset, optimizer, cb = Flux.throttle(evalcb, 60))
        devAccuracy = evaluate(embedding, encoder, decoder, α, β, linear, Ubs, Vbs, Wbs, options)
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
    close(file)
    
    # save the model to a BSON file
    @save options[:modelPath] machine

    @info "Total weight of final word embeddings = $(sum(embedding.word.W))"
    @info "Evaluating the model on the training set..."
    accuracy = evaluate(embedding, encoder, decoder, α, β, linear, Xbs, Y0bs, Ybs, options)
    @info "Training accuracy = $accuracy"
    accuracyValidation = evaluate(embedding, encoder, decoder, α, β, linear, Ubs, Vbs, Wbs, options)
    @info "Validation accuracy = $accuracyValidation"
    machine
end

"""
    evaluate(embedding, encoder, decoder, α, β, linear, Xbs, Y0bs, Ybs, options)

    Evaluate the accuracy of the model on a dataset. 
"""
function evaluate(embedding, encoder, decoder, α, β, linear, Xbs, Y0bs, Ybs, options)
    numBatches = length(Xbs)
    @floop ThreadedEx(basesize=numBatches÷options[:numCores]) for i=1:numBatches
        Ŷb = Flux.onecold.(model(Xbs[i], Y0bs[i], embedding, encoder, decoder, α, β, linear))
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
    @info "\tTotal matched tokens = $(numMatches)/$(numTokens)"
    return numMatches/numTokens
end

function trainVUD()
    options = optionsVUD
    options[:wordSize] = 16
    options[:hiddenSize] = 32
    train(options)
end

# """
#     predict(model, Xbs, Y0bs, Ybs,  split, options)

#     Predict a (training) data set, save result to a CoNLL-2003 evaluation script.
# """
# function predict(model, Xbs, Y0bs, Ybs, split::Symbol, paddingY::Int=1)
#     numBatches = length(Xbs)
#     @floop ThreadedEx(basesize=numBatches÷options[:numCores]) for i=1:numBatches
#         Ŷb = Flux.onecold.(model(Xbs[i], Y0bs[i]))
#         Yb = Flux.onecold.(Ybs[i])
#         truth, pred = Array{Array{String,1},1}(), Array{Array{String,1},1}()
#         for t=1:length(Yb)
#             n = options[:maxSequenceLength]
#             # find the last position of non-padded element
#             while Yb[t][n] == paddingY
#                 n = n - 1
#             end
#             push!(truth, vocabularies.labels[Yb[t][1:n]])
#             push!(pred, vocabularies.labels[Ŷb[t][1:n]])
#         end
#         @reduce(ss = append!!(EmptyVector(), [(truth, pred)]))
#     end
#     file = open(options[split], "w")
#     result = Array{String,1}()
#     for b=1:numBatches
#         truths = ss[b][1]
#         preds = ss[b][2]
#         for i = 1:length(truths)
#             x = map((a, b) -> string(a, ' ', b), truths[i], preds[i])
#             s = join(x, "\n")
#             push!(result, s * "\n")
#         end
#     end
#     write(file, join(result, "\n"))
#     close(file)
# end


# """
#     predict(sentence, labelIndex)

#     Find the label sequence for a given sentence.
# """
# function predict(sentence, labelIndex::Dict{String,Int})
#     Flux.reset!(machine)
#     ps = [labelIndex["BOS"]]
#     Xs, Y0s, Ys = batch([sentence], wordIndex, shapeIndex, posIndex, labelIndex)
#     Xb = first(Xs)
#     Hb = encode(Xb)
#     Y0 = repeat(Flux.onehotbatch(ps, 1:numLabels), 1,size(Xb[1], 2))
#     m = min(length(sentence.tokens), size(Xb[1], 2))
#     for t=1:m
#         currentY = Flux.onehot(ps[end], 1:numLabels)
#         Y0[:,t] = currentY
#         Y0b = [ Int.(Y0) ]
#         output = decode(Hb, Y0b)
#         Ŷ = softmax(output[1][:,t])
#         # nextY = Flux.onecold(Ŷ)     # use a hard selection approach, always choose the label with the best probability
#         nextY = wsample(1:numLabels, Ŷ) # use a soft selection approach to sample a label from the distribution
#         push!(ps, nextY)
#     end
#     return vocabularies.labels[ps[2:end]]
# end

# """
#     predict(sentences, labelIndex)
# """
# function predict(sentences::Array{Sentence}, labelIndex::Dict{String,Int})
#     map(sentence -> predict(sentence, labelIndex), sentences)
# end

# function diagnose(sentence)
#     Xs, Y0s, Ys = batch([sentence], wordIndex, shapeIndex, posIndex, labelIndex)
#     Xb = first(Xs)
#     H = encode(first(Xb))
#     Y0b = first(Y0s)
#     Y0 = first(Y0b)
#     vocabularies.labels[Flux.onecold(decode(H, Y0))]
# end

