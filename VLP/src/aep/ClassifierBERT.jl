# phuonglh@gmail.com, December 2020.

module TransitionClassifierBERT

using Flux
using Flux: @epochs
using BSON: @save, @load
using CUDA

include("EmbeddingWSP.jl")
include("BagOfBERT.jl")
include("Classifier.jl")
using .TransitionClassifier.Corpus
using .TransitionClassifier

# This should be included after `Oracle.jl to override options...`
include("Options.jl")

using Flux
using Transformers
using Transformers.Basic
using Transformers.Pretrain

# load a pre-trained BERT model for English (see ~/.julia/datadeps/)
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
# load mBERT (see ~/.julia/datadeps/)
# bert_model, wordpiece, tokenizer = pretrain"bert-multi_cased_L-12_H-768_A-12"
vocab = Vocabulary(wordpiece)


"""
    bertify(sentence)

    Transforms a sentence to a vector using a pre-trained BERT model.
"""
function bertify(sentence::String)::Matrix{Float32}
    pieces = sentence |> tokenizer |> wordpiece
    pieces = [pieces; "[pad]"]
    piece_indices = vocab(pieces)
    segment_indices = fill(1, length(pieces))

    sample = (tok = piece_indices, segment = segment_indices)
    embeddings = sample |> bert_model.embed
    # compute a matrix of shape (768 x length(pieces))
    features = embeddings |> bert_model.transformers
    return features
end

"""
    vectorize(sentence, wordIndex, shapeIndex, posIndex, labelIndex, options)

    Vectorize a training sentence. An oracle is used to extract (context, transition) pairs from 
    the sentence. Then each context is vectorized to a tuple of (token matrix of the sentence, word id array of the context).
    The word id array of the sentence is the same across all contexts. This function returns an array of pairs (xs, ys) where 
    each xs is a pair (w, x). Each token matrix is a 3-row matrix corresponding to the word id, shape id, and part-of-speech id arrays.
"""
function vectorize(sentence::Sentence, wordIndex::Dict{String,Int}, shapeIndex::Dict{String,Int}, posIndex::Dict{String,Int}, labelIndex::Dict{String,Int}, options)
    ws = TransitionClassifier.vectorizeSentence(sentence, wordIndex, shapeIndex, posIndex, options)
    contexts = TransitionClassifier.decode(sentence)
    fs = map(context -> TransitionClassifier.extract(context.features, ["ws", "wq"]), contexts)
    words = map(token -> lowercase(token.word), sentence.tokens)
    append!(words, [options[:padding]])
    positionIndex = Dict{String,Int}(word => i for (i, word) in enumerate(words))
    xs = map(f -> map(word -> positionIndex[lowercase(word)], f), fs)
    s = join(map(token -> token.word, sentence.tokens), " ")
    bert = bertify(s)
    ys = map(context -> get(labelIndex, context.transition, labelIndex["SH"]), contexts)
    # return a collection of tuples for this sentence, use Flux.batch to convert ws to a matrix of size 3 x (maxSequenceLength+1).
    # xs to a matrix of size 4 x numberOfContexts
    # s is a raw sentence (for feeding into a BERT model)
    # convert each output batch to an one-hot matrix of size (numLabels x numberOfContexts)
    ((Flux.batch(ws), Flux.batch(xs), bert), Flux.onehotbatch(ys, 1:length(labelIndex)))
end

"""
    batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex, options)

    Create batches of data for training or evaluating. Each batch contains a pair (`Xb`, `Yb`) where 
    `Xb` is an array of `batchSize` samples. `Yb` is an one-hot matrix of size (`numLabels` x `batchSize`).
"""
function batch(sentences::Array{Sentence}, wordIndex::Dict{String,Int}, shapeIndex::Dict{String,Int}, posIndex::Dict{String,Int}, labelIndex::Dict{String,Int}, options)
    # vectorizes all sentences 
    samples = map(sentence -> vectorize(sentence, wordIndex, shapeIndex, posIndex, labelIndex, options), sentences)
    X = map(sample -> sample[1], samples)
    Y = map(sample -> sample[2], samples)
    # build batches of data for training
    Xs = collect(Iterators.partition(X, options[:batchSize]))
    Ys = collect(Iterators.partition(Y, options[:batchSize]))
    (Xs, Ys)
end


"""
    train(options)

    Train a classifier model.
"""
function train(options)
    sentences = Corpus.readCorpusUD(options[:trainCorpus], options[:maxSequenceLength])
    @info "#(sentencesTrain) = $(length(sentences))"
    contexts = collect(Iterators.flatten(map(sentence -> TransitionClassifier.decode(sentence), sentences)))
    @info "#(contextsTrain) = $(length(contexts))"
    vocabularies = TransitionClassifier.buildVocab(contexts)

    prepend!(vocabularies.words, [options[:unknown]])

    labelIndex = Dict{String, Int}(label => i for (i, label) in enumerate(vocabularies.labels))
    wordIndex = Dict{String, Int}(word => i for (i, word) in enumerate(vocabularies.words))
    shapeIndex = Dict{String, Int}(shape => i for (i, shape) in enumerate(vocabularies.shapes))
    posIndex = Dict{String, Int}(tag => i for (i, tag) in enumerate(vocabularies.partsOfSpeech))

    vocabSize = min(options[:vocabSize], length(wordIndex))
    @info "vocabSize = $(vocabSize)"

    # build training dataset
    println("Building training dataset...")
    Xs, Ys = batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex, options)
    dataset = collect(zip(Xs, Ys))
    @info "numBatches  = $(length(dataset))"
    @info size(Xs[1][1][1]), size(Xs[1][1][2]), size(Xs[1][1][3])
    @info size(Ys[1][1])

    mlp = Chain(
        BagOfBERT(
            EmbeddingWSP(vocabSize, options[:wordSize], length(shapeIndex), options[:shapeSize], length(posIndex), options[:posSize]),
            identity,
            identity
        ),
        Dense(options[:featuresPerContext] * (options[:wordSize] + options[:shapeSize] + options[:posSize]) + 768, options[:hiddenSize], σ),
        Dense(options[:hiddenSize], length(labelIndex))
    )
    # save an index to an external file
    function saveIndex(index, path)
        file = open(path, "w")
        for f in keys(index)
            write(file, string(f, " ", index[f]), "\n")
        end
        close(file)
    end
    saveIndex(wordIndex, options[:wordPath])
    saveIndex(shapeIndex, options[:shapePath])
    saveIndex(posIndex, options[:posPath])
    saveIndex(labelIndex, options[:labelPath])

    # bring the dataset and the model to GPU if any
    if options[:gpu]
        @info "Bring data to GPU..."
        dataset = map(p -> p |> gpu, dataset)
        mlp = mlp |> gpu
    end

    @info "Total weight of initial word embeddings = $(sum(mlp[1].fs[1].word.W))"

    # build development dataset
    println("Building development dataset...")
    sentencesDev = Corpus.readCorpusUD(options[:validCorpus], options[:maxSequenceLength])
    @info "#(sentencesDev) = $(length(sentencesDev))"
    contextsDev = collect(Iterators.flatten(map(sentence -> TransitionClassifier.decode(sentence), sentencesDev)))
    @info "#(contextsDev) = $(length(contextsDev))"

    XsDev, YsDev = batch(sentencesDev, wordIndex, shapeIndex, posIndex, labelIndex, options)
    datasetDev = collect(zip(XsDev, YsDev))
    @info "numBatchesDev  = $(length(datasetDev))"

    # define a loss function, an optimizer and train the model
    function loss(X, Y)
        value = sum(Flux.logitcrossentropy(mlp(X[i]), Y[i]) for i=1:length(Y))
        Flux.reset!(mlp)
        return value
    end
    optimizer = ADAM()
    file = open(options[:logPath], "w")
    write(file, "dev. loss,trainingAcc,devAcc\n")
    evalcb = function()
        devLoss = sum(loss(datasetDev[i]...) for i=1:length(datasetDev))
        mlpc = mlp |> cpu
        trainingAccuracy = TransitionClassifier.evaluate(mlpc, Xs, Ys)
        devAccuracy = TransitionClassifier.evaluate(mlpc, XsDev, YsDev)
        @info string("\tdevLoss = $(devLoss), training accuracy = $(trainingAccuracy), development accuracy = $(devAccuracy)")
        write(file, string(devLoss, ',', trainingAccuracy, ',', devAccuracy, "\n"))
    end
    # train the model until the development loss increases
    t = 1
    k = 0
    bestDevAccuracy = 0
    @time while (t <= options[:numEpochs])
        @info "Epoch $t, k = $k"
        Flux.train!(loss, params(mlp), dataset, optimizer, cb = Flux.throttle(evalcb, 60))
        devAccuracy = TransitionClassifier.evaluate(mlp, XsDev, YsDev)
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
    @info "Total weight of final word embeddings = $(sum(mlp[1].fs[1].word.W))"

    # evaluate the model on the training set
    @info "Evaluating the model..."
    accuracy = TransitionClassifier.evaluate(mlp, Xs, Ys)
    @info "Training accuracy = $accuracy"
    accuracyDev = TransitionClassifier.evaluate(mlp, XsDev, YsDev)
    @info "Development accuracy = $(accuracyDev)"
    
    # save the model to a BSON file
    if (options[:gpu])
        mlp = mlp |> cpu
    end
    @save options[:modelPath] mlp
    mlp
end


end # module