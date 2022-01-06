# phuonglh@gmail.com, December 2020.

module TransitionClassifierBERT

using Flux
using Flux: @epochs
using BSON: @save, @load
using CUDA

include("Classifier.jl")
using .TransitionClassifier

"""
    train(options)

    Train a classifier model.
"""
function train(options)
    sentences = Corpus.readCorpusUD(options[:trainCorpus], options[:maxSequenceLength])
    @info "#(sentencesTrain) = $(length(sentences))"
    contexts = collect(Iterators.flatten(map(sentence -> decode(sentence), sentences)))
    @info "#(contextsTrain) = $(length(contexts))"
    vocabularies = vocab(contexts)

    prepend!(vocabularies.words, [options[:unknown]])

    labelIndex = Dict{String, Int}(label => i for (i, label) in enumerate(vocabularies.labels))
    wordIndex = Dict{String, Int}(word => i for (i, word) in enumerate(vocabularies.words))
    shapeIndex = Dict{String, Int}(shape => i for (i, shape) in enumerate(vocabularies.shapes))
    posIndex = Dict{String, Int}(tag => i for (i, tag) in enumerate(vocabularies.partsOfSpeech))

    vocabSize = min(options[:vocabSize], length(wordIndex))
    @info "vocabSize = $(vocabSize)"

    # build training dataset
    Xs, Ys = batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex, options)
    dataset = collect(zip(Xs, Ys))
    @info "numBatches  = $(length(dataset))"
    @info size(Xs[1][1][1]), size(Xs[1][1][2])
    @info size(Ys[1][1])

    recurrentLayer = if options[:bidirectional]
        BiGRU(options[:wordSize] + options[:shapeSize] + options[:posSize], options[:recurrentSize])
    else
        GRU(options[:wordSize] + options[:shapeSize] + options[:posSize], options[:recurrentSize])
    end
    mlp = Chain(
        Join(
            EmbeddingWSP(vocabSize, options[:wordSize], length(shapeIndex), options[:shapeSize], length(posIndex), options[:posSize]),
            recurrentLayer
        ),
        Dense(options[:featuresPerContext] * options[:recurrentSize], options[:hiddenSize], σ),
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
    # @info typeof(dataset[1][1]), size(dataset[1][1])
    # @info typeof(dataset[1][2]), size(dataset[1][2])

    @info "Total weight of initial word embeddings = $(sum(mlp[1].fs[1].word.W))"

    # build development dataset
    sentencesDev = Corpus.readCorpusUD(options[:validCorpus], options[:maxSequenceLength])
    @info "#(sentencesDev) = $(length(sentencesDev))"
    contextsDev = collect(Iterators.flatten(map(sentence -> decode(sentence), sentencesDev)))
    @info "#(contextsDev) = $(length(contextsDev))"

    Xs, Ys = batch(sentences, wordIndex, shapeIndex, posIndex, labelIndex, options)
    dataset = collect(zip(Xs, Ys))
    @info "numBatches  = $(length(dataset))"

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
        trainingAccuracy = evaluate(mlpc, Xs, Ys)
        devAccuracy = evaluate(mlpc, XsDev, YsDev)
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
        devAccuracy = evaluate(mlp, XsDev, YsDev)
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
    accuracy = evaluate(mlp, Xs, Ys)
    @info "Training accuracy = $accuracy"
    accuracyDev = evaluate(mlp, XsDev, YsDev)
    @info "Development accuracy = $(accuracyDev)"
    
    # save the model to a BSON file
    if (options[:gpu])
        mlp = mlp |> cpu
    end
    @save options[:modelPath] mlp
    mlp
end


end # module