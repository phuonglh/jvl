# phuonglh@gmail.com
# Implementation of a Natural Language Inference (NLI) model using BERT

module NLI

using Transformers
using Transformers.Basic
using Transformers.Pretrain

using Distributed
using Flux
using Flux: @epochs
using BSON: @save, @load
using FLoops

include("Corpus.jl")
using .Corpus

options = Dict{Symbol,Any}(
    :batchSize => 32,
    :hiddenSize => 32,
    :modelPath => string(pwd(), "/dat/nli/x/032/en.bson"),
    :numEpochs => 40,
    :numCores => 4,
    :gpu => false
)

# load a pre-trained BERT model for English (see ~/.julia/datadeps/)
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
# load mBERT (see ~/.julia/datadeps/)
# bert_model, wordpiece, tokenizer = pretrain"bert-multi_cased_L-12_H-768_A-12"
vocab = Vocabulary(wordpiece)

# read train/dev./test datasets
trainDF, devDF, testDF = Corpus.readXNLI("dat/nli/x/en.train.jsonl"), Corpus.readXNLI("dat/nli/x/en.dev.json", false), Corpus.readXNLI("dat/nli/x/en.test.json", false)

"""
    featurize(sentence1, sentence2)

    Find the BERT representation of a pair of sentences. This function returns a `768 x T` matrix where 
    `T` is the total length of the two sentences.
"""
function featurize(sentence1, sentence2)
    pieces1 = sentence1 |> tokenizer |> wordpiece
    pieces2 = sentence2 |> tokenizer |> wordpiece
    pieces = ["[CLS]"; pieces1; "[SEP]"; pieces2; "[SEP]"]
    piece_indices = vocab(pieces)
    segment_indices = [fill(1, length(pieces1)+2); fill(2, length(pieces2)+1)]

    sample = (tok = piece_indices, segment = segment_indices)
    embeddings = sample |> bert_model.embed
    features = embeddings |> bert_model.transformers
    return features
end

function batch(df, training::Bool=true)
    pairs = zip(df[:, :sentence1], df[:, :sentence2])
    Xs = collect(Iterators.partition(pairs, options[:batchSize]))
    # take the last hidden state of the BERT representation
    function f(xs)
        Xb = map(x -> featurize(x[1], x[2])[:, end], xs)
        return Flux.stack(Xb, 2)
    end
    # each batch of sentence pairs is transformed into a matrix of shape (768 x batchSize)
    Xb = pmap(xs -> f(xs), Xs)
    if training
        labels = df[:, :label]
        ys = collect(Iterators.partition(labels, options[:batchSize]))
        # each batch of labels is transformed into an onehot matrix of shape (3 x batchSize)
        Yb = map(y -> Flux.onehotbatch(y, 1:3), ys)
        return Xb, Yb
    else
        return Xb
    end
end

# define a neural network of two layers for classification
model = Chain(Dense(768, options[:hiddenSize]), Dense(options[:hiddenSize], 3))

function train(model)
    @info "Extracting BERT representations of the training samples..."
    @time Xb, Yb = batch(trainDF)
    @info "Extracting BERT representations of the development samples..."
    @time Xb_dev, Yb_dev = batch(devDF)
    # the loss function on a batch (X, y)
    function loss(X, Y)
        Ŷ = model(X)
        return Flux.logitcrossentropy(Ŷ, Y)
    end

    # bring data and model to GPU if set
    trainingData = collect(zip(Xb, Yb))
    developmentData = collect(zip(Xb_dev, Yb_dev))
    @info string("Number of training batches = ", length(trainingData))
    @info string("Number of development batches = ", length(developmentData))
    if options[:gpu]
        @info "Moving data to GPU..."
        trainingData = map(p -> p |> gpu, trainingData)
        developmentData = map(p -> p |> gpu, developmentData)
        @info "Moving model to GPU..."
        model = model |> gpu
    end
    optimizer = ADAM()
    accuracy = Array{Tuple{Float64,Float64},1}()
    evalcb = function()
        ℓ = sum(loss(trainingData[i]...) for i=1:length(trainingData))
        a = evaluate(model, Xb, Yb, options)
        b = evaluate(model, Xb_dev, Yb_dev, options)
        @info string("loss = ", ℓ, ", training accuracy = ", a, ", development accuracy = ", b)
        push!(accuracy, (a, b))
    end
    # train the model until the validation accuracy decreases 2 consecutive times
    t, k = 1, 0
    bestDevAccuracy = 0
    @time while (t <= options[:numEpochs]) 
        @info "Epoch $t, k = $k"
        Flux.train!(loss, params(model), trainingData, optimizer, cb = Flux.throttle(evalcb, 60))
        devAccuracy = evaluate(model, Xb, Yb, options)
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
        model = model |> cpu
    end
    @save options[:modelPath] model
    # compute test score
    @info "Extracting BERT representations of the test samples..."
    @time Xb_test, Yb_test = batch(testDF)
    c = evaluate(model, Xb_test, Yb_test, options)
    @info string("test accuracy = ", c)
end

"""
    evaluate(model, Xb, Yb, options)

    Evaluates the accuracy of the classifier using threaded execution.
"""
function evaluate(model, Xb, Yb, options)
    numBatches = length(Xb)
    @floop ThreadedEx(basesize=numBatches÷options[:numCores]) for i=1:numBatches
        Ŷ = Flux.onecold(model(Xb[i]))
        Y = Flux.onecold(Yb[i])  
        matches = sum(Ŷ .== Y)
        @reduce(numMatches += matches, numSents += length(Y))
    end
    @info "Total matches = $(numMatches)/$(numSents)"
    return 100 * (numMatches/numSents)
end

"""
    evaluate(model, df, options)

    Evaluates the accuracy of the classifier on a data frame.
"""
function evaluate(model, df, options)
    @info "Extracting BERT representations of the training samples..."
    @time Xb, Yb = batch(df)
    evaluate(model, Xb, Yb, options)
end

end # module