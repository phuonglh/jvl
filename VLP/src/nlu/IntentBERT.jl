# Intent classification using BERT
# phuonglh

using DelimitedFiles
using Flux
using Flux: @epochs
using BSON: @save, @load
using FLoops
using Random


include("Corpus.jl")

function readData(corpusPath::String, vectorPath::String)
    df = Corpus.readIntents(corpusPath)
    X = readdlm(vectorPath)
    X = Float32.(X)
    return (X, df[:, :intent])
end


optionsBERT = Dict{Symbol,Any}(
    :hiddenSize => 128,
    :batchSize => 64,
    :numEpochs => 50,
    :corpusPath => string(pwd(), "/dat/nlu/sample.txt"),
    :vectorPath => string(pwd(), "/dat/nlu/sample-BERTified.txt"),
    :modelPath => string(pwd(), "/dat/nlu/bert.bson"),
    :labelPath => string(pwd(), "/dat/nlu/label-bert.txt"),
    :numCores => 4,
    :verbose => false,
    :split => [0.8, 0.2],
    :gpu => false
)

"""
    evaluate(encoder, Xs, Ys, options)

    Evaluates the accuracy of the classifier using threaded execution.
"""
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

function train(options)
    X, y = readData(options[:corpusPath], options[:vectorPath])
    Random.seed!(220712)
    n = length(y)
    xs = shuffle(1:n)
    X, y = X[xs,:], y[xs]
    j = Int(round(n*options[:split][2]))
    X_test, y_test = X[1:j,:], y[1:j]
    X_train, y_train = X[j+1:n,:], y[j+1:n]
    labels = unique(y)
    labelIndex = Dict{String,Int}(x => i for (i, x) in enumerate(labels))
    saveIndex(labelIndex, options[:labelPath])
    @info "#(labels) = $(length(labels))"

    batch(X, y) = begin
        xs = [X[i,:] for i=1:size(X,1)]
        ys = map(e -> Flux.onehot(e, labels), y)
        Xb = Iterators.partition(xs, options[:batchSize])
        Yb = Iterators.partition(ys, options[:batchSize])
        # stack each input batch as a matrix
        U = map(b -> Flux.batch(b), Xb)
        # stack each output batch as a 2-d matrix
        V = map(b -> Int.(Flux.batch(b)), Yb)
        return (U, V)
    end
    # prepare batches of data for training and validation
    Xs, Ys = batch(X_train, y_train)
    Xv, Yv = batch(X_test, y_test)

    # define a model for sentence encoding
    encoder = Chain(
        Dense(size(X,2), options[:hiddenSize]),
        Dense(options[:hiddenSize], length(labels))
    )

    # the loss function on a batch
    function loss(Xb, Yb)
        Ŷb = encoder(Xb)
        Flux.reset!(encoder)
        return Flux.logitcrossentropy(Ŷb, Yb)
    end

    trainingData = collect(zip(Xs, Ys))
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