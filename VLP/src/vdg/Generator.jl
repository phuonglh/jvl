# A simple implementation of Vietnamese diacritics generation in Julia
# using recurrent neural networks (GRU, LSTM)
# (C) phuonglh@gmail.com

#module VDG

using Flux
using Flux: @epochs
using BSON: @save, @load

using FLoops
using Random

# using CSV
# using DataFrames
# using Plots

include("DiacriticsRemoval.jl")
include("Utils.jl")
include("BiRNN.jl")
include("Embedding.jl")

options = Dict{Symbol,Any}(
    :sampleSize => 4_000,
    :dataPath => string(pwd(), "/dat/vdg/010K.txt"),
    :alphabetPath => string(pwd(), "/dat/vdg/alphabet.txt"),
    :modelPath => string(pwd(), "/dat/vdg/vdg.bson"),
    :maxSequenceLength => 32,
    :batchSize => 64,
    :numEpochs => 40,
    :padX => 'P',
    :padY => 'Q',
    :consonant => 'S',
    :punctuation => 'U',
    :digit => 'D', 
    :numeric => 'N', 
    :mark => 'M',
    :gpu => false,
    :embeddingSize => 100,
    :hiddenSize => 200,
    :split => [0.8, 0.2],
    :η => 1E-3 # learning rate for Adam optimizer
)

# define the label set
labelSet = union(keys(charMap), values(charMap), [options[:padY], options[:consonant]])
labelVec = unique(labelSet)
sort!(labelVec)
padIdxY  = findall(c -> c == options[:padY], labelVec)[1] # note that findall return a vector!

function g(c::Char)
    if isnumeric(c) return options[:numeric]; end
    if ispunct(c) return options[:punctuation]; end
    if '\u0300' <= c && c <= '\u0323' return options[:mark]; end
    return c
end

# prepare input sequence
function transformInput(text::String)::String
    join(map(c -> g(c), collect(text)))
end

# prepare target (output) sequence
function transformOutput(text::String)::String
    map(c -> c ∈ labelSet ? c : options[:consonant], text)
end

function splitBySpace(s::String, maxSeqLen::Int)
    # convert string to an array of characters for ease of indexing
    x = collect(strip(s))
    xs = Vector{String}()
    while (!isempty(x)) 
        t = min(length(x)+1, maxSeqLen)
        if (t == maxSeqLen) 
            # find the last space
            while (t > 0 && x[t] != ' ') t -= 1; end
        end
        u = x[1:t-1]
        push!(xs, strip(join(u)));
        x = x[t:end]
        @info x
    end
    return xs
end

"""
  vectorize(text, alphabet)

  Transforms a sentence into an array of integer indices based on an alphabet for training. 
  In training mode, the text is normal with diacritics and serves as output sequence. This function 
  returns a pair of matrix `(x, y)` where `x` is a sequence of length `maxSequenceLength` 
  and `y` is a matrix of size `|alphabet| x maxSequenceLength`. 
  In prediction mode, this function returns only `x`. 
"""
function vectorize(text::String, alphabet::Array{Char}, training::Bool=true)
    # truncate or pad input sequences to have the same maxSequenceLength
    n = options[:maxSequenceLength]
    x = removeDiacritics(text)
    x = transformInput(x)
    # slice x into subarrays of equal length; note that we make sure 
    # that the string is always split such that the last character is a space
    xs = collect(Iterators.partition(x, n))
    # xs = splitBySpace(x, n)
    # # pad the subarrays with the pad character
    # for i=1:length(xs)
    #     px = fill(options[:padX], n - length(xs[i]))
    #     xs[i] = vcat(xs[i], px)
    # end
    # pad the last subarray with the pad character
    px = fill(options[:padX], n - length(xs[end]))
    xs[end] = vcat(xs[end], px)
    # now all the subarrays in xs are of the same length, 
    # convert each of them into an integer sequence of length maxSeqLen
    Xs = map(x -> Int.(indexin(x, alphabet)), xs)
    if (training) 
        # ys = collect(Iterators.partition(text, n))
        texte = transformOutput(text)
        ys = collect(Iterators.partition(texte, n))
        py = fill(options[:padY], n - length(ys[end]))
        ys[end] = vcat(ys[end], py)
        Ys = map(y -> Float32.(Flux.onehotbatch(y, labelVec)), ys)
        return (Xs, Ys)
    else
        return Xs
    end
end

"""
    evaluate(model, Xb, Yb)

    Evaluates the accuracy of a mini-batch, return the total labels in the batch and the number 
    of correctly predicted labels. This function is used in training for showing performance score.
"""
function evaluate(model, Xb, Yb)::Tuple{Int,Int}
    batchSize = length(Xb)
    as = map(X -> Flux.onecold(model(X)), Xb)
    bs = map(Y -> Flux.onecold(Y), Yb)
    # find the real length of target sequence (without padding symbol)
    total = 0
    correct = 0
    for i = 1:batchSize
        t = options[:maxSequenceLength]
        while t > 0 && bs[i][t] == padIdxY
            t = t - 1
        end
        total = total + t
        correct = correct + sum(as[i][1:t] .== bs[i][1:t])
    end
    return (total, correct)
end

function train(options)
    # read input data frame and create inp/out sequences
    # df = CSV.File(options[:dataPath]) |> DataFrame
    # ys = map(y -> lowercase(y), df[:, :question])
    lines = readlines(options[:dataPath])
    N = min(options[:sampleSize], length(lines))
    @info "N = $(N)"
    ys = map(y -> lowercase(y), lines[1:N])
    # xs = map(y -> removeDiacritics(y), ys)
    xs = map(y -> transformInput(removeDiacritics(y)), ys)
    # create and save an alphabet of the training data
    alphabet = unique(join(xs))
    sort!(alphabet)
    prepend!(alphabet, options[:padX])
    @info "alphabet = $(join(alphabet))"
    saveAlphabet(alphabet, options[:alphabetPath])
    @info "labelVec = $labelVec"
    @info "size(alphabet) = $(length(alphabet))"
    @info "size(labelVec) = $(length(labelVec))"

    # create training sequences
    XYs = map(y -> vectorize(y, alphabet), ys)
    Xs = collect(Iterators.flatten(map(xy -> xy[1], XYs)))
    Ys = collect(Iterators.flatten(map(xy -> xy[2], XYs)))
    # batch inp/out sequences
    Xbs = collect(Iterators.partition(Xs, options[:batchSize]))
    Ybs = collect(Iterators.partition(Ys, options[:batchSize]))    
    if options[:gpu] 
        @info "Bringing data to GPU..."
        Xbs = map(t -> gpu.(t), Xbs)
        Ybs = map(t -> gpu.(t), Ybs)
    end
    dataset = collect(zip(Xbs, Ybs))
    @info "#(batches) = $(length(dataset))"
    @info "typeof(X1) = $(typeof(Xbs[1]))" 
    @info "typeof(Y1) = $(typeof(Ybs[1]))" 
    # split training/test parts
    Random.seed!(220712)
    n = length(Xbs)
    is = shuffle(1:n)
    dataset = dataset[is]
    j = Int(round(n*options[:split][2]))
    dataset_test = dataset[1:j]       # test part
    dataset_train = dataset[j+1:end]  # training part

    # define a model
    model = Chain(
        Embedding(length(alphabet), options[:embeddingSize]),
        BiGRU(options[:embeddingSize], options[:hiddenSize]),
        Dense(options[:hiddenSize], length(labelVec))
    )
    @info model
    # compute the loss of the model on a batch
    function loss(Xb, Yb)
        function g(X, Y)
            ys = Flux.onecold(Y)
            t = options[:maxSequenceLength]
            while (t > 0 && ys[t] == padIdxY) t = t - 1; end
            Z = model(X)
            return Flux.logitcrossentropy(Z[:,1:t], Y[:,1:t])
        end
        # need to reset the state of the model between each batch
        # since the batches are independent, not a continuation of samples
        Flux.reset!(model) 
        return sum(g(Xb[i], Yb[i]) for i=1:length(Xb))
    end
    optimizer = RMSProp(options[:η], 0.95)
    
    accuracy_test, accuracy_train = Array{Float64,1}(), Array{Float64,1}()
    Js = []
    # bestLoss = Inf
    # stop = false
    function evalcb() 
        J = sum(loss(dataset_train[i]...) for i=1:length(dataset_train))
        L = sum(loss(dataset_test[i]...) for i=1:length(dataset_test))
        @info "J(θ) = $J, L(θ) = $L"
        push!(Js, (J, L))
        # if (L < bestLoss) 
        #     bestLoss = L; 
        # else
        #     stop = true
        # end 

        # pairs_test = [evaluate(model, dataset_test[i]...) for i=1:length(dataset_test)]
        # u, v = reduce(((a1, b1), (a2, b2)) -> (a1 + a2, b1 + b2), pairs_test)
        # push!(accuracy_test, v/u)
        # pairs_train = [evaluate(model, dataset_train[i]...) for i=1:length(dataset_train)]
        # u, v = reduce(((a1, b1), (a2, b2)) -> (a1 + a2, b1 + b2), pairs_train)
        # push!(accuracy_train, v/u)
        # @info "loss = $J, test accuracy = $(accuracy_test[end]), training accuracy = $(accuracy_train[end])"
    end
    for t=1:options[:numEpochs]
        @time Flux.train!(loss, Flux.params(model), dataset_train, optimizer, cb = Flux.throttle(evalcb, 60))
        # if (stop) 
        #     @info "Stop training at iteration $t."
        #     break; 
        # end
    end
    if (options[:gpu])
        model = model |> cpu
    end
    @save options[:modelPath] model
    # report test accuracy and training accuracy
    pairs_test = [evaluate(model, dataset_test[i]...) for i=1:length(dataset_test)]
    u, v = reduce(((a1, b1), (a2, b2)) -> (a1 + a2, b1 + b2), pairs_test)
    push!(accuracy_test, v/u)
    pairs_train = [evaluate(model, dataset_train[i]...) for i=1:length(dataset_train)]
    u, v = reduce(((a1, b1), (a2, b2)) -> (a1 + a2, b1 + b2), pairs_train)
    push!(accuracy_train, v/u)
    @info "test accuracy = $(accuracy_test[end]), training accuracy = $(accuracy_train[end])"
    # plot(1:length(Js), xlabel="iteration", ylabel="loss", label=["J"])
    return model, Js
end

function predict(text::String, model, alphabet::Array{Char})::String
    Flux.reset!(model) # important since the batch size is changed
    Xs = vectorize(lowercase(text), alphabet, false)
    zs = map(X -> Flux.onecold(model(X)), Xs)
    cs = map(z -> join(labelVec[z]), zs)
    texte = collect(join(cs))[1:length(text)]
    @info texte
    is = findall(c -> c == options[:consonant], texte)
    texte[is] .= collect(text)[is]
    return join(texte)
end

function test(text, model)
    alphabet = loadAlphabet(options[:alphabetPath])
    test(text, model, alphabet)
end

function test(text, model, alphabet)
    Xs, Ys = vectorize(lowercase(text), alphabet)
    # input
    xs = Flux.onecold(Xs[1])
    @info alphabet[xs]
    # target
    ys = Flux.onecold(Ys[1])
    vs = join(map(y -> labelVec[y], ys))
    @info vs
    # prediction
    zs = Flux.onecold(model(Xs[1]))
    ws = join(map(z -> labelVec[z], zs))
    @info ws
end

#end # module

