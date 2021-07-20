# Vietnamese diacritics generation
# (C) phuonglh@gmail.com, 2021

module VDG

using Base.Iterators: isempty
using FLoops: Iterators
using CSV
using DataFrames
using Flux
using Flux: @epochs
using BSON: @save, @load

using FLoops


include("DiacriticsRemoval.jl")


options = Dict{Symbol,Any}(
    :dataPath => string(pwd(), "/dat/qas/monre.csv"),
    :labelPath => string(pwd(), "/dat/vdg/label.txt"),
    :alphabetPath => string(pwd(), "/dat/vdg/alphabet.txt"),
    :modelPath => string(pwd(), "/dat/vdg/vdg.bson"),
    :maxSequenceLength => 64,
    :batchSize => 16,
    :numEpochs => 10,
    :unkChar => 'X',
    :padChar => 'P',
    :gpu => false,
    :hiddenSize => 64
)

"""
  vectorize(text, alphabet, label=Array{Char,1}())

  Transforms a sentence into an array of vectors based on an alphabet and 
  a label set for training. In training mode, `label` is given and this function 
  returns a pair of matrix `(x, y)` where `x` is a matrix of size `|alphabet| x maxSequenceLength` 
  and `y` is a matrix of size `|label| x maxSequenceLength`. 
  In test mode, this function returns only `x` matrix.
"""
function vectorize(text::String, alphabet::Array{Char}, label=Array{Char,1}())
    # truncate or pad input sequences to have the same maxSequenceLength
    n = options[:maxSequenceLength]
    x = removeDiacritics(text)
    # slice x into subarrays of equal length
    xs = collect(Iterators.partition(x, n))
    # pad the last subarray with the pad character
    ps = fill(options[:padChar], n - length(xs[end]))
    xs[end] = vcat(xs[end], ps)
    # now all the subarrays in xs is of the same length, 
    # convert them into one-hot matrices of size (|alphabet| x maxSeqLen)
    Xs = map(x -> Float32.(Flux.onehotbatch(x, alphabet)), xs)
    if (!isempty(label)) # training mode
        texte = map(c -> c ∈ keys(charMap) ? c : options[:unkChar], text)
        ys = collect(Iterators.partition(texte, n))
        ys[end] = vcat(ys[end], ps)
        Ys = map(y -> Float32.(Flux.onehotbatch(y, label)), ys)
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
    # find the real length of target sequence (without padding symbol of index 1)
    total = 0
    correct = 0
    padding = 1
    for i = 1:batchSize
        t = options[:maxSequenceLength]
        while t > 0 && bs[i][t] == padding
            t = t - 1
        end
        total = total + t
        correct = correct + sum(as[i][1:t] .== bs[i][1:t])
    end
    Flux.reset!(model)
    return (total, correct)
end

"""
    saveIndex(index, path)
    
    Save an index to an external file.
"""
function saveIndex(index, path)
    file = open(path, "w")
    for f in keys(index)
        write(file, string(f, " ", index[f]), "\n")
    end
    close(file)
end

function train(options)
    # read input data frame and create inp/out sequences
    df = CSV.File(options[:dataPath]) |> DataFrame
    ys = map(y -> lowercase(y), df[:, :question])
    xs = map(y -> removeDiacritics(y), ys)
    # create and save alphabet index
    alphabetY = unique(join(ys))
    alphabet = unique(join(alphabetY, values(charMap)))
    prepend!(alphabet, options[:padChar])
    charIndex = Dict{Char,Int}(c => i for (i, c) in enumerate(alphabet))
    saveIndex(charIndex, options[:alphabetPath])
    # create and save label index
    label = collect(keys(charMap))
    prepend!(label, options[:unkChar])
    prepend!(label, options[:padChar])
    labelIndex = Dict{Char,Int}(c => i for (i, c) in enumerate(label))
    saveIndex(labelIndex, options[:labelPath])
    # create training sequences
    XYs = map(x -> vectorize(x, alphabet, label), xs)
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
    # define a model
    model = Chain(
        GRU(length(alphabet), options[:hiddenSize]), 
        Dense(options[:hiddenSize], length(label))
    )
    @info model
    # compute the loss of the model on a batch
    function loss(Xb, Yb)
        batchSize = length(Xb)
        value = sum(Flux.logitcrossentropy(model(Xb[i]), Yb[i]) for i=1:batchSize)
        Flux.reset!(model)
        return value
    end
    optimizer = ADAM()
    
    function evalcb() 
        a = evaluate(model, Xbs[1], Ybs[1])
        @info "loss(Xb[1]) = $(loss(dataset[1]...)), accuracy(Xb[1]) = $a"
    end
    @epochs options[:numEpochs] Flux.train!(loss, params(model), dataset, optimizer, cb = Flux.throttle(evalcb, 60))
    if (options[:gpu])
        model = model |> cpu
    end
    @save options[:modelPath] model
    # evaluate the training accuracy of the model
    pairs = [evaluate(model, collect(Xbs[i]), collect(Ybs[i])) for i=1:length(Xbs)]
    result = reduce(((a, b), (c, d)) -> (a + c, b + d), pairs)
    @info "training accuracy = $(result[2]/result[1]) [$(result[2])/$(result[1])]."
    return model    
end

end # module

