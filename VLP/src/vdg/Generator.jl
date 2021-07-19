# Vietnamese diacritics generation
# (C) phuonglh@gmail.com, 2021

module VDG

using FLoops: Iterators
using CSV
using DataFrames
using Flux
using Flux: @epochs
using BSON: @save, @load

using FLoops


include("DiacriticsRemoval.jl")


options = Dict{Symbol,Any}(
    :dataPath => string(pwd(), "/dat/qas/monre-100.csv"),
    :labelPath => string(pwd(), "/dat/qas/charIndex.txt"),
    :maxSequenceLength => 64,
    :batchSize => 16,
    :unkChar => 'X',
    :padChar => 'P'
)

"""
  vectorize(text, alphabet, training=true)

  Transforms a sentence into an array of vectors based on an alphabet and 
  a label set for training. Return a pair of matrix `(x, y)` where
  `x` is a matrix of size `|alphabet| x maxSequenceLength`.
"""
function vectorize(text::String, alphabet::Array{Char}, training::Bool=true)
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
    Xs = map(x -> Flux.onehotbatch(x, alphabet), xs)
    if (training)
        ys = collect(Iterators.partition(text, n))
        ys[end] = vcat(ys[end], ps)
        Ys = map(y -> Flux.onehotbatch(y, alphabet), ys)
        return (Xs, Ys)
    else
        return Xs
    end
end

function train(options)
    # read input data frame and create inp/out sequences
    df = CSV.File(options[:dataPath]) |> DataFrame
    ys = map(y -> lowercase(y), df[:, :question])
    xs = map(y -> removeDiacritics(y), ys)
    # create alphabet and char index
    alphabetY = unique(join(ys))
    alphabet = unique(join(alphabetY, values(charMap)))
    prepend!(alphabet, options[:unkChar])
    prepend!(alphabet, options[:padChar])
    charIndex = Dict{Char,Int}(c => i for (i, c) in enumerate(alphabet))
    # write the char index to an external file (for use in prediction)
    # file = open(options[:labelPath], "w")
    # for f in keys(charIndex)
    #     write(file, string(f, " ", charIndex[f]), "\n")
    # end
    # close(file)
    # create training sequences
    XYs = map(x -> vectorize(x, alphabet), xs)
    Xs = collect(Iterators.flatten(map(xy -> xy[1], XYs)))
    Ys = collect(Iterators.flatten(map(xy -> xy[2], XYs)))
    # batch inp/out sequences
    Xb = collect(Iterators.partition(Xs, options[:batchSize]))
    Yb = collect(Iterators.partition(Ys, options[:batchSize]))
    # create batches of 3-d matrices of size (|alphabet| x maxSeqLen x batchSize)
    Xbs = map(A -> Flux.batch(A), Xb)
    Ybs = map(A -> Flux.batch(A), Yb)
    return (Xbs, Ybs)
end

end # module


