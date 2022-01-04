using BSON: @load
using DataFrames
using CSV

include("Intent.jl")
include("Embedding.jl")
include("GRU3.jl")
include("Utils.jl")
include("Corpus.jl")


mode="train"

options = Intent.options
if mode == "train"
    encoder, accuracy = Intent.train(options)
else
    if mode == "eval"
        @load options[:modelPath] encoder
        wordIndex = loadIndex(options[:wordPath])
        labelIndex = loadIndex(options[:labelPath])
        df = Corpus.readIntents(options[:corpusPath])
        Xs, Ys = Intent.batch(df, wordIndex, labelIndex, options)
        accuracy = Intent.evaluate(encoder, Xs, Ys, options)
    end
end
