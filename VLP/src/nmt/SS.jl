using Flux
using Flux: @epochs
using BSON: @save, @load


include("Embedding.jl")


options = Dict{Symbol,Any}(
    :minFreq => 2,
    :maxSequenceLength => 40,
    :batchSize => 16,
    :delimiters => r"""[\s.,?"]+""",
    :corpusPath => string(pwd(), "/dat/nmt/1000.txt"),
    :sourcePath => string(pwd(), "/dat/nmt/src.txt"),
    :targetPath => string(pwd(), "/dat/nmt/tar.txt"),
    :inputSize => 32,
    :outputSize => 32,
    :hiddenSize => 16,
    :PAD => "[PAD]",
    :EOS => "[EOS]",
    :UNK => "[UNK]",
    :gpu => false
)

function readCorpus(options)::Array{Tuple{String,String}}
    f(line) = begin
        parts = strip.(string.(split(line, r"\t+")))
        if length(parts) == 2
            (parts[1], parts[2])
        else
            ("", "")
        end
    end
    lines = readlines(options[:corpusPath])
    selectedLines = filter(line -> length(line) > 40, lines)
    map(line -> f(line), selectedLines)
end

"""
    vocab(sentences, options)

    Builds vocabularies of words, shapes, parts-of-speech, and labels. The word vocabulary is sorted by frequency.
    Only words whose count is greater than `minFreq` are kept.
"""
function vocab(sentences::Array{String}, options)::Dict{String,Int}
    tokens = Iterators.flatten(map(sentence -> string.(split(sentence, options[:delimiters])), sentences))
    wordFrequency = Dict{String, Int}()
    for token in tokens
        word = lowercase(strip(token))
        haskey(wordFrequency, word) ? wordFrequency[word] += 1 : wordFrequency[word] = 1
    end
    # filter out infrequent words
    filter!(p -> p.second >= options[:minFreq], wordFrequency)
    wordFrequency
end

function batch(pairs::Array{Tuple{String,String}}, sourceDict, targetDict, options)
    Xs, Ys = Array{Array{Int},1}(), Array{Array{Int},1}()
    b = options[:batchSize]
    m = options[:maxSequenceLength]
    for pair in pairs
        srcWords = lowercase.(string.(split(pair[1], options[:delimiters])))
        tarWords = lowercase.(string.(split(pair[2], options[:delimiters])))
        X = map(word -> get(sourceDict, word, sourceDict[options[:UNK]]), srcWords)
        Y = map(word -> get(targetDict, word, targetDict[options[:UNK]]), tarWords)
        if (length(X) <= m && length(Y) <= m) 
            xs = Flux.rpad(X, b, sourceDict[options[:PAD]])
            ys = Flux.rpad(Y, b, targetDict[options[:PAD]])
            push!(Xs, xs)
            push!(Ys, ys)
        end
    end
    Xs, Ys
end

function train(options)
    pairs = readCorpus(options)
    sourceSentences = map(pair -> pair[1], pairs)
    targetSentences = map(pair -> pair[2], pairs)
    sourceDict = vocab(sourceSentences, options)
    targetDict = vocab(targetSentences, options)
    m, n = length(sourceDict), length(targetDict)
    sourceDict[options[:PAD]] = m+1; sourceDict[options[:UNK]] = m+2; sourceDict[options[:PAD]] = m+3
    targetDict[options[:PAD]] = n+1; targetDict[options[:UNK]] = n+2; targetDict[options[:PAD]] = n+3

    sourceEmbedding = Embedding(length(sourceDict), options[:hiddenSize])
    targetEmbedding = Embedding(length(targetDict), options[:hiddenSize])
end

