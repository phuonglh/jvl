using Random

struct Vocabularies
    words::Array{String}
    shapes::Array{String}
    partsOfSpeech::Array{String}
    labels::Array{String}
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

"""
    loadIndex(path)

    Load an index from a file which is previously saved by `saveIndex()` function.
"""
function loadIndex(path)::Dict{String,Int}
    lines = readlines(path)
    pairs = Array{Tuple{String,Int},1}()
    for line in lines
        j = findlast(' ', line)
        i = parse(Int, line[j+1:end])
        w = line[1:prevind(line,j)]
        push!(pairs, (w, i))
    end
    return Dict(pair[1] => pair[2] for pair in pairs)
end

"""
    shuffleSentences(sentences)

    Shuffle an array of sentences randomly.
"""
function shuffleSentences(sentences::Array{Sentence})::Array{Sentence}
    rng = MersenneTwister(220712)
    shuffle!(rng, sentences)
end

"""
    splitSentences(sentences, ratios)

    Split an array of sentences into three parts according to given ratios. This 
    function is helpful for train./dev./test partitionning.
"""
function splitSentences(sentences::Array{Sentence}, ratios=[0.70, 0.15, 0.15])
    m = Int(round(length(sentences)*ratios[1]))
    first = sentences[1:m]
    n = Int(round(length(sentences)*(ratios[1] + ratios[2])))
    second = sentences[m+1:n]
    third = sentences[n+1:end]
    return (first, second, third)
end