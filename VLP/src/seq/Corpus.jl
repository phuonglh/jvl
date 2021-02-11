include("../tok/VietnameseTokenizer.jl")
include("Sentence.jl")

using .VietnameseTokenizer


"""
    readCorpusUD(path, maxSentenceLength=40)

    Read a CoNLLU file to build dependency graphs. Each graph is a sentence.
"""
function readCorpusUD(path::String, maxSentenceLength::Int=40)::Array{Sentence}
    lines = filter(line -> !startswith(line, "#"), readlines(path))
    append!(lines, [""])
    sentences = []
    tokens = []
    for line in lines
        parts = split(strip(line), r"\t+")
        if length(parts) == 1
            prepend!(tokens, [Token("ROOT", Dict(:id => "0", :lemma => "NA", :upos => "NA", :pos => "NA", :fs => "NA", :head => "NA", :label => "NA"))])
            # add sentence if it is not too long...
            if length(tokens) <= maxSentenceLength
              push!(sentences, Sentence(tokens))
            end
            empty!(tokens)
        else
            word = parts[2]
            fullLabel = parts[8]
            colonIndex = findfirst(':', fullLabel)
            label = if (colonIndex !== nothing) fullLabel[1:colonIndex-1] else fullLabel end
            annotation = Dict(:id => parts[1], :lemma => parts[3], :upos => parts[4], :pos => parts[5], :fs => parts[6], :head => parts[7], :label => label)
            push!(tokens, Token(word, annotation))
        end
    end
    sentences
end

"""
    readCorpusCoNLL(path, threeColumns::Bool=false, maxSentenceLength=40)

    Read a CoNLL-2003 file to build named-entity tagged sentences. The Bahasa Indonesia 
    corpus has 3 columns: word, part-of-speech, and NE tag; if reading this corpus, we 
    need to set the last argument to true.
"""
function readCorpusCoNLL(path::String, threeColumns::Bool=false, maxSentenceLength::Int=40)::Array{Sentence}
    function createToken(line::String)::Token
        parts = string.(split(line, r"[\s]+"))
        annotation = if threeColumns 
            Dict(:p => parts[2], :c => "_", :e => parts[3], :s => VietnameseTokenizer.shape(parts[1]))
        else
            Dict(:p => parts[2], :c => parts[3], :e => parts[4], :s => VietnameseTokenizer.shape(parts[1]))
        end
        Token(parts[1], annotation)
    end

    sentences = []
    lines = readlines(path)
    n = length(lines)
    indexedLines = collect(zip(1:n, map(line -> strip(line), lines)))
    emptyIndices = map(p -> p[1], filter(p -> isempty(p[2]), indexedLines))
    j = 1
    for i in emptyIndices
        xs = lines[j:i-1]
        if (isempty(xs))
            @warn ("Problematic line: $i")
        end
        tokens = createToken.(xs)
        push!(sentences, Sentence(tokens))
        j = i+1
    end
    sentences
end

"""
    readCorpusVLSP(path, maxSentenceLength)

    Read VLSP-2010 corpus for part-of-speech tagging
"""
function readCorpusVLSP(path::String, maxSentenceLength::Int=40)::Array{Sentence}
    sentences = Sentence[]
    lines = readlines(path)
    for line in lines
        wts = string.(split(strip(line), r"[\s]+"))
        tokens = Token[]
        for wt in wts
            annotation = Dict{Symbol,String}()
            parts = string.(split(wt, r"/"))
            if (length(parts) == 2)
                annotation[:pos] = parts[2]
                token = Token(parts[1], annotation)
                push!(tokens, token)
            else
                j = findlast("/", wt)[1]
                if (j == length(wt)) # the case ///
                    annotation[:pos] = "/"
                    push!(tokens, Token("/", annotation))
                else
                    w = wt[1:j-1]
                    annotation[:pos] = wt[j+1:end]
                    push!(tokens, Token(w, annotation))
                end
            end
            annotation[:upos] = "NA"
        end
        n = min(maxSentenceLength, length(tokens))
        push!(sentences, Sentence(tokens[1:n]))
    end
    # filter sentences...
    filter(sentence -> length(sentence.tokens) <= maxSentenceLength, sentences)
end
