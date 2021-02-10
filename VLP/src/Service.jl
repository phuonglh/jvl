module Service

using ..Model, ..Mapper

using ..VietnameseTokenizer
using ..PoSTagger

function listPrimes(obj)::Array{Int}
    @info obj
    @assert haskey(obj, :u) && !isempty(obj.u)
    @assert haskey(obj, :v) && !isempty(obj.v)
    primes = Array{Int,1}()
    for k=max(2,obj.u):obj.v
        p = true
        for j = 2:sqrt(k)
            if mod(k, j) == 0
                p = false
                break
            end
        end
        if p push!(primes, k); end
    end
    return primes
end

"""
    tokenize(obj)

    Tokenize a text in a JSON object and return the result.
"""
function tokenize(obj)::Analysis
    @info obj
    @assert haskey(obj, :text) && !isempty(obj.text)
    tokens = VietnameseTokenizer.tokenize(obj.text)
    words = join(map(token -> token.text, tokens), " ")
    analysis = Analysis(obj.text, words, "NA")
    Mapper.store!(:tok, words)
    return analysis
end

"""
    tag(obj)

    Tag a text in a JSON object and return the result.
"""
function tag(obj)
    @info obj
    @assert haskey(obj, :text) && !isempty(obj.text)
    tokens = VietnameseTokenizer.tokenize(obj.text)
    words = map(token -> replace(token.text, " " => "_"), tokens)
    ts = map(word -> PoSTagger.Token(word, Dict(:upos => "NA")), words)
    @info ts
    sentence = PoSTagger.Sentence(ts)
    tags = PoSTagger.run(Model.encoderPoS, [sentence], Model.options, Model.wordIndexPoS, Model.shapeIndexPoS, Model.posIndexPoS, Model.labelIndexPoS)
    pairs = collect(zip(words, tags[1]))
    xs = join(pairs, ", ")
    analysis = Analysis(obj.text, xs, "NA")
    Mapper.store!(:tag, pairs)
    return analysis
end

end # module