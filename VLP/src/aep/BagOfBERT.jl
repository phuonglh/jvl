# phuonglh@gmail.com
# December 2021, add an additional experiment with BERT embeddings


struct BagOfBERT
    fs # [EmbeddingWSP, identity, bertify]
end

BagOfBERT(fs...) = BagOfBERT(fs)
function (g::BagOfBERT)(x::SubArray)
    g(x[1])
end

"""
    a: token id matrix of size (3 x sentenceLength), each column contains 3 ids for (word, shape, tag)
    b: token position matrix of size 4 x m, in which each column corresponds to a parsing configuration
    c: a BERT embeddings matrix for a given sentence
    This layer is used in the `ClassifierBERT.jl`.
"""
function (g::BagOfBERT)(x::Tuple{Matrix{Int},Matrix{Int},Matrix{Float32}})
    a, B, c = x
    us = g.fs[1](a) # embedding matrix from word, shape and part-of-speech tags
    m = size(b,2)
    b = g.fs[2](B) # identity
    # token embeddings
    as = [vec(us[:, b[:,j]]) for j=1:m] # apply for each column in b
    α = hcat(as...)
    bert = g.fs[3](c) # identity
    # in each column, we compute sum of 4 BERT vectors instead of concatenating (with function vec)
    # to reduce the dimensionality
    vs = [vec(sum(bert[:, b[:,j]], dims=2)) for j=1:m] # apply for each column in b
    β = hcat(vs...) # stack vs to get the output matrix instead of an array of arrays
    @assert size(α, 2) == size(β, 2) # should be equal to m
    # stack α and β matrices
    return vcat(α, β)
end

Flux.@functor BagOfBERT
