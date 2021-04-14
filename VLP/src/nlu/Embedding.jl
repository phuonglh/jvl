#=
    Build an embedding layer for NLP.
    phuonglh@gmail.com
    November 12, 2019, updated on December 9, 2020
=#

using Flux

# Each embedding layer contains a matrix of all word vectors, 
# each column is the vector of the corresponding word index.
struct Embedding
    W
end

Embedding(inp::Int, out::Int) = Embedding(rand(Float32, out, inp))

# overload call, so the object can be used as a function
# x is a word index (1 <= x < vocabSize)
(f::Embedding)(x) = f.W[:,x]

# make the embedding layer trainable
Flux.@functor Embedding

# add length functions for our types
# 
Base.length(e::Embedding) = length(e.W)
