#=
    Build an embedding layer for NLP.
    phuonglh@gmail.com
    November 12, 2019, updated on December 9, 2020
=#

using Flux
using CUDA

# Each embedding layer contains a matrix of all word vectors, 
# each column is the vector of the corresponding word index.
struct Embedding
    W
end

Embedding(inp::Int, out::Int) = Embedding(rand(Float32, out, inp))

# overload call, so the object can be used as a function
# x is a word index (1 <= x < vocabSize) or a vector of word indices
(f::Embedding)(x::Int) = f.W[:,x]

# x is a vector of word indices
(f::Embedding)(x::Array{Int,1}) = f.W[:,x]
(f::Embedding)(x::CuArray{Int,1}) = f.W[:,x]

# x is a matrix of word indices
(f::Embedding)(x::Array{Int,2}) = f.W[:,x]
(f::Embedding)(x::CuArray{Int,2}) = f.W[:,x]


# make the embedding layer trainable
Flux.@functor Embedding

# add length function for our types
# 
Base.length(e::Embedding) = length(e.W)

# overload print
Base.show(io::IO, e::Embedding) = print(io, "Embedding$(size(e.W))")