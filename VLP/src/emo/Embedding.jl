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

# add length functions for our types
# 
Base.length(e::Embedding) = length(e.W)


## Join two layers. The first layer is an embedding layer. 

struct Join
    first
    second
end

Join(fs...) = Join(fs[1], fs[2])

"""
    a: token ids
    b: real-valued features
"""
function (g::Join)(x::Tuple{Vector{Int},Vector{Float32}})
    a, b = x
    as = g.first(a) # token embeddings
    u = vec(sum(as, dims=2)) 
    v = g.second(b)
    vcat(u, v)
end

function (g::Join)(x::SubArray)
    g(x[1])
end

# batch application
function (g::Join)(xs::Vector{Tuple{Vector{Int},Vector{Float32}}})
    zs = [g(x) for x ∈ xs]
    hcat(zs...)
end

Flux.@functor Join

# JoinR
# Apply a RNN to a sequence of token ids and then extract sequence presentation
# before joining them.

struct JoinR
    first
    second
    rnn
end

JoinR(fs...) = JoinR(fs[1], fs[2])

"""
    a: token ids
    b: real-valued features
"""
function (g::JoinR)(x::Tuple{Vector{Int},Vector{Float32}})
    a, b = x
    as = g.first(a) # token embeddings
    Flux.reset!(rnn) # reset for next input
    u = g.rnn(as)[:,end] # apply RNN and extract the last state
    v = g.second(b)
    vcat(u, v)
end

function (g::JoinR)(x::SubArray)
    g(x[1])
end

# batch application
function (g::JoinR)(xs::Vector{Tuple{Vector{Int},Vector{Float32}}})
    zs = [g(x) for x ∈ xs]
    hcat(zs...)
end

Flux.@functor JoinR

