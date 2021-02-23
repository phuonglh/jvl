#=
    Build an embedding layer for NLP.
    phuonglh@gmail.com
    November 12, 2019, updated on December 9, 2020
=#

using Flux
using Flux: @adjoint

# Each embedding layer contains a matrix of all word vectors, 
# each column is the vector of the corresponding word index.
struct EmbeddingConcat
    W
end

EmbeddingConcat(inp::Int, out::Int) = EmbeddingConcat(rand(Float32, out, inp))

# overload call, so the object can be used as a function
# x is a word index or an array, or a matrix
(f::EmbeddingConcat)(x) = hcat([vcat([f.W[:,x[i,t]] for i=1:size(x,1)]...) for t=1:size(x,2)]...)

# A = f.W[:,x]
# bs = [vcat(A[:,:,i]...) for i=1:size(A,3)]
# hcat(bs...)
# overload the EmbeddingConcat constructor for back-propagation to work
# @adjoint EmbeddingConcat(W) = EmbeddingConcat(W), df -> (df.W,)

# make the embedding layer trainable
Flux.@functor EmbeddingConcat

