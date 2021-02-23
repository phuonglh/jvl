#=
    Build an embedding layer for NLP.
    phuonglh@gmail.com
    November 12, 2019, updated on December 9, 2020
=#

using Flux

# Each embedding layer contains a matrix of all word vectors, 
# each column is the vector of the corresponding word index.
struct EmbeddingConcat
    W
end

EmbeddingConcat(inp::Int, out::Int) = EmbeddingConcat(rand(Float32, out, inp))

# overload call, so the object can be used as a function
# x is a word index or an array, or a matrix
(f::EmbeddingConcat)(x) = begin
    # if x is a matrix, A is then a 3-d tensor
    A = f.W[:, x] 
    # concatenate all columns of matrix A[:, :, i]
    hcat((vcat(A[:,:,i]...) for i=1:size(x,2))...)
end

# make the embedding layer trainable
Flux.@functor EmbeddingConcat

