# phuonglh@gmail.com
# A user-defined layer to accept multiple inputs and output a 
# concatenated vector of some selected vectors. The input is fed in
# as a tuple of 2 components: one token sequence for embedding path and 
# one token sequence for RNN path

using Flux

struct Join
    fs # functions (typically two layers: [EmbeddingWSP, RNN])
end

Join(fs...) = Join(fs)

"""
    a: token id matrix of size (3 x sentenceLength), each column contains 3 ids for (word, shape, tag)
    b: token position vector which corresponds to a parsing configuration (4-element vector)
"""
function (g::Join)(x::Tuple{Array{Int,2},Array{Int,1}})
    a, b = x
    as = g.fs[1](a) # matrix of size (e_w + e_s + e_p) x sentenceLength
    u = g.fs[2](as) # if `fs[2]` is a RNN and `as` is an index array, this gives a matrix of size out x sentenceLength
    vec(u[:, b])  # if `b` is an index array, this gives a concatenated vector of length |b| * out
end

function (g::Join)(x::SubArray)
    g(x[1])
end

"""
    a: token id matrix of size (3 x sentenceLength), each column contains 3 ids for (word, shape, tag)
    b: token position matrix of size 4 x k, in which each column corresponds to a parsing configuration
"""
function (g::Join)(x::Tuple{Array{Int,2},Array{Int,2}})
    a, b = x
    as = g.fs[1](a)
    u = g.fs[2](as) # if `fs[2]` is a RNN and `as` is an index array, this gives a matrix u
    vs = [vec(u[:, b[:,j]]) for j=1:size(b,2)] # apply for each column in b
    hcat(vs...) # stack vs to get the output matrix instead of an array of arrays
end


Flux.@functor Join


## phuonglh@gmail.com
##
## A user-defined layer for concatenating the output of the Join layer with pre-trained 
## TransE embeddings. TransE embedding matrix should be pre-loaded -- they are not trained 
# by the back-propagation algorithm.

struct PretrainedEmbedding
    W
end

# overload call, so the object can be used as a function
# x is a word index (1 <= x < vocabSize) 
(f::PretrainedEmbedding)(x::Int) = vcat(f.W[:,x])

# x is a column vector of word indices, the output should be concatenated to 
# produce a column vector 
(f::PretrainedEmbedding)(x::Array{Int}) = vcat([f.W[:,x[j]] for j=1:length(x)]...)

struct ConcatLayer
    join::Join
    embedding::PretrainedEmbedding
end

ConcatLayer(fs...) = ConcatLayer(fs)

"""
    Input is a tuple of three elements `(a, b, c)`. The pair `(a, b)` will be passed 
    into the Join layer. The matrix `c` will be treated by the embedding layer. Each 
    column of `c` contains 4 word indices of the current parsing config.
    This method is used in training where a batch is processed.
"""
function (f::ConcatLayer)(x::Tuple{Array{Int,2},Array{Int,2},Array{Int,2}})
    a, b, c = x
    u = f.join((a, b))
    v = hcat([f.embedding(c[:,j]) for j=1:size(c,2)]...)
    vcat(u, v)
end

"""
    This method is used in decoding each configuration (testing phase).
"""
function (f::ConcatLayer)(x::Tuple{Array{Int,2},Array{Int,1},Array{Int,1}})
    a, b, c = x
    u = f.join((a, b)) # a column vector 
    v = f.embedding(c) # a column vector
    vcat(u, v)
end

Flux.@functor ConcatLayer