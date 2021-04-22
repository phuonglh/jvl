# phuonglh
# A modified version of GRU to take 3-d matrix as input. 
# This is useful for batch processing.

using Flux
using CUDA

struct GRU3
    gru
end

GRU3(inp::Int, out::Int) = GRU3(GRU(inp, out))

# overload call, so the object can be used as a function
(f::GRU3)(x::Array{Float32,3}) = begin
    m = size(x)[3]
    # xs = map(i -> f.gru(x[:,:,i])[:,end], collect(1:m)) # take the last output state
    xs = map(i -> sum(f.gru(x[:,:,i]), dims=2), collect(1:m)) # take the bag-of-embeddings
    hcat(xs...)
end
(f::GRU3)(x::CuArray{Float32,3}) = begin
    m = size(x)[3]
    # xs = map(i -> f.gru(x[:,:,i])[:,end], collect(1:m)) # take the last output state
    xs = map(i -> sum(f.gru(x[:,:,i]), dims=2), collect(1:m)) # take the bag-of-embeddings
    hcat(xs...)
end

# make the embedding layer trainable
Flux.@functor GRU3
