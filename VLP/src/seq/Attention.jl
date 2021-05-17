using Flux

struct Attention
    W
end

Attention(d::Int) = Attention(rand(d, d))

(a:Attention)(h, s) = h' * a.W * s

Flux.@functor Attention