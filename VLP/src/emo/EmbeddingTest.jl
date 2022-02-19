# Test the JoinR layer

using Flux
include("Embedding.jl")

first = Embedding(10, 4)
second = identity
rnn = GRU(4, 2)

layer = JoinR(first, second, rnn)

# single application of the layer
a = [1, 3, 5, 7, 9]
b = Float32.([0.1, 0.2, 0.3])
x = (a, b)
y = layer(x) # y should have 5 dimensions

# batch application of the layer, note that the b vectors 
# in the batch should have the same length for the hcat function to work
a2 = [2, 4, 6]
b2 = Float32.([0.4, 0.5, 0.6])
x2 = (a2, b2)

xs = [(a, b), (a2, b2)]
ys = layer(xs)


