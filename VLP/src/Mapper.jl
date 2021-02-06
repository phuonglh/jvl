module Mapper

using ..Model

const STORE = Dict{Int, Array{Int}}()
const COUNTER = Ref{Int64}(0)

function store!(primes)
    id = COUNTER[] += 1
    STORE[id] = primes
end


end # module