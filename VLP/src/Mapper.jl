module Mapper

using ..Model

const STORE = Dict{(Int,Symbol),Any}()
const COUNTER = Ref{Int64}(0)

function store!(task, result)
    id = COUNTER[] += 1
    STORE[(id,task)] = result
end


end # module