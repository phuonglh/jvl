module Mapper

using ..Model

const STORE = Dict{Symbol,Any}()

function store!(task, result)
    @info task
    @info result
    STORE[task] = result
end


end # module