using DataStructures
include("BinaryNode.jl")

struct Config
    stack::Stack{BinaryNode}
    queue::Queue{Int}
    words::Vector{String}
end

struct Context
    features::Array{String}
    transition::String
end

# phuonglh: A simple hack function to copy a stack. The library DataStructures does not provide 
# this function by default (Feb., 2021)
Base.copy(s::Stack) = begin
    elements = collect(s.store)
    s2 = Stack{eltype(s)}()
    for e in elements
        push!(s2, e)
    end
    return s2
end

# phuonglh: Another simple hack function to copy a queue. The library DataStructures does not provide 
# this function by default (Feb., 2021)
Base.copy(q::Queue) = begin
    elements = collect(q.store)
    q2 = Queue{eltype(q)}()
    for e in elements
        enqueue!(q2, e)
    end
    return q2
end

function next(config::Config, transition::String)::Config
    if transition == "SHIFT"
        # create a node (w_i)
        i = dequeue!(config.queue)
        w = BinaryNode(config.words[i])
        push!(config.stack, w)
    elseif startswith(transition, "UNARY")
        j = findfirst('-', transition)
        X = transition[j+1:end]
        s0 = pop!(config.stack)
        x = BinaryNode(X)
        updateLeftChild(s0, x)
        push!(config.stack, x)
    elseif startswith(transition, "REDUCE")
        j = findlast('-', transition)
        X = transition[j+1:end]
        s0 = pop!(config.stack)
        s1 = pop!(config.stack)
        x = BinaryNode(X)
        if occursin("-L", transition)
            updateLeftChild(s0, x)
            updateRightChild(s1, x)
        else
            updateLeftChild(s1, x)
            updateRightChild(s0, x)
        end
        push!(config.stack, x)
    end
    return Config(copy(config.stack), copy(config.queue), config.words)
end

