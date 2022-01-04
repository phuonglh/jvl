if !isdefined(@__MODULE__, :BinaryNode)
    include("./BinaryNode.jl")
end


function isSibling(left::BinaryNode, right::BinaryNode)
    return left.head == right.head
end

function decode(tree:: BinaryNode)
end