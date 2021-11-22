if !isdefined(@__MODULE__, :BinaryNode)
  include("./BinaryNode.jl")
end

## Things we need to define to get children of a node.
function AbstractTrees.children(node::BinaryNode)
  if isdefined(node, :left)
      if isdefined(node, :right)
          return (node.left, node.right)
      end
      return (node.left,)
  end
  isdefined(node, :right) && return (node.right,)
  return ()
end

## Things that make printing prettier
AbstractTrees.printnode(io::IO, node::BinaryNode) = print(io, node.data)

## Optional enhancements
# These next two definitions allow inference of the item type in iteration.
# (They are not sufficient to solve all internal inference issues, however.)
Base.eltype(::Type{<:TreeIterator{BinaryNode{T}}}) where T = BinaryNode{T}
Base.IteratorEltype(::Type{<:TreeIterator{BinaryNode{T}}}) where T = Base.HasEltype()

