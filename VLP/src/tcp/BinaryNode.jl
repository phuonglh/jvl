
using AbstractTrees

mutable struct BinaryNode{T}
  data::T
  parent::BinaryNode{T}
  left::BinaryNode{T}
  right::BinaryNode{T}

  # root constructor
  BinaryNode{T}(data) where T = new{T}(data)
  # child node constructor
  BinaryNode{T}(data, parent) where T = new{T}(data, parent)
end

"Short-cut constructors"
BinaryNode(data) = BinaryNode{typeof(data)}(data)
BinaryNode(data, parent) = BinaryNode{typeof(data)}(data, parent)

"Add left child with some data to a given node."
function addLeftChild(data, parent::BinaryNode)
  !isdefined(parent, :left) || error("Left child is already assigned!")
  node = typeof(parent)(data, parent)
  parent.left = node
end

"Update left child of a node."
function updateLeftChild(node::BinaryNode, parent::BinaryNode)
  !isdefined(parent, :left) || error("Left child is already assigned!")
  parent.left = node
end

"Add right child with some data to a given node."
function addRightChild(data, parent::BinaryNode)
  !isdefined(parent, :right) || error("Right child is already assigned!")
  node = typeof(parent)(data, parent)
  parent.right = node
end

"Update right child of a node."
function updateRightChild(node::BinaryNode, parent::BinaryNode)
  !isdefined(parent, :right) || error("Right child is already assigned!")
  parent.right = node
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

