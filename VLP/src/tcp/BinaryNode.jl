
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

BinaryNode(data) = BinaryNode{typeof(data)}(data)

"Add left child with some data to a parent."
function addLeftChild(data, parent::BinaryNode)
  !isdefined(parent, :left) || error("Left child is already assigned!")
  node = typeof(parent)(data, parent)
  parent.left = node
end

function updateLeftChild(node::BinaryNode, parent::BinaryNode)
  !isdefined(parent, :left) || error("Left child is already assigned!")
  parent.left = node
end

"Add right child with some data to a parent."
function addRightChild(data, parent::BinaryNode)
  !isdefined(parent, :right) || error("Right child is already assigned!")
  node = typeof(parent)(data, parent)
  parent.right = node
end

function updateRightChild(node::BinaryNode, parent::BinaryNode)
  !isdefined(parent, :right) || error("Right child is already assigned!")
  parent.right = node
end
