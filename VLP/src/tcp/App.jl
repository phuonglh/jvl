include("BinaryNode.jl")

s = BinaryNode("S-r")

np0 = BinaryNode("NP-r", s)
addLeftChild("the", np0)
updateLeftChild(np0, s)

np1 = BinaryNode("NP-r*", np0)
addLeftChild("little", np1)
addRightChild("boy", np1)
updateRightChild(np1, np0)

s0 = BinaryNode("S-l*", s)
updateRightChild(s0, s)
vp = BinaryNode("VP-l", s0)
addLeftChild("likes", vp)
addRightChild(".", s0)
updateLeftChild(vp, s0)

np2 = BinaryNode("NP-r", vp)
addLeftChild("red", np2)
addRightChild("potatoes", np2)
updateRightChild(np2, vp)
