include("BinaryNode.jl")
include("Token.jl")

function tree1()
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
    return s
end

function tree2()
    the = BinaryNode(Token(1, "the"))
    little = BinaryNode(Token(2, "little"))
    boy = BinaryNode(Token(3, "boy"))
    np1 = BinaryNode(Token(100, "NP-R", 3))
    updateLeftChild(little, np1)
    updateRightChild(boy, np1)
    np2 = BinaryNode(Token(101, "NP-R", 3))
    updateLeftChild(the, np2)
    updateRightChild(np1, np2)

    likes = BinaryNode(Token(4, "likes"))
    red = BinaryNode(Token(5, "red"))
    potatoes = BinaryNode(Token(6, "potatoes"))
    point = BinaryNode(Token(7, "."))
    vp = BinaryNode(Token(102, "VP-R"))
    np3 = BinaryNode(Token(103, "NP-R", 6))
    updateLeftChild(red, np3)
    updateRightChild(potatoes, np3)
    updateLeftChild(likes, vp)
    updateRightChild(np3, vp)

    s1 = BinaryNode(Token(104, "S-L"))
    updateLeftChild(vp, s1)
    updateRightChild(point, s1)
    
    s = BinaryNode(Token(105, "S-R"))
    updateLeftChild(np2, s)
    updateRightChild(s1, s)
    return s
end
