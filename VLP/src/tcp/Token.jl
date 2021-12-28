struct Token
    id::Int
    label::String
    head::Int
    metadata
end

Token(id, label) = Token(id, label, -1, Dict())
Token(id, label, head) = Token(id, label, head, Dict())