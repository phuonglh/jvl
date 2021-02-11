struct Token
    word::String
    annotation::Dict{Symbol,String}
end

struct Sentence
    tokens::Array{Token}
end

