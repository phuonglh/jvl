module Corpus

using DataFrames
using JSON3


labelMap = Dict("neutral" => 1, "contradiction" => 2, "entailment" => 3)

"""
    read(path, jsonl=true)

    Reads a XNLI-format JSON or JSONL file and return a data frame with 3 columns (:sentence1, :sentence2, :label)
"""
function readXNLI(path::String, jsonl::Bool=true)::DataFrame
    objects = if jsonl
        lines = readlines(path)
        map(line -> JSON3.read(line), lines)
    else
        content = read(path, String)
        JSON3.read(content)
    end
    s1 = map(object -> object.sentence1, objects)
    s2 = map(object -> object.sentence2, objects)
    ls = map(object -> labelMap[object.gold_label], objects)
    DataFrame(:sentence1 => s1, :sentence2 => s2, :label => ls)
end

end # module