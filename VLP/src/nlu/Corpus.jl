# phuonglh@gmail.com

module Corpus

export read

using DataFrames
using JSON3


"""
    read(path)

    Reads a .txt or .csv or .json file and build a data frame for the intent detection module.
    The data frame should have two columns :intent and :text. The JSON data format should be similar 
    to the `accounts.json` sample file.
"""
function readIntents(path::String)::DataFrame
    if endswith(path, ".txt") || endswith(path, ".csv")
        df = DataFrame(CSV.File(path))
        dropmissing!(df)
    else
        if (endswith(path, ".json"))
            json_st = read(path, String)
            samples = JSON3.read(json_st)[:nlu]
            intents = Array{String,1}()
            texts = Array{String,1}()
            for sample in samples
                i = sample[:intent]
                ts = sample[:examples]
                for t in ts
                    push!(intents, i)
                    # remove values in t before adding it to the array
                    push!(texts, replace(t, r"\[[\w\s]+\]" => ""))
                end
            end
            df = DataFrame(:intent => intents, :text => texts)
        else
            DataFrame(:intent => [], :text => [])
        end
    end
end

"""
    extractEntities(st)

    Extracts an array of entities from a given string. If the input string contains type/value pairs such as 
    "chuyển [2 triệu](amount) sang tài khoản [00220712](destination)"
    then the function extracts the following list: [("2 triệu", "amount"), ("00220712", "destination")].
"""
function extractEntities(st::String)::Array{Tuple{String,String},1}
    pattern = r"(?<value>\[[\w\s]+\])(?<type>\(\w+\))"
    idx = 1
    entities = Array{Tuple{String,String},1}()
    while idx <= length(st)
        m = match(pattern, st, idx)
        if m === nothing
            break
        else
            push!(entities, (m[:value][2:end-1], m[:type][2:end-1]))
            idx = m.offset + length(m.match)
        end
    end
    return entities
end

end # module