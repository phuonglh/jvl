using Random

"""
    saveIndex(index, path)
    
    Save an index to an external file.
"""
function saveIndex(index, path)
    file = open(path, "w")
    for f in keys(index)
        write(file, string(f, " ", index[f]), "\n")
    end
    close(file)
end

"""
    loadIndex(path)

    Load an index from a file which is previously saved by `saveIndex()` function.
"""
function loadIndex(path)::Dict{String,Int}
    lines = readlines(path)
    pairs = Array{Tuple{String,Int},1}()
    for line in lines
        j = findlast(' ', line)
        i = parse(Int, line[j+1:end])
        w = line[1:prevind(line,j)]
        push!(pairs, (w, i))
    end
    return Dict(pair[1] => pair[2] for pair in pairs)
end

