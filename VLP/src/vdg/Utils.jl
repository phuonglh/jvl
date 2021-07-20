"""
    loadIndex(path)

    Load an index from a file which is previously saved by `saveIndex()` function.
"""
function loadIndex(path)::Dict{Char,Int}
    lines = readlines(path)
    pairs = Array{Tuple{Char,Int},1}()
    for line in lines
        j = findlast(' ', line)
        i = parse(Int, line[j+1:end])
        c = line[1:prevind(line,j)][1]
        push!(pairs, (c, i))
    end
    return Dict(pair[1] => pair[2] for pair in pairs)
end


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
    loadAlphabet(path)
"""
function loadAlphabet(path)::Array{Char}
    charIndex = loadIndex(path)
    pairs = map(c -> (charIndex[c], c), collect(keys(charIndex)))
    sort!(pairs, by = pair -> pair[1])
    return map(p -> p[2], pairs)
end
