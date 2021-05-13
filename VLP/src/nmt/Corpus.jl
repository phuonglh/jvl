# phuonglh
# 

function readCorpusWithTab(options)::Array{Tuple{String,String}}
    f(line) = begin
        parts = strip.(string.(split(line, r"\t+")))
        if length(parts) == 2
            (parts[1], parts[2])
        else
            ("", "")
        end
    end
    lines = readlines(options[:corpusPath])
    selectedLines = filter(line -> length(line) > 40, lines)
    map(line -> f(line), selectedLines)
end

function readCorpusEuroparl(options)::Array{Tuple{String,String}}
    sourceLines = readlines(options[:sourceCorpusPath])
    targetLines = readlines(options[:targetCorpusPath])
    collect(zip(sourceLines, targetLines))
end