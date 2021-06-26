# phuonglh@gmail.com
# June 2021

using StructTypes
using JSON3
using DataFrames

function loadJsonFile(path::String, source::String="MONRE")
    lines = readlines(path)
    content = join(lines)
    data = JSON3.read(content)[:data]
    id = map(d -> d[:idSqlManage], data)
    question = map(d -> strip(replace(get(d, :question, ""), r"[\n\r]+" => " ")), data)
    domain = map(d -> get(d, :linhVuc, ""), data)
    keyword = map(d -> get(d, :questionKeyword, Array{String,1}()), data)
    df = DataFrame(:id => id, :question => question, :domain => domain, :keyword => keyword)
    return df[df.question .!= "", :]
end

function loadDirectory(directory::String, source::String="MONRE")
    paths = map(path -> string(directory, "/", path, ".json"), ["01", "02", "03", "04"])
    dfs = map(path -> loadJsonFile(path, source), paths)
    vcat(dfs...)
end

function describe(df)
    select(qf, :question => ByRow(length))
end