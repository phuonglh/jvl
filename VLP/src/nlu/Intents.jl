# Intent Detection module in Julia
# We use the dataset at https://github.com/xliuhw/NLU-Evaluation-Data
# phuonglh@gmail.com

using CSV
using DataFrames

function readCorpus(path::String)::DataFrame
    df = DataFrame(CSV.File(path))
    ef = select(df, :intent => :intent, :answer => :text)
    dropmissing!(ef)
end

path = "dat/nlu/xliuhw/AnnotatedData/NLU-Data-Home-Domain-Annotated-All.csv"
df = readCorpus(path)
intents = unique(df[:, :intent])