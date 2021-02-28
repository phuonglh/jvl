# phuonglh
# Analyze processing results (tagging, parsing, etc.)
# Results in JSON files are loaded in to data frames, then they are analyzed.

using JSON3
using JSONTables
using DataFrames

"""
    aep(language)

    Read experimental results of the AEP module into a data frame.
"""
function aep(language::String="vie")::DataFrame
    path = string("dat/aep/", language, "-score.jsonl")
    xs = readlines(path)
    s = string("[", join(xs, ","), "]");
    jt = jsontable(s)
    df = DataFrame(jt)
    ef = select(df, :wordSize => :w, :posSize => :p, :shapeSize => :s, 
        :embeddingSize => :e, :hiddenSize => :h, # should group by these columns
        :trainingTime => :time,        
        :trainingAccuracy => :t, :developmentAccuracy => :d, :testAccuracy => :a,
        :trainingUAS => :tu, :devUAS => :du, :testUAS => :u,
        :trainingLAS => :tl, :devLAS => :dl, :testLAS => :l,
    )
end

function analyze(df::DataFrame)
    # group by embedding size (:e) and compute averages
    gf = groupby(df, :e)
    # ff = combine(gf, :t => mean, :d => mean, :a => mean, :tu => mean, :tl => mean, :du => mean, :dl => mean, :u => mean, :l => mean)
    ff = combine(gf, valuecols(gf) .=> mean)
    # group by two columns
    hf = groupby(df, [:e, :w])
    kf = combine(hf, valuecols(hf) .=> mean)
end