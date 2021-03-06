# phuonglh
# Analyze processing results (tagging, parsing, etc.)
# Results in JSON files are loaded in to data frames, then they are analyzed.

using JSON3
using JSONTables
using DataFrames
using CSV
using Statistics

"""
    aep(language)

    Read experimental results of the AEP module into a data frame.
"""
function aep(language::String="vie")::DataFrame
    path = string("dat/aep/", language, "-score-BiGRU.jsonl")
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

function analyzeAEP(df::DataFrame)
    # group by embedding size (:e) and compute averages
    gf = groupby(df, :e)
    # ff = combine(gf, :t => mean, :d => mean, :a => mean, :tu => mean, :tl => mean, :du => mean, :dl => mean, :u => mean, :l => mean)
    ff = combine(gf, valuecols(gf) .=> mean)
    # group by two columns
    hf = groupby(df, [:e, :w])
    kf = combine(hf, valuecols(hf) .=> mean)
end

function tdp(language::String="vie", arch::String="bof")::DataFrame
    path = string("dat/tdp/experiments-", language, "-", arch, ".tsv")
    df = DataFrame(CSV.File(path))
    ef = select(df, 
        :embeddingSize => (x -> Int.(x)) => :e, 
        :hiddenSize => (x -> Int.(x)) => :h,
        :trainingAcc => :t, :devAcc => :d, :testAcc => :a,
        :trainingUAS => :tu, :devUAS => :du, :testUAS => :u,
        :trainingLAS => :tl, :devLAS => :dl, :testLAS => :l,
    )
end

function analyseTDP(df::DataFrame)
    # group by embedding size (:e) and hidden size (:h)
    hf = groupby(df, [:e, :h])
    ff = combine(hf, valuecols(hf) .=> (x -> round.(mean(x)*100, digits=2)))
end