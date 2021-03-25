# phuonglh
# Analyze processing results (tagging, parsing, etc.)
# Results in JSON files are loaded in to data frames, then they are analyzed.

using JSON3
using JSONTables
using DataFrames
using CSV
using Statistics
using Plots

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

"""
    aep(language, ex)

    Read experimental results of the AEP module into a data frame.
"""
function aep(language::String="vie", ex::String="")::DataFrame
    path = string("dat/aep/experiments-", language, ".jsonl", ex)
    xs = readlines(path)
    s = string("[", join(xs, ","), "]");
    jt = jsontable(s)
    df = DataFrame(jt)
    ef = select(df, 
        :wordSize => :w, :posSize => :p, :shapeSize => :s, 
        :bidirectional => :bi,
        :recurrentSize => :r, :hiddenSize => :h, # should group by these columns
        :trainingTime => :time,        
        :trainingAccuracy => :t, :developmentAccuracy => :d, :testAccuracy => :a,
        :trainingUAS => :tu, :devUAS => :du, :testUAS => :u,
        :trainingLAS => :tl, :devLAS => :dl, :testLAS => :l,
    )
end

function analyzeAEP(language::String="vie", ex::String="", unidirectional::Bool=true)
    df = aep(language, ex)
    hs = [64, 128, 256]
    for h in hs
        u = if unidirectional ".u" else ".b"; end
        output = string("dat/aep/aep-", language, ".h", h, u, ex, ".txt")
        ef = df[df.bi .== unidirectional, :] # filter unidirectional or bidirectional results
        ef = ef[ef.h .== h, :] # filter hidden size
        kf = groupby(ef, [:r, :w])
        gf = combine(kf, valuecols(kf) .=> (x -> round.(mean(x)*100, digits=2)))
        hf = select(gf, :r => :r, :w => :w, :u_function => :u, :l_function => :l)
        CSV.write(output, hf)
    end
end

"""
    compareAEP(language, w, h, unidirectional)

    Compares empirical results between standard and extended parsing model given 
    a word embedding size and a hidden size. The x-axis shows recurrent sizes and 
    the y-axis shows the LAS values.
"""
function compareAEP(language::String="vie", w::Int=25, h::Int=64, unidirectional::Bool=true)
    u = if unidirectional ".u" else ".b"; end
    output1 = string("dat/aep/aep-", language, ".h", h, u, ".txt")
    output2 = string("dat/aep/aep-", language, ".h", h, u, ".ex.txt")
    df1 = CSV.read(output1, DataFrame)
    df2 = CSV.read(output2, DataFrame)
    dfA = df1[df1.w .== w, :]
    dfB = df2[df2.w .== w, :]
    plot(dfA[:, :r], [dfA[:, :l], dfB[:, :l]], label=["standard" "extended"], 
        title=string("w=", w, ", h=", h),
        xlabel="recurrent size", ylabel="LAS", legend=:bottomright)
end