# phuonglh
# Evaluate the performance of classifier and parser, write scores
# to JSON file

using JSON3
using JSONTables
using DataFrames
using Statistics
using Flux

include("Parser.jl")

using .ArcEagerParser.TransitionClassifier

"""
    experiment(options, times=3)

    Perform experimentation with a given number of times.
"""
function experiment(options, times=3)
    scorePath = options[:scorePath]
    file = if isfile(scorePath)
        open(scorePath, "a")
    else
        open(scorePath, "w")
    end
    sentencesTrain = TransitionClassifier.readCorpusUD(options[:trainCorpus], options[:maxSequenceLength])
    sentencesDev = TransitionClassifier.readCorpusUD(options[:validCorpus], options[:maxSequenceLength])
    sentencesTest = TransitionClassifier.readCorpusUD(options[:testCorpus], options[:maxSequenceLength])
    
    for t = 1:times
        local elapsedTime = time_ns()
        TransitionClassifier.train(options)
        elapsedTime = time_ns() - elapsedTime
        mlp, wordIndex, shapeIndex, posIndex, labelIndex = TransitionClassifier.load(options)
        accuracyTrain = TransitionClassifier.evaluate(mlp, wordIndex, shapeIndex, posIndex, labelIndex, options, sentencesTrain)
        accuracyDev = TransitionClassifier.evaluate(mlp, wordIndex, shapeIndex, posIndex, labelIndex, options, sentencesDev)
        accuracyTest = TransitionClassifier.evaluate(mlp, wordIndex, shapeIndex, posIndex, labelIndex, options, sentencesTest)
        trainingUAS, trainingLAS = ArcEagerParser.evaluate(mlp, wordIndex, shapeIndex, posIndex, labelIndex, options, sentencesTrain)
        devUAS, devLAS = ArcEagerParser.evaluate(mlp, wordIndex, shapeIndex, posIndex, labelIndex, options, sentencesDev)
        testUAS, testLAS = ArcEagerParser.evaluate(mlp, wordIndex, shapeIndex, posIndex, labelIndex, options, sentencesTest)
        local scores = Dict{Symbol,Any}(
            :trainCorpus => options[:trainCorpus],
            :minFreq => options[:minFreq],
            :maxSequenceLength => options[:maxSequenceLength],
            :wordSize => options[:wordSize],
            :shapeSize => options[:shapeSize],
            :posSize => options[:posSize], 
            :embeddingSize => options[:embeddingSize],
            :hiddenSize => options[:hiddenSize],
            :batchSize => options[:batchSize],
            :trainingTime => elapsedTime,
            :trainingAccuracy => accuracyTrain,
            :developmentAccuracy => accuracyDev,
            :testAccuracy => accuracyTest,
            :trainingUAS => trainingUAS,
            :trainingLAS => trainingLAS,
            :devUAS => devUAS,
            :devLAS => devLAS,
            :testUAS => testUAS,
            :testLAS => testLAS
        )
        line = JSON3.write(scores)
        write(file, string(line, "\n"))
        flush(file)
    end
    close(file)
end

"""
    toDF(options)

    Load experimental results into a data frame for analysis.
"""
function toDF(options)
    # read lines from the score path, concatenate them into an json array object
    lines = readlines(options[:scorePath])
    s = string("[", join(lines, ","), "]")
    # convert to a json table
    jt = jsontable(s)
    # convert to a data frame
    DataFrame(jt)
end


"""
    analyse(options)

    Analyse the experimental results.
"""
function analyse(options)
    df = toDF(options)
    # select test scores and hidden size to see the effect of varying hidden size
    testScores = select(df, [:hiddenSize, :testAccuracy, :testUAS, :testLAS])
    # group test scores by hidden size
    gdf = groupby(testScores, :hiddenSize)
    # compute mean scores for each group
    combine(gdf, names(gdf) .=> mean)
end
