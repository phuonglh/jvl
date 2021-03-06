# phuonglh
# Evaluate the performance of classifier and parser, write scores
# to JSON file

using JSON3
using JSONTables
using DataFrames
using Statistics
using Flux

include("ParserEx.jl")

using .ArcEagerParserEx.TransitionClassifierEx

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
    sentencesTrain = TransitionClassifierEx.readCorpusUD(options[:trainCorpus], options[:maxSequenceLength])
    sentencesDev = TransitionClassifierEx.readCorpusUD(options[:validCorpus], options[:maxSequenceLength])
    sentencesTest = TransitionClassifierEx.readCorpusUD(options[:testCorpus], options[:maxSequenceLength])
    
    for t = 1:times
        local elapsedTime = time_ns()
        TransitionClassifierEx.train(options)
        elapsedTime = time_ns() - elapsedTime
        mlp, wordIndex, shapeIndex, posIndex, labelIndex = TransitionClassifier.load(options)
        accuracyTrain = TransitionClassifierEx.evaluate(mlp, wordIndex, shapeIndex, posIndex, labelIndex, options, sentencesTrain)
        accuracyDev = TransitionClassifierEx.evaluate(mlp, wordIndex, shapeIndex, posIndex, labelIndex, options, sentencesDev)
        accuracyTest = TransitionClassifierEx.evaluate(mlp, wordIndex, shapeIndex, posIndex, labelIndex, options, sentencesTest)
        trainingUAS, trainingLAS = ArcEagerParserEx.evaluate(mlp, wordIndex, shapeIndex, posIndex, labelIndex, options, sentencesTrain)
        devUAS, devLAS = ArcEagerParserEx.evaluate(mlp, wordIndex, shapeIndex, posIndex, labelIndex, options, sentencesDev)
        testUAS, testLAS = ArcEagerParserEx.evaluate(mlp, wordIndex, shapeIndex, posIndex, labelIndex, options, sentencesTest)
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
