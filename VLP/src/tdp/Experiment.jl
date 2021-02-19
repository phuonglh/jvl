# phuonglh
# An utility to perform series of experiments and write results to TSV files
# for further analysis.

include("Parser.jl")

using Flux
using .DependencyParser.TransitionClassifier

language = "vie" # ind/eng

options = if language == "vie"
    TransitionClassifier.optionsVUD
elseif language == "ind"
    TransitionClassifier.optionsGSD
else
    TransitionClassifier.optionsEWT
end

embeddingSizes = [25, 50, 75, 100]
hiddenSizes = [32, 64, 128, 256, 300]

sentences = TransitionClassifier.readCorpusUD(options[:trainCorpus])
sentencesDev = TransitionClassifier.readCorpusUD(options[:validCorpus])
sentencesTest = TransitionClassifier.readCorpusUD(options[:testCorpus])

file = open(string(pwd(), "/dat/tdp/experiments-", language, ".tsv"), "w")
write(file, "embeddingSize\thiddenSize\ttrainingAcc\tdevAcc\ttestAcc\trainingUAS\ttrainingLAS\tdevUAS\tdevLAS\ttestUAS\ttestLAS\n")
numExperiments = 3
for k = 1:numExperiments
    for e in embeddingSizes
        for h in hiddenSizes 
            # update hyper-parameters
            options[:embeddingSize] = e
            options[:hiddenSize] = h
            # train a classifier
            TransitionClassifier.train(options)
            # load the classifier as well as necessary indices
            mlp, featureIndex, labelIndex = TransitionClassifier.load(options)
            # evaluate the classifier
            trainingAcc = TransitionClassifier.evaluate(mlp, featureIndex, labelIndex, options, sentences)
            devAcc = TransitionClassifier.evaluate(mlp, featureIndex, labelIndex, options, sentencesDev)
            testAcc = TransitionClassifier.evaluate(mlp, featureIndex, labelIndex, options, sentencesTest)
            # evaluate the parser
            trainingUAS, trainingLAS = DependencyParser.evaluate(mlp, featureIndex, labelIndex, options, sentences)
            devUAS, devLAS = DependencyParser.evaluate(mlp, featureIndex, labelIndex, options, sentencesDev)
            testUAS, testLAS = DependencyParser.evaluate(mlp, featureIndex, labelIndex, options, sentencesTest)
            # collect and write result
            parts = [e, h, trainingAcc, devAcc, testAcc, trainingUAS, trainingLAS, devUAS, devLAS, testUAS, testLAS]
            write(file, string(join(parts, "\t"), "\n"))
            flush(file)
        end
    end
end
close(file)
