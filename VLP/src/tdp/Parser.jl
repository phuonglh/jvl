module DependencyParser

using Flux
using FLoops

include("Classifier.jl")
using .TransitionClassifier

include("Oracle.jl")
using .Corpus

include("../tok/VietnameseTokenizer.jl")
using .VietnameseTokenizer



"""
    run(mlp, featureIndex, labelIndex, options, sentences)

    Parse an array of sentences. A pre-trained transition classifier, its associated feature index and label index 
    will be loaded from saved files in advance. The input sentences are updated in-place by appending :h and :l annotations to 
    each token for head and dependency label predictions.
"""
function run(mlp, featureIndex, labelIndex, options::Dict{Symbol,Any}, sentences::Array{Sentence})
    # create (index => label) map
    labels = Dict{Int,String}(labelIndex[label] => label for label in keys(labelIndex))
    # parse the sentences in parallel to speed up the processing
    @floop ThreadedEx(basesize=length(sentences)÷options[:numCores]) for sentence in sentences
        σ = Stack{String}()
        β = Queue{String}()
        tokenMap = Dict{String,Token}(token.annotation[:id] => token for token in sentence.tokens)
        for id in map(token -> token.annotation[:id], sentence.tokens)
            enqueue!(β, id)
        end
        A = Array{Arc,1}()
        config = Config(σ, β, A)
        config = next(config, "SH")
        transition = "SH"
        while !isempty(β) && !isempty(σ)
            features = featurize(config, tokenMap)
            x = unique(map(f -> get(featureIndex, f, 1), features))
            # pad x if necessary
            append!(x, ones(options[:featuresPerContext] - length(x)))
            y = mlp(x)
            transition = labels[Flux.onecold(y)[1]]
            config = next(config, transition)
        end
        # updated the predicted head (:h) and predicted dependency labels (:l) for all tokens
        for arc in config.arcs
            token = tokenMap[arc.dependent]
            token.annotation[:h] = arc.head
            token.annotation[:l] = arc.label
        end
    end
end

"""
    evaluate(mlp, featureIndex, labelIndex, options, sentences)

    Evaluate the accuracy of the parser: compute the UAS and LAS scores.
"""
function evaluate(mlp, featureIndex, labelIndex, options::Dict{Symbol,Any}, sentences::Array{Sentence})::Tuple{Float64,Float64}
    run(mlp, featureIndex, labelIndex, options, sentences)
    uas, las = 0, 0
    numTokens = 0
    for sentence in sentences
        numTokens += length(sentence.tokens)
        for token in sentence.tokens
            if (:h ∈ keys(token.annotation))
                uas += (token.annotation[:head] == token.annotation[:h])
                if (:l ∈ keys(token.annotation))
                    las += (token.annotation[:head] == token.annotation[:h]) && (token.annotation[:label] == token.annotation[:l])
                end
            end
        end
    end
    (uas, las) ./ numTokens
end

"""
    evaluate(options)

    Evaluate the performance of the parser on all train/dev./test datasets.
"""
function evaluate(options)
    mlp, featureIndex, labelIndex = TransitionClassifier.load(options)
    sentences = readCorpusUD(options[:trainCorpus], options[:maxSequenceLength])
    @time uas, las = evaluate(mlp, featureIndex, labelIndex, options, sentences)
    @info length(sentences)
    @info "Training scores: UAS = $uas, LAS = $las"

    sentences = readCorpusUD(options[:validCorpus], options[:maxSequenceLength])
    @time uas, las = evaluate(mlp, featureIndex, labelIndex, options, sentences)
    @info length(sentences)
    @info "Development scores: UAS = $uas, LAS = $las"

    sentences = readCorpusUD(options[:testCorpus], options[:maxSequenceLength])
    @time uas, las = evaluate(mlp, featureIndex, labelIndex, options, sentences)
    @info length(sentences)
    @info "Test scores: UAS = $uas, LAS = $las"
end

end # module