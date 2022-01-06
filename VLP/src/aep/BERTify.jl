# Compute all EWT sentences into (English) BERT vectors.
# phuonglh@gmail.com
# December 2021

include("../tdp/Oracle.jl")
include("Options.jl")

using Transformers
using Transformers.Basic
using Transformers.Pretrain

# train/valid/test splits
inputPath = optionsEWT[:trainCorpus]
graphs = Corpus.readCorpusUD(inputPath, optionsEWT[:maxSequenceLength])
println("Number of sentences = $(length(graphs))")

function convert(graph)
    ws = map(token -> token.word, graph.tokens)
    return join(ws[2:end], " ")
end

# convert all graphs to raw sentences
sentences = map(convert, graphs)

# load a pre-trained BERT model for English (see ~/.julia/datadeps/)
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
# load mBERT (see ~/.julia/datadeps/)
# bert_model, wordpiece, tokenizer = pretrain"bert-multi_cased_L-12_H-768_A-12"
vocab = Vocabulary(wordpiece)

"""
    featurize(sentence)

    Transforms a sentence to a vector using a pre-trained BERT model.
"""
function featurize(sentence::String)::Matrix{Float32}
    pieces = sentence |> tokenizer |> wordpiece
    piece_indices = vocab(pieces)
    segment_indices = fill(1, length(pieces))

    sample = (tok = piece_indices, segment = segment_indices)
    embeddings = sample |> bert_model.embed
    # compute a matrix of shape (768 x length(pieces))
    features = embeddings |> bert_model.transformers
    return features
end

# transform all the sentences to vectors
println("BERTifying sentences...")
@time Xs = map(featurize, sentences)

# save the vectors to a CSV file
outputPath = string(inputPath, "-BERTified.txt")
file = open(outputPath, "w")
for X in Xs
    ss = map(j -> join(X[:,j], " "), 1:size(X,2))
    write(file, join(ss, "\n"))
    write(file, "\n\n")
end
close(file)

println("Done.")
