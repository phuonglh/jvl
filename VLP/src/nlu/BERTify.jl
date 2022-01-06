# Convert the intent detection samples into vectors using BERT.
# phuonglh@gmail.com

using Transformers
using Transformers.Basic
using Transformers.Pretrain
using CSV
using DataFrames

include("Corpus.jl")
using .Corpus

# read sample sentences
df = readIntents("dat/nlu/sample.txt")
sentences = df[:, :text]


# load a pre-trained BERT model for English (see ~/.julia/datadeps/)
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
# load mBERT (see ~/.julia/datadeps/)
# bert_model, wordpiece, tokenizer = pretrain"bert-multi_cased_L-12_H-768_A-12"
vocab = Vocabulary(wordpiece)

"""
    featurize(sentence)

    Transforms a sentence to a vector using a pre-trained BERT model.
"""
function featurize(sentence::String)::Vector{Float32}
    pieces = sentence |> tokenizer |> wordpiece
    pieces = ["[CLS]"; pieces; "[SEP]"]
    piece_indices = vocab(pieces)
    segment_indices = fill(1, length(pieces))

    sample = (tok = piece_indices, segment = segment_indices)
    embeddings = sample |> bert_model.embed
    # compute a matrix of shape (768 x length(pieces))
    features = embeddings |> bert_model.transformers
    # bag-of-features, which is a Matrix{Float32} matrix of shape (768 x 1)
    v = sum(features, dims=2)
    return vec(v)
end

# transform all the sentences to vectors
println("BERTifying sentences...")
# This should take about 550 seconds on a MBP mid-2015 for all 10,000 sentences.
@time xs = map(featurize, sentences)

# save the vectors to a CSV file
outputPath = "dat/nlu/sample-BERTified.txt"
file = open(outputPath, "w")
for x in xs
    write(file, join(x, " "))
    write(file, "\n")
end
close(file)

println("Done.")