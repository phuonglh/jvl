# phuonglh@gmail.com
# 

module Ranker

using Transformers
using Transformers.Basic
using Transformers.Pretrain

# load a pre-trained mBERT model (see ~/.julia/datadeps/)
bert_model, wordpiece, tokenizer = pretrain"bert-multi_cased_L-12_H-768_A-12"
# build a vocab
vocab = Vocabulary(wordpiece)

"""
    featurize(sentence1, sentence2)

    Find the BERT representation of a pair of sentences. This function returns a `768 x T` matrix where 
    `T` is the total length of the two sentences.

    `sentence` is a sequence of combined keywords, `sentence2` is a sample question.
"""
function featurize(sentence1, sentence2)
    pieces1 = sentence1 |> tokenizer |> wordpiece
    pieces2 = sentence2 |> tokenizer |> wordpiece
    pieces = ["[CLS]"; pieces1; "[SEP]"; pieces2; "[SEP]"]
    piece_indices = vocab(pieces)
    segment_indices = [fill(1, length(pieces1)+2); fill(2, length(pieces2)+1)]

    sample = (tok = piece_indices, segment = segment_indices)
    embeddings = sample |> bert_model.embed
    features = embeddings |> bert_model.transformers
    return features
end

end # module