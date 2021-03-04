
# Vietnamese dependency treebank
optionsVUD = Dict{Symbol,Any}(
    :mode => :train,
    :minFreq => 1,
    :lowercase => true,
    :maxSequenceLength => 40,
    :featuresPerContext => 4,
    :vocabSize => 2^16,
    :wordSize => 25,
    :shapeSize => 4,
    :posSize => 16, 
    :embeddingSize => 32, # RNN output dimension
    :hiddenSize => 128, # Dense layer output dimension
    :batchSize => 32,
    :numEpochs => 30,
    :trainCorpus => string(pwd(), "/dat/dep/vie/vi_vtb-ud-train.conllu"),
    :validCorpus => string(pwd(), "/dat/dep/vie/vi_vtb-ud-dev.conllu"),
    :testCorpus => string(pwd(), "/dat/dep/vie/vi_vtb-ud-test.conllu"),
    :modelPath => string(pwd(), "/dat/aep/vie-mlp.bson"),
    :wordPath => string(pwd(), "/dat/aep/vie-word.txt"),
    :shapePath => string(pwd(), "/dat/aep/vie-shape.txt"),
    :posPath => string(pwd(), "/dat/aep/vie-partOfSpeech.txt"),
    :labelPath => string(pwd(), "/dat/aep/vie-label.txt"),
    :numCores => 4,
    :verbose => false,
    :gpu => false,
    :logPath => string(pwd(), "/dat/aep/vie-loss.txt"),
    :scorePath => string(pwd(), "/dat/aep/vie-score-BiGRU.jsonl"),
    :unknown => "[unk]",
    :padding => "[pad]"
)

# English Web Treebank corpus
optionsEWT = Dict{Symbol,Any}(
    :mode => :train,
    :minFreq => 2,
    :lowercase => true,
    :maxSequenceLength => 40,
    :featuresPerContext => 4,
    :vocabSize => 2^16,
    :wordSize => 100,
    :shapeSize => 4,
    :posSize => 25, 
    :embeddingSize => 64,
    :hiddenSize => 128,
    :batchSize => 32,
    :numEpochs => 30,
    :trainCorpus => string(pwd(), "/dat/dep/eng/2.7/en_ewt-ud-train.conllu"),
    :validCorpus => string(pwd(), "/dat/dep/eng/2.7/en_ewt-ud-dev.conllu"),
    :testCorpus => string(pwd(), "/dat/dep/eng/2.7/en_ewt-ud-test.conllu"),
    :modelPath => string(pwd(), "/dat/aep/eng-mlp.bson"),
    :wordPath => string(pwd(), "/dat/aep/eng-word.txt"),
    :shapePath => string(pwd(), "/dat/aep/eng-shape.txt"),
    :posPath => string(pwd(), "/dat/aep/eng-partOfSpeech.txt"),
    :labelPath => string(pwd(), "/dat/aep/eng-label.txt"),
    :numCores => 4,
    :verbose => false,
    :gpu => false,
    :logPath => string(pwd(), "/dat/aep/eng-loss.txt"),
    :scorePath => string(pwd(), "/dat/aep/eng-score-BiGRU.jsonl"),
    :unknown => "[unk]",
    :padding => "[pad]"
)

# Bahasa Indonesia GSD corpus
optionsGSD = Dict{Symbol,Any}(
    :mode => :train,
    :minFreq => 2,
    :lowercase => true,
    :maxSequenceLength => 40,
    :featuresPerContext => 4,
    :vocabSize => 2^16,
    :wordSize => 100,
    :shapeSize => 4,
    :posSize => 25, 
    :embeddingSize => 64,
    :hiddenSize => 128,
    :batchSize => 32,
    :numEpochs => 30,
    :trainCorpus => string(pwd(), "/dat/dep/ind/id_gsd-ud-train.conllu"),
    :validCorpus => string(pwd(), "/dat/dep/ind/id_gsd-ud-dev.conllu"),
    :testCorpus => string(pwd(), "/dat/dep/ind/id_gsd-ud-test.conllu"),
    :modelPath => string(pwd(), "/dat/aep/ind-mlp.bson"),
    :wordPath => string(pwd(), "/dat/aep/ind-word.txt"),
    :shapePath => string(pwd(), "/dat/aep/ind-shape.txt"),
    :posPath => string(pwd(), "/dat/aep/ind-partOfSpeech.txt"),
    :labelPath => string(pwd(), "/dat/aep/ind-label.txt"),
    :numCores => 4,
    :verbose => false,
    :gpu => false,
    :logPath => string(pwd(), "/dat/aep/ind-loss.txt"),
    :scorePath => string(pwd(), "/dat/aep/ind-score-BiGRU.jsonl"),
    :unknown => "[unk]",
    :padding => "[pad]"
)
