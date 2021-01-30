
# Vietnamese dependency treebank
optionsVUD = Dict{Symbol,Any}(
    :minFreq => 2,
    :lowercase => true,
    :maxSequenceLength => 40,
    :featuresPerContext => 20,
    :numFeatures => 2^16,
    :embeddingSize => 100,
    :hiddenSize => 64,
    :batchSize => 32,
    :numEpochs => 40,
    :trainCorpus => string(pwd(), "/dat/dep/vie/vi_vtb-ud-train.conllu"),
    :devCorpus => string(pwd(), "/dat/dep/vie/vi_vtb-ud-dev.conllu"),
    :testCorpus => string(pwd(), "/dat/dep/vie/vi_vtb-ud-test.conllu"),
    :modelPath => string(pwd(), "/tdp/dat/vie/mlp.bson"),
    :vocabPath => string(pwd(), "/tdp/dat/vie/vocab.txt"),
    :labelPath => string(pwd(), "/tdp/dat/vie/label.txt"),
    :statPath => string(pwd(), "/tdp/dat/vie/stat.txt"),
    :numCores => 4,
    :verbose => false,
    :gpu => false,
    :logPath => string(pwd(), "/tdp/dat/vie/loss.txt")
)

# English Web Treebank corpus
optionsEWT = Dict{Symbol,Any}(
    :minFreq => 2,
    :lowercase => true,
    :maxSequenceLength => 40,
    :featuresPerContext => 20,
    :numFeatures => 2^16,
    :embeddingSize => 100,
    :hiddenSize => 128,
    :batchSize => 32,
    :numEpochs => 40,
    :trainCorpus => string(pwd(), "/dat/dep/eng/2.7/en_ewt-ud-train.conllu"),
    :devCorpus => string(pwd(), "/dat/dep/eng/2.7/en_ewt-ud-dev.conllu"),
    :testCorpus => string(pwd(), "/dat/dep/eng/2.7/en_ewt-ud-test.conllu"),
    :modelPath => string(pwd(), "/tdp/dat/eng/mlp.bson"),
    :vocabPath => string(pwd(), "/tdp/dat/eng/vocab.txt"),
    :labelPath => string(pwd(), "/tdp/dat/eng/label.txt"),
    :statPath => string(pwd(), "/tdp/dat/eng/stat.txt"),
    :numCores => 4,
    :verbose => false,
    :gpu => false,
    :logPath => string(pwd(), "/tdp/dat/eng/loss.txt")
)

# Bahasa Indonesia dependency treebank
optionsGSD = Dict{Symbol,Any}(
    :minFreq => 2,
    :lowercase => true,
    :maxSequenceLength => 40,
    :featuresPerContext => 20,
    :numFeatures => 2^16,
    :embeddingSize => 100,
    :hiddenSize => 64,
    :batchSize => 32,
    :numEpochs => 40,
    :trainCorpus => string(pwd(), "/dat/dep/ind/id_gsd-ud-train.conllu"),
    :devCorpus => string(pwd(), "/dat/dep/ind/id_gsd-ud-dev.conllu"),
    :testCorpus => string(pwd(), "/dat/dep/ind/id_gsd-ud-test.conllu"),
    :modelPath => string(pwd(), "/tdp/dat/ind/mlp.bson"),
    :vocabPath => string(pwd(), "/tdp/dat/ind/vocab.txt"),
    :labelPath => string(pwd(), "/tdp/dat/ind/label.txt"),
    :statPath => string(pwd(), "/tdp/dat/ind/stat.txt"),
    :numCores => 4,
    :verbose => false,
    :gpu => false,
    :logPath => string(pwd(), "/tdp/dat/ind/loss.txt")
)
