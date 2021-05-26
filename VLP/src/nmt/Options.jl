
options = Dict{Symbol,Any}(
    :minFreq => 3,
    :maxSequenceLength => 30,
    :batchSize => 64,
    :numEpochs => 100,
    :numCores => 4,
    :α => 1E-4,
    :delimiters => r"""[\s.,?!;:")(']+""",
    :corpusPath => string(pwd(), "/dat/nmt/1000.txt"),
    :sourceCorpusPath => string(pwd(), "/dat/nmt/5000.fr"),
    :sourceDictPath => string(pwd(), "/dat/nmt/src.txt"),
    :targetCorpusPath => string(pwd(), "/dat/nmt/5000.en"),
    :targetDictPath => string(pwd(), "/dat/nmt/tar.txt"),
    :logPath => string(pwd(), "/dat/nmt/nmt.log"),
    :modelPath => string(pwd(), "/dat/nmt/model.bson"),
    :inputSize => 100,
    :hiddenSize => 256,
    :outputSize => 100,
    :PAD => "[PAD]",
    :EOS => "[EOS]",
    :UNK => "[UNK]",
    :gpu => false
)