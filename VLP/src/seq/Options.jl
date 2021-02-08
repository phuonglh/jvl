# For Vietnamese PoS tagging on the VLSP-2010 treebank
optionsVLSP2010 = Dict{Symbol,Any}(
    :minFreq => 2,
    :lowercase => true,
    :vocabSize => 2^16,
    :wordSize => 50,
    :shapeSize => 4,
    :posSize => 1,
    :hiddenSize => 64,
    :maxSequenceLength => 40,
    :batchSize => 32,
    :numEpochs => 20,
    :trainCorpus => string(pwd(), "/dat/vtb/vtb-tagged.txt"),
    :validCorpus => string(pwd(), "/dat/vtb/vtb-tagged.txt"),
    :testCorpus => string(pwd(), "/dat/vtb/vtb-tagged.txt"),
    :modelPath => string(pwd(), "/dat/pos/vtb-network.bson"),
    :wordPath => string(pwd(), "/dat/pos/vtb-word.txt"),
    :shapePath => string(pwd(), "/dat/pos/vtb-shape.txt"),
    :posPath => string(pwd(), "/dat/pos/vtb-partOfSpeech.txt"),
    :labelPath => string(pwd(), "/dat/pos/vtb-label.txt"),
    :numCores => 4,
    :verbose => false,
    :logPath => string(pwd(), "/dat/pos/vtb-loss.txt"),
    :unknown => "[UNK]",
    :paddingX => "[PAD_X]",
    :paddingY => "[PAD_Y]",
    :trainOutput => string(pwd(), "/dat/pos/vtb-train.out"),
    :validOutput => string(pwd(), "/dat/pos/vtb-test.out"),
    :testOutput => string(pwd(), "/dat/pos/vtb-test.out"),
    :columnFormat => false
)

# For Vietnamese PoS tagging on the Vietnamese UD treebank
optionsVUD = Dict{Symbol,Any}(
    :minFreq => 2,
    :lowercase => true,
    :vocabSize => 2^16,
    :wordSize => 50,
    :shapeSize => 4,
    :posSize => 8,
    :hiddenSize => 64,
    :maxSequenceLength => 40,
    :batchSize => 32,
    :numEpochs => 20,
    :trainCorpus => string(pwd(), "/dat/dep/vie/vi_vtb-ud-train.conllu"),
    :validCorpus => string(pwd(), "/dat/dep/vie/vi_vtb-ud-dev.conllu"),
    :testCorpus => string(pwd(), "/dat/dep/vie/vi_vtb-ud-test.conllu"),
    :modelPath => string(pwd(), "/dat/pos/vie-network.bson"),
    :wordPath => string(pwd(), "/dat/pos/vie-word.txt"),
    :shapePath => string(pwd(), "/dat/pos/vie-shape.txt"),
    :posPath => string(pwd(), "/dat/pos/vie-partOfSpeech.txt"),
    :labelPath => string(pwd(), "/dat/pos/vie-label.txt"),
    :numCores => 4,
    :verbose => false,
    :logPath => string(pwd(), "/dat/pos/vie-loss.txt"),
    :unknown => "[UNK]",
    :paddingX => "[PAD_X]",
    :paddingY => "[PAD_Y]",
    :trainOutput => string(pwd(), "/dat/pos/vie-train.out"),
    :validOutput => string(pwd(), "/dat/pos/vie-test.out"),
    :testOutput => string(pwd(), "/dat/pos/vie-test.out"),
    :columnFormat => true
)

# For English PoS tagging on the English Web Treebank
optionsEWT = Dict{Symbol,Any}(
    :minFreq => 2,
    :lowercase => true,
    :vocabSize => 2^16,
    :wordSize => 100,
    :shapeSize => 4,
    :posSize => 8,
    :hiddenSize => 64,
    :maxSequenceLength => 40,
    :batchSize => 32,
    :numEpochs => 40,
    :trainCorpus => string(pwd(), "/dat/dep/eng/2.7/en_ewt-ud-train.conllu"),
    :validCorpus => string(pwd(), "/dat/dep/eng/2.7/en_ewt-ud-dev.conllu"),
    :testCorpus => string(pwd(), "/dat/dep/eng/2.7/en_ewt-ud-test.conllu"),
    :modelPath => string(pwd(), "/dat/pos/eng-network.bson"),
    :wordPath => string(pwd(), "/dat/pos/eng-word.txt"),
    :shapePath => string(pwd(), "/dat/pos/eng-shape.txt"),
    :posPath => string(pwd(), "/dat/pos/eng-partOfSpeech.txt"),
    :labelPath => string(pwd(), "/dat/pos/eng-label.txt"),
    :numCores => 4,
    :verbose => false,
    :logPath => string(pwd(), "/dat/pos/eng-loss.txt"),
    :unknown => "[UNK]",
    :paddingX => "[PAD_X]",
    :paddingY => "[PAD_Y]",
    :trainOutput => string(pwd(), "/dat/pos/eng-train.out"),
    :validOutput => string(pwd(), "/dat/pos/eng-test.out"),
    :testOutput => string(pwd(), "/dat/pos/eng-test.out"),
    :columnFormat => true
)

# For Bahasa Indonesia PoS tagging on GSD treebank
optionsGSD = Dict{Symbol,Any}(
    :minFreq => 1,
    :lowercase => true,
    :vocabSize => 2^16,
    :wordSize => 50,
    :shapeSize => 4,
    :posSize => 16,
    :hiddenSize => 32,
    :maxSequenceLength => 40,
    :batchSize => 32,
    :numEpochs => 20,
    :trainCorpus => string(pwd(), "/dat/dep/ind/id_gsd-ud-train.conllu"),
    :validCorpus => string(pwd(), "/dat/dep/ind/id_gsd-ud-dev.conllu"),
    :testCorpus => string(pwd(), "/dat/dep/ind/id_gsd-ud-test.conllu"),
    :modelPath => string(pwd(), "/dat/pos/ind-network.bson"),
    :wordPath => string(pwd(), "/dat/pos/ind-word.txt"),
    :shapePath => string(pwd(), "/dat/pos/ind-shape.txt"),
    :posPath => string(pwd(), "/dat/pos/ind-partOfSpeech.txt"),
    :labelPath => string(pwd(), "/dat/pos/ind-label.txt"),
    :numCores => 4,
    :verbose => false,
    :logPath => string(pwd(), "/dat/pos/ind-loss.txt"),
    :unknown => "[UNK]",
    :paddingX => "[PAD_X]",
    :paddingY => "[PAD_Y]",
    :trainOutput => string(pwd(), "/dat/pos/ind-train.out"),
    :validOutput => string(pwd(), "/dat/pos/ind-test.out"),
    :testOutput => string(pwd(), "/dat/pos/ind-test.out"),
    :columnFormat => true
)

# For Vietnamese NER
optionsVLSP2016 = Dict{Symbol,Any}(
    :minFreq => 2,
    :lowercase => true,
    :vocabSize => 2^16,
    :wordSize => 25,
    :shapeSize => 4,
    :posSize => 16,
    :hiddenSize => 32,
    :maxSequenceLength => 40,
    :batchSize => 32,
    :numEpochs => 20,
    :trainCorpus => string(pwd(), "/dat/ner/vie/vie.train"),
    :validCorpus => string(pwd(), "/dat/ner/vie/vie.test"),
    :testCorpus => string(pwd(), "/dat/ner/vie/vie.test"),
    :modelPath => string(pwd(), "/dat/ner/vie-network.bson"),
    :wordPath => string(pwd(), "/dat/ner/vie-word.txt"),
    :shapePath => string(pwd(), "/dat/ner/vie-shape.txt"),
    :posPath => string(pwd(), "/dat/ner/vie-partOfSpeech.txt"),
    :labelPath => string(pwd(), "/dat/ner/vie-label.txt"),
    :numCores => 4,
    :verbose => false,
    :logPath => string(pwd(), "/dat/ner/vie-loss.txt"),
    :unknown => "[UNK]",
    :paddingX => "[PAD_X]",
    :paddingY => "[PAD_Y]",
    :trainOutput => string(pwd(), "/dat/ner/vie-train.out"),
    :validOutput => string(pwd(), "/dat/ner/vie-test.out"),
    :testOutput => string(pwd(), "/dat/ner/vie-test.out"),
    :threeColumns => false
)

# For English CoNLL-2003 NER
optionsCoNLL2003 = Dict{Symbol,Any}(
    :minFreq => 2,
    :lowercase => true,
    :vocabSize => 2^16,
    :wordSize => 100,
    :shapeSize => 4,
    :posSize => 25,
    :hiddenSize => 64,
    :maxSequenceLength => 40,
    :batchSize => 32,
    :numEpochs => 20,
    :trainCorpus => string(pwd(), "/dat/ner/eng/eng.train"),
    :validCorpus => string(pwd(), "/dat/ner/eng/eng.testa"),
    :testCorpus => string(pwd(), "/dat/ner/eng/eng.testb"),
    :modelPath => string(pwd(), "/dat/ner/eng-network.bson"),
    :wordPath => string(pwd(), "/dat/ner/eng-word.txt"),
    :shapePath => string(pwd(), "/dat/ner/eng-shape.txt"),
    :posPath => string(pwd(), "/dat/ner/eng-partOfSpeech.txt"),
    :labelPath => string(pwd(), "/dat/ner/eng-label.txt"),
    :numCores => 4,
    :verbose => false,
    :logPath => string(pwd(), "/dat/ner/eng-loss.txt"),
    :unknown => "[UNK]",
    :paddingX => "[PAD_X]",
    :paddingY => "[PAD_Y]",
    :trainOutput => string(pwd(), "/dat/ner/eng-train.out"),
    :validOutput => string(pwd(), "/dat/ner/eng-testa.out"),
    :testOutput => string(pwd(), "/dat/ner/eng-testb.out"),
    :threeColumns => false
)

# For Bahasa Indonesia NER (2020)
optionsKIK2020 = Dict{Symbol,Any}(
    :minFreq => 2,
    :lowercase => true,
    :vocabSize => 2^16,
    :wordSize => 100,
    :shapeSize => 4,
    :posSize => 25,
    :hiddenSize => 64,
    :maxSequenceLength => 40,
    :batchSize => 32,
    :numEpochs => 20,
    :trainCorpus => string(pwd(), "/dat/ner/ind/train.txt"),
    :validCorpus => string(pwd(), "/dat/ner/ind/dev.txt"),
    :testCorpus => string(pwd(), "/dat/ner/ind/test.txt"),
    :modelPath => string(pwd(), "/dat/ner/ind-network.bson"),
    :wordPath => string(pwd(), "/dat/ner/ind-word.txt"),
    :shapePath => string(pwd(), "/dat/ner/ind-shape.txt"),
    :posPath => string(pwd(), "/dat/ner/ind-partOfSpeech.txt"),
    :labelPath => string(pwd(), "/dat/ner/ind-label.txt"),
    :numCores => 4,
    :verbose => false,
    :logPath => string(pwd(), "/dat/ner/ind-loss.txt"),
    :unknown => "[UNK]",
    :paddingX => "[PAD_X]",
    :paddingY => "[PAD_Y]",
    :trainOutput => string(pwd(), "/dat/ner/ind-train.out"),
    :validOutput => string(pwd(), "/dat/ner/ind-dev.out"),
    :testOutput => string(pwd(), "/dat/ner/ind-test.out"),
    :threeColumns => true
)
