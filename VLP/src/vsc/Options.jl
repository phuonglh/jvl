# phuonglh
# Options used for ScRNN model.

options = Dict{Symbol,Any}(
    :hiddenSize => 32,
    :maxSequenceLength => 80,
    :numEpochs => 200,
    :batchSize => 32, 
    :labels => [:n, :s, :r, :i, :d, :P], # :p is the padding symbol
    :inputPath => string(pwd(), "/dat/vsc/100.txt.inp"),
    :outputPath => string(pwd(), "/dat/vsc/100.txt.out"),
    :modelPath => string(pwd(), "/dat/vsc/100.bson"),
    :alphabetPath => string(pwd(), "/dat/vsc/100.alphabet"),
    :gpu => false,
    :verbose => false
)

optionsVTB = Dict{Symbol,Any}(
    :hiddenSize => 128,
    :maxSequenceLength => 80,
    :numEpochs => 40,
    :batchSize => 32, 
    :labels => [:n, :s, :r, :i, :d, :P], # :p is the padding symbol
    :inputPath => string(pwd(), "/dat/vsc/vtb.txt.inp"),
    :outputPath => string(pwd(), "/dat/vsc/vtb.txt.out"),
    :modelPath => string(pwd(), "/dat/vsc/vtb.bson"),
    :alphabetPath => string(pwd(), "/dat/vsc/vtb.alphabet"),
    :gpu => false,
    :verbose => false
)

optionsVLSP = Dict{Symbol,Any}(
    :hiddenSize => 128,
    :maxSequenceLength => 80,
    :numEpochs => 40,
    :batchSize => 32, 
    :labels => [:n, :s, :r, :i, :d, :P], # :p is the padding symbol
    :inputPath => string(pwd(), "/dat/vsc/vlsp.txt.inp"),
    :outputPath => string(pwd(), "/dat/vsc/vlsp.txt.out"),
    :modelPath => string(pwd(), "/dat/vsc/vlsp.bson"),
    :alphabetPath => string(pwd(), "/dat/vsc/vlsp.alphabet"),
    :gpu => false,
    :verbose => false
)

# Vietnamese UD treebank (training split)
optionsVUD = Dict{Symbol,Any}(
    :hiddenSize => 128,
    :maxSequenceLength => 40,
    :numEpochs => 40,
    :batchSize => 32, 
    :labels => [:n, :s, :r, :i, :d, :P], # :p is the padding symbol
    :inputPath => string(pwd(), "/dat/vsc/vi_vtb-ud-train.txt.inp"),
    :outputPath => string(pwd(), "/dat/vsc/vi_vtb-ud-train.txt.out"),
    :modelPath => string(pwd(), "/dat/vsc/vud.bson"),
    :alphabetPath => string(pwd(), "/dat/vsc/vud.alphabet"),
    :gpu => false,
    :verbose => false
)

# Finance corpus
optionsFinance = Dict{Symbol,Any}(
    :hiddenSize => 128,
    :maxSequenceLength => 40,
    :numEpochs => 40,
    :batchSize => 64, 
    :labels => [:n, :s, :r, :i, :d, :P], # :p is the padding symbol
    :inputPath => string(pwd(), "/dat/vsc/finance.txt.inp"),
    :outputPath => string(pwd(), "/dat/vsc/finance.txt.out"),
    :modelPath => string(pwd(), "/dat/vsc/finance.bson"),
    :alphabetPath => string(pwd(), "/dat/vsc/finance.alphabet"),
    :gpu => false,
    :verbose => false
)
