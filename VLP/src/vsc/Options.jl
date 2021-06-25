# phuonglh
# Options used for ScRNN model.

options = Dict{Symbol,Any}(
    :hiddenSize => 32,
    :maxSequenceLength => 80,
    :numEpochs => 40,
    :batchSize => 32, 
    :labels => [:n, :s, :r, :i, :d, :P], # :p is the padding symbol
    :inputPath => string(pwd(), "/dat/vsc/200.txt.inp"),
    :outputPath => string(pwd(), "/dat/vsc/200.txt.out"),
    :modelPath => string(pwd(), "/dat/vsc/200.bson"),
    :alphabetPath => string(pwd(), "/dat/vsc/200.alphabet"),
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
