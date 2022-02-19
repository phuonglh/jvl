# phuonglh
# 
options = Dict{Symbol,Any}(
    :maxSequenceLength => 4,
    :afinnSize => 32,
    :recurrentSize => 32,
    :hiddenSize => 32, 
    :batchSize => 64,
    :α => 1E-3, 
    :numEpochs => 40,
    :modelPath => string(pwd(), "/dat/emo/EMP/emp.bson")
)