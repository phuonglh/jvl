# phuonglh
# 
options = Dict{Symbol,Any}(
    :maxSequenceLength => 4, # max number of AFINN tokens 
    :afinnSize => 32,
    :recurrentSize => 16,
    :hiddenSize => 16, 
    :batchSize => 64,
    :α => 1E-3, 
    :numEpochs => 30,
    :minFrequency => 5,
    :maxFrequency => 80,
    :modelPath => string(pwd(), "/dat/emo/EMP/emp.bson")
)