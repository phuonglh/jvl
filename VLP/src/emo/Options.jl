# phuonglh
# 
options = Dict{Symbol,Any}(
    :afinnSize => 16,
    :hiddenSize => 32, 
    :batchSize => 64,
    :α => 1E-3, 
    :numEpochs => 100,
    :modelPath => string(pwd(), "/dat/emo/EMP/emp.bson")
)