# phuonglh
# 
options = Dict{Symbol,Any}(
    :afinnSize => 32,
    :hiddenSize => 32, 
    :batchSize => 64,
    :α => 1E-3, 
    :numEpochs => 80,
    :modelPath => string(pwd(), "/dat/emo/EMP/emp.bson")
)