# phuonglh@gmail.com
# January 16, 2022
# WASSA 2022 Shared Task on Empathy Detection and Emotion Classification

using DataFrames
using CSV
using Flux

include("Options.jl")

df = DataFrame(CSV.File("dat/emo/messages_train_ready_for_WS.tsv", header=true))
ef = df[:, [:essay, :empathy, :distress, :gender, :education, :race, :age, :income]]

genders = unique(ef[:, :gender])        # [1, 2, 5]
educations = unique(ef[:, :education])  # [4, 6, 5, 7, 2, 3]
races = unique(ef[:, :race])            # [1, 5, 2, 3, 4, 6]

"""
    createBatches(df)

    Takes an input data frame and create batches of (X, Y) matrices. Each input matrix 
    X is of shape (17 x batchSize), and each output matrix Y of shape (2 x batchSize).
"""
function createBatches(df)
    function preprocess(t::NamedTuple)
        x = [t[:gender], t[:education], t[:race], t[:age]/10, t[:income]/10_000]
        f = vcat(Flux.onehot(x[1], genders), Flux.onehot(x[2], educations), Flux.onehot(x[3], races), x[4], x[5])
        Float32.(f)
    end
    namedTuples = Tables.rowtable(df)
    as = map(t -> (preprocess(t), Float32.([t[:empathy], t[:distress]])), namedTuples)
    bs = Flux.Data.DataLoader(as, batchsize = options[:batchSize])
    Xs = map(b -> Flux.stack(map(p -> p[1], b), 2), bs)
    Ys = map(b -> Flux.stack(map(p -> p[2], b), 2), bs)
    (Xs, Ys)
end

function createModel()
    Chain(
        Dense(17, options[:hiddenSize], relu),
        Dense(options[:hiddenSize], 2)
    )
end

function train(df)
    m = createModel()
    loss(X, Y) = Flux.Losses.mse(m(X), Y)
    ps = Flux.params(m)
    Xs, Ys = createBatches(df)
    data = zip(Xs, Ys)
    optimizer = ADAM(options[:α])
    cbf() = @show(sum(loss(X, Y) for (X, Y) in data))
    Flux.@epochs options[:numEpochs] Flux.train!(loss, ps, data, optimizer, cb = Flux.throttle(cbf, 5))
    m
end