# phuonglh@gmail.com
# January 16, 2022
# WASSA 2022 Shared Task on Empathy Detection and Emotion Classification

using DataFrames
using CSV
using Flux
using BSON: @save, @load
using TextAnalysis
using WordTokenizers
using Statistics
using Plots; # plotly()

include("Embedding.jl")
include("Options.jl")

# training data
cf = DataFrame(CSV.File("dat/emo/messages_train_ready_for_WS.tsv", header=true))
df = cf[:, [:essay, :empathy, :distress, :gender, :education, :race, :age, :income]]
# dev. data
cfd = DataFrame(CSV.File("dat/emo/messages_dev_features_ready_for_WS_2022.tsv", header=true))
dfd = cfd[:, [:essay, :gender, :education, :race, :age, :income]]
efd = DataFrame(CSV.File("dat/emo/goldstandard_dev_2022.tsv", header=false))
dfd[:, :empathy] = efd[:, 1]
dfd[:, :distress] = efd[:, 2]

# 
genders = unique(ef[:, :gender])        # [1, 2, 5]
educations = unique(ef[:, :education])  # [4, 6, 5, 7, 2, 3]
races = unique(ef[:, :race])            # [1, 5, 2, 3, 4, 6]

# load AFINN map
adf = CSV.File("dat/emo/AFINN/AFINN-111.txt", header=false) |> DataFrame
afinn = Dict(zip(adf[:,1], adf[:,2])) # or use "Pair." instead of "zip"


function preprocessDocument(document)
    d = remove_case!(document)
    prepare!(d, strip_articles | strip_numbers | strip_html_tags)
    return d
end

function spot(df, afinn, outputPath)
    texts = df[:, :essay]
    tokenized = map(text -> tokenize(lowercase(text)), texts)
    selection = map(tokens -> filter(token -> token ∈ keys(afinn), tokens), tokenized)
    open(outputPath, "w") do file
        ss = map(tokens -> join(tokens, " "), selection)
        write(file, join(ss, "\n"))
        write(file, "\n")
    end
end

# create a corpus containing the texts, lower case the texts
# texts = df[:, :essay]
# documents = map(text -> preprocessDocument(StringDocument(text)), texts)
# corpus = Corpus(documents)
# # build the lexicon from this corpus
# update_lexicon!(corpus)
# lexiconWASA = lexicon(corpus)

# create a lexicon containing words that appear in the AFINN lexicon
document = FileDocument("dat/emo/EMP/afinn_train.txt")
corpus = Corpus([document])
update_lexicon!(corpus)
lexiconAFINN = lexicon(corpus) # 1,149 entries


"""
    preprocess(t)

    Pre-process input features.
"""
function preprocess(t::NamedTuple)
    tokens = tokenize(lowercase(t[:essay]))
    afinnTokens = filter(token -> token ∈ keys(lexiconAFINN), tokens)
    u = map(token -> lexiconAFINN[token], afinnTokens)
    if isempty(u)
        u = [ 1 ]
    end
    x = [t[:gender], t[:education], t[:race], t[:age]/10, t[:income]/10_000]
    v = vcat(Flux.onehot(x[1], genders), Flux.onehot(x[2], educations), Flux.onehot(x[3], races), x[4], x[5])
    # return a pair of input vectors
    (u, Float32.(v))
end

"""
    createBatches(df)

    Takes an input data frame and create batches of (X, Y) matrices. Each input matrix 
    X is of shape (17 x batchSize), and each output matrix Y of shape (2 x batchSize).
"""
function createBatches(df)
    namedTuples = Tables.rowtable(df)
    as = map(t -> (preprocess(t), Float32.([t[:empathy], t[:distress]])), namedTuples)
    bs = Flux.Data.DataLoader(as, batchsize = options[:batchSize])
    # Xs = map(b -> Flux.stack(map(p -> p[1], b), 2), bs)
    # Ys = map(b -> Flux.stack(map(p -> p[2], b), 2), bs)
    Xs = map(b -> map(p -> p[1], b), bs)
    Ys = map(b -> Flux.stack(map(p -> p[2], b), 2), bs)
    (Xs, Ys)
end

function createModel()
    Chain(
        Join(
            Embedding(length(lexiconAFINN), options[:afinnSize]),
            identity,
        ),
        Dense(17 + options[:afinnSize], options[:hiddenSize], relu),
        Dense(options[:hiddenSize], 2)
    )
end

"""
    train(df, dfd)

    Trains the model on a training data frame and a development data frame, returns a model.
"""
function train(df, dfd)
    model = createModel()
    @info sum(model[1].first.W)
    loss(X, Y) = Flux.Losses.mse(model(X), Y)
    ps = Flux.params(model)
    Xs, Ys = createBatches(df)
    Xsd, Ysd = createBatches(dfd)
    data = zip(Xs, Ys)
    datad = zip(Xsd, Ysd)
    optimizer = ADAM(options[:α])
    Js = []
    cbf() = begin
        γ = mean(loss(X, Y) for (X, Y) in data)  # training loss
        ζ = mean(loss(X, Y) for (X, Y) in datad) # dev. loss
        @show(γ, ζ) 
        push!(Js, (γ, ζ))
    end
    Flux.@epochs options[:numEpochs] Flux.train!(loss, ps, data, optimizer, cb = Flux.throttle(cbf, 5))
    @save options[:modelPath] model
    @info sum(model[1].first.W)
    return model, Js
end

"""
    predict(df, model, outputPath)

    Predicts an input df and writes results to an external file. The input df 
    is pre-processed and batched before being fed to a model.
"""
function predict(df, model, outputPath)
    namedTuples = Tables.rowtable(df)
    as = map(t -> (preprocess(t),), namedTuples) # create a tuple with empty second elements
    bs = Flux.Data.DataLoader(as, batchsize = options[:batchSize])
    Xs = map(b -> map(p -> p[1], b), bs)
    Zs = map(X -> model(X), Xs)
    toString(Z) = begin
        ss = [join(Z[:,j], "\t") for j=1:size(Z,2)]
        join(ss, "\n")
    end
    open(outputPath, "w") do file
        ss = map(Z -> toString(Z), Zs)
        write(file, join(ss, "\n"))
        write(file, "\n")
    end
end

function main()
    model, Js = train(df, dfd)
    n = length(Js)
    J = hcat(map(p -> p[1], Js), map(p -> p[2], Js))
    plot(1:n, J, label=["train" "dev."], lw=2, xlabel="epoch", ylabel="loss")
    predict(df, model, "dat/emo/EMP/res/predictions_EMP.tsv")
    predict(dfd, model, "dat/emo/EMP/res/predictions_EMP_d.tsv")
end