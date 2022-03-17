# phuonglh@gmail.com
# January 16, 2022
# WASSA 2022 Shared Task on Empathy Detection and Emotion Classification

using DataFrames
using CSV
using Flux
using BSON: @save, @load
using TextAnalysis
using WordTokenizers
using Languages
using Statistics
using Plots; # plotly()

include("Embedding.jl")
include("Options.jl")

# training data
cf = DataFrame(CSV.File("dat/emo/messages_train_ready_for_WS.tsv", header=true))
df = cf[:, [:essay, :empathy, :distress, :emotion, :gender, :education, :race, :age, :income]]
# dev. data
cfd = DataFrame(CSV.File("dat/emo/messages_dev_features_ready_for_WS_2022.tsv", header=true))
dfd = cfd[:, [:essay, :gender, :education, :race, :age, :income]]
efd = DataFrame(CSV.File("dat/emo/goldstandard_dev_2022.tsv", header=false))
dfd[:, :empathy] = efd[:, 1]
dfd[:, :distress] = efd[:, 2]
dfd[:, :emotion] = efd[:, 3]
# test data
cft = DataFrame(CSV.File("dat/emo/messages_test_features_ready_for_WS_2022.tsv", header=true))
dft = cft[:, [:essay, :gender, :education, :race, :age, :income]]

# 
genders = unique(df[:, :gender])        # [1, 2, 5]
educations = unique(df[:, :education])  # [4, 6, 5, 7, 2, 3]
races = unique(df[:, :race])            # [1, 5, 2, 3, 4, 6]

# load AFINN map
adf = CSV.File("dat/emo/AFINN/AFINN-111.txt", header=false) |> DataFrame
afinn = Dict(zip(adf[:,1], adf[:,2])) # or use "Pair." instead of "zip"

# load NRC map
ndf = CSV.File("dat/emo/NRC/NRC-emotion-lexicon-v0.92/words.txt", header=false) |> DataFrame
# append a separator to the emotion column
transform!(ndf, :2 => x -> x .* "|")
# group by term
gdf = groupby(ndf, :1)
# combine emotions for each term
nrc = combine(gdf, :4 => join)
emotionsNRC = Dict(
    "anger" => 1, "fear" => 2, "anticipation" => 3, "trust" => 4, 
    "surprise" => 5, "sadness" => 6, "joy" => 7, "disgust" => 8, 
    "negative" => 9, "positive" => 10
)

function preprocessDocument(document)
    # lowercase and strip some tokens/punctuations
    remove_case!(document)
    prepare!(document, strip_stopwords | strip_articles | strip_numbers | strip_html_tags | strip_punctuation | strip_pronouns)
    stem!(document)
    return document
end

"""
    spotTokens(df, afinn, outputPath)

    Finds AFINN negative-score tokens in the essays of a df. 
"""
function spotTokens(df, afinn, outputPath="dat/emo/EMP/afinn_train.txt")
    texts = df[:, :essay]
    tokenized = map(text -> unique(tokenize(lowercase(text))), texts)
    selection = map(tokens -> filter(token -> token ∈ keys(afinn) && afinn[token] < 0, tokens), tokenized)
    open(outputPath, "w") do file
        ss = map(tokens -> join(tokens, " "), selection)
        write(file, join(ss, "\n"))
        write(file, "\n")
    end
end

function spotScores(df, afinn, outputPath)
    texts = df[:, :essay]
    tokenized = map(text -> unique(tokenize(lowercase(text))), texts)
    toScore(tokens) = map(token -> afinn[token], tokens)
    selection = map(tokens -> toScore(filter(token -> token ∈ keys(afinn), tokens)), tokenized)
    open(outputPath, "w") do file
        ss = map(tokens -> join(tokens, " "), selection)
        write(file, join(ss, "\n"))
        write(file, "\n")
    end
end


# create a corpus containing the texts, lower case the texts
texts = df[:, :essay]
documents = map(text -> preprocessDocument(StringDocument(text)), texts)
corpus = Corpus(documents)
# build the lexicon from this corpus
update_lexicon!(corpus)
# prune rare terms in the lexicon
for term in keys(corpus.lexicon)
    freq = corpus.lexicon[term]
    if freq <= options[:minFrequency] || freq >= options[:maxFrequency]
        delete!(corpus.lexicon, term)
    end
end
# create token map
mapTokens = Dict(term => i for (i, term) in enumerate(keys(corpus.lexicon)))

# # compute tf-idf matrix
# M = DocumentTermMatrix(corpus)
# tfidf = tf_idf(M)
# d = size(tfidf,2) # domain dimension

# create a lexicon containing words that appear in the AFINN lexicon
documentAFINN = FileDocument("dat/emo/EMP/afinn_train.txt")
corpusAFINN = Corpus([documentAFINN])
update_lexicon!(corpusAFINN)
# prune rare terms in the lexicon
for term in keys(corpusAFINN.lexicon)
    freq = corpusAFINN.lexicon[term]
    if freq < 2 || freq > 80
        delete!(corpusAFINN.lexicon, term)
    end
end # ==> 490 entries kept with cutoff threshold = 2
# create AFINN map
mapAFINN = Dict(term => i for (i, term) in enumerate(keys(lexiconAFINN)))

# create a lexicon containing terms that appear in the NRC lexicon
lexiconNRC = Dict(zip(nrc[:,1], nrc[:,2]))

"""
    preprocess(t)

    Compute features of a given sample, where `t` is a row (named tuple).
"""
function preprocess(t::NamedTuple)
    tokens = tokenize(lowercase(t[:essay]))
    # AFINN features
    afinnTokens = filter(token -> token ∈ keys(mapAFINN), tokens)
    u = map(token -> mapAFINN[token], afinnTokens)
    if isempty(u)
        u = [ length(mapAFINN)+1 ] # UNK token
    end

    # # Token features
    # selectedTokens = filter(token -> token ∈ keys(mapTokens), tokens)
    # u = map(token -> mapTokens[token], selectedTokens)
    # if isempty(u)
    #     u = [ length(mapTokens)+1 ] # UNK token
    # end

    # make the u vector the same length for the back-propagation of recurrent layers to work
    # only required by the JoinR model
    maxLen = options[:maxSequenceLength]
    u = if length(u) <= maxLen vcat(u, fill(1, maxLen-length(u))) else u[1:maxLen] end

    # NRC features
    β(token) = begin
        affects = split(lexiconNRC[token], "|")
        js = map(affect -> emotionsNRC[affect], affects[1:end-1]) # end-1 since there is a | at the end.
        vec(sum(Flux.onehotbatch(js, 1:length(emotionsNRC)), dims=2))
    end
    nrcTokens = filter(token -> token ∈ keys(lexiconNRC), tokens)
    ws = map(token -> β(token), nrcTokens)
    w = if isempty(ws)
        ws = zeros(length(emotionsNRC))
    else
        sum(ws)
    end

    # concatenation
    # x = [t[:gender], t[:education], t[:race], t[:age]/10, t[:income]/10_000]
    # v = vcat(Flux.onehot(x[1], genders), Flux.onehot(x[2], educations), Flux.onehot(x[3], races), x[4], x[5], w)
    x = [t[:gender], t[:education], t[:race]]
    v = vcat(Flux.onehot(x[1], genders), Flux.onehot(x[2], educations), Flux.onehot(x[3], races), w)
    # return a pair of input vectors: 
        # u is the AFINN-token vector; 
        # v is the real-valued vector concatenated with the NRC vector
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
    # 1. Chain(
    #     Join(
    #         Embedding(length(lexiconAFINN), options[:afinnSize]),
    #         identity,
    #     ),
    #     Dense(10 + 17 + options[:afinnSize], options[:hiddenSize], relu),
    #     Dense(options[:hiddenSize], 2)
    # )
    # 2. Chain(
    #     JoinR(
    #         Embedding(length(lexiconAFINN), options[:afinnSize]),            
    #         identity,
    #         GRU(options[:afinnSize], options[:recurrentSize])
    #     ),
    #     Dense(10 + 17 + options[:recurrentSize], options[:hiddenSize], relu),
    #     Dense(options[:hiddenSize], 2)
    # )
    Chain(
        JoinR(
            Embedding(length(mapAFINN) + 1, options[:afinnSize]),
            #Embedding(length(mapTokens) + 1, options[:afinnSize]),
            identity,
            GRU(options[:afinnSize], options[:recurrentSize])
        ),
        Dense(10 + 15 + options[:recurrentSize], options[:hiddenSize], relu),
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
    @info options
    model, Js = train(df, dfd)
    predict(df, model, "dat/emo/EMP/res/predictions_EMP_t.tsv")
    predict(dfd, model, "dat/emo/EMP/res/predictions_EMP_d.tsv")
    predict(dft, model, "dat/emo/EMP/res/predictions_EMP.tsv")
    return model, Js
end

function visualize(Js)
    J = hcat(map(p -> p[1], Js), map(p -> p[2], Js))
    plot(10:length(Js), J[10:end,:], label=["train" "dev."], lw=2, xlabel="epoch", ylabel="loss")
end


function emotionStatistic()
    gdf = groupby(df, :emotion)
    st = combine(gdf, nrow) # emotion histogram
end