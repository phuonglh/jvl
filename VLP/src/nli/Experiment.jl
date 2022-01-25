include("NLI.jl")
using Flux

hiddenSizes = [16, 32, 64, 128, 256]
@info "Loading BERT representations of the training samples..."
@time Xb, Yb = NLI.batchVectors(NLI.trainDF, string(pwd(), "/dat/nli/x/train.txt"))
@info "Loading BERT representations of the development samples..."
@time Xb_dev, Yb_dev = NLI.batchVectors(NLI.devDF, string(pwd(), "/dat/nli/x/dev.txt"))

# compute test score
@info "Loading BERT representations of the test samples..."
@time Xb_test, Yb_test = NLI.batchVectors(NLI.testDF, string(pwd(), "/dat/nli/x/test.txt"))

file = open("dat/nli/x/scores.txt", "w")
for hiddenSize in hiddenSizes
    for k=1:3
        NLI.options[:hiddenSize] = hiddenSize
        NLI.options[:modelPath] = string(pwd(), "/dat/nli/x/", hiddenSize, "/en.bson.", k)
        # define a neural network of two layers for classification
        model = Chain(Dense(768, NLI.options[:hiddenSize]), Dense(NLI.options[:hiddenSize], 3))
        a, b, c = NLI.run(NLI.options, model, Xb, Yb, Xb_dev, Yb_dev, Xb_test, Yb_test)
        write(file, string(a, " ", b, " ", c, "\n"))
    end
end
close(file)