using Flux
include("ClassifierBERT.jl")
include("ParserBERT.jl")
options = ArcEagerParserBERT.TransitionClassifierBERT.optionsEWT
ArcEagerParserBERT.evaluate(options)

