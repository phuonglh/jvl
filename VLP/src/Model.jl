module Model

export wordIndexPoS, shapeIndexPoS, posIndexPoS, labelIndexPoS, encoderPoS
export wordIndexName, shapeIndexName, posIndexName, labelIndexName, encoderName

using ..PoSTagger
using ..NameTagger

# Part-of-speech tagging
options = PoSTagger.optionsVLSP2010

wordIndexPoS = PoSTagger.loadIndex(options[:wordPath])
shapeIndexPoS = PoSTagger.loadIndex(options[:shapePath])
posIndexPoS = PoSTagger.loadIndex(options[:posPath])
labelIndexPoS = PoSTagger.loadIndex(options[:labelPath])

Core.eval(Main, :(import VLP.PoSTagger, Flux))
encoderPoS = PoSTagger.loadEncoder(options)

# Named-entity recognition
optionsName = NameTagger.optionsVLSP2016

wordIndexName = NameTagger.loadIndex(optionsName[:wordPath])
shapeIndexName = NameTagger.loadIndex(optionsName[:shapePath])
posIndexName = NameTagger.loadIndex(optionsName[:posPath])
labelIndexName = NameTagger.loadIndex(optionsName[:labelPath])

Core.eval(Main, :(import VLP.NameTagger, Flux))
encoderName = NameTagger.loadEncoder(optionsName)

end # module