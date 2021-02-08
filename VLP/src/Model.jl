module Model

using ..PoSTagger

options = PoSTagger.optionsVLSP2010

wordIndexPoS = PoSTagger.loadIndex(options[:wordPath])
shapeIndexPoS = PoSTagger.loadIndex(options[:shapePath])
posIndexPoS = PoSTagger.loadIndex(options[:posPath])
labelIndexPoS = PoSTagger.loadIndex(options[:labelPath])

encoderPoS = PoSTagger.loadEncoder(options)

end # module