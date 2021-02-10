module VLP

export VietnameseTokenizer, PoSTagger, NameTagger, Model, Mapper, Service, Resource, Client

include("tok/VietnameseTokenizer.jl")
using .VietnameseTokenizer

include("seq/PoSTagger.jl")
using .PoSTagger

include("seq/NameTagger.jl")
using .NameTagger

include("Model.jl")
using .Model

include("Mapper.jl")
using .Mapper

include("Service.jl")
using .Service

include("Resource.jl")
using .Resource

include("Client.jl")
using .Client

function run()
    Resource.run()
end

end # module