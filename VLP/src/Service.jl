module Service

using ..Model, ..Mapper

using ..VietnameseTokenizer

function listPrimes(obj)::Array{Int}
    @info obj
    @assert haskey(obj, :u) && !isempty(obj.u)
    @assert haskey(obj, :v) && !isempty(obj.v)
    primes = Array{Int,1}()
    for k=max(2,obj.u):obj.v
        p = true
        for j = 2:sqrt(k)
            if mod(k, j) == 0
                p = false
                break
            end
        end
        if p push!(primes, k); end
    end
    return primes
end

"""
    tokenize(obj)

    Tokenize the text in a JSON object and return the result in the form 
    of an array of (word, shape) pairs.
"""
function tokenize(obj)::Array{Tuple{String,String}}
    @info obj
    @assert haskey(obj, :text) && !isempty(obj.text)
    tokens = VietnameseTokenizer.tokenize(obj.text)
    xs = map(token -> (token.text, token.form), tokens)
    Mapper.store!(:tok, xs)
    return xs
end

end # module