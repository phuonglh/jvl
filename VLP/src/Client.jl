module Client

using HTTP, JSON3
using ..Model

const SERVER = Ref{String}("http://localhost:8085")

"""
    listPrimes(u, v)

    List all prime numbers from `u` to `v` where `u` and `v` are two 
    positive integers.
"""
function listPrimes(u, v)::Array{Int}
    body = (; u, v)
    resp = HTTP.post(string(SERVER[], "/primes"), [], JSON3.write(body))
    return JSON3.read(resp.body, Array{Int})
end

"""
    tokenize(text)

    Tokenize a text and return result.
"""
function tokenize(text)::Analysis
    body = (; text)
    resp = HTTP.post(string(SERVER[], "/tok"), [], JSON3.write(body))
    return JSON3.read(resp.body, Analysis)
end

"""
    tag(text)

    Part-of-speech tag a text and return result.
"""
function tag(text)::Analysis
    body = (; text)
    resp = HTTP.post(string(SERVER[], "/tag"), [], JSON3.write(body))
    return JSON3.read(resp.body, Analysis)
end

end # module