module Client

using HTTP, JSON3
using ..Model

const SERVER = Ref{String}("http://localhost:8080")

"""
    listPrimes(u, v)

    List all prime numbers from `u` to `v` where `u` and `v` are two 
    positive integers.
"""
function listPrimes(u, v)
    body = (; u, v)
    resp = HTTP.post(string(SERVER[], "/primes"), [], JSON3.write(body))
    return JSON3.read(resp.body, Array{Int})
end

end # module