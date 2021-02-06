module Resource

using HTTP, JSON3
using ..Model, ..Service

const ROUTER = HTTP.Router()

listPrimes(req) = Service.listPrimes(JSON3.read(req.body))::Array{Int}
HTTP.@register(ROUTER, "POST", "/primes", listPrimes)

function requestHandler(req)
    obj = HTTP.handle(ROUTER, req)
    return HTTP.Response(200, JSON3.write(obj))
end

function run()
    HTTP.serve(requestHandler, "0.0.0.0", 8080)
end

end # module