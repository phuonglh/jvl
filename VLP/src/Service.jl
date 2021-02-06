module Service

using ..Model, ..Mapper

function listPrimes(obj)
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
    Mapper.store!(primes)
    return primes
end

end # module