using Flux

function foo()
    s = Float32.([1, 2, 3])
    gru = GRU(3, 3)
    V = Float32.(rand(3, 5))
    println(V)
    function bar(j)
        s = gru(V[:,j])
        @info s
        return s
    end
    println("Initial s: ")
    @info s
    x = [bar(j) for j=1:5]
    println("Final s: ")
    @info s
end