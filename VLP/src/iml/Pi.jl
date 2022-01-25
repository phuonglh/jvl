function fv(n)
    x = √2
    r = x/2
    for _=2:n
        y = √(2 + x)
        r = r * y/2
        x = y
    end
    return 2/r
end

using Plots
n = 10
plot(1:10, [fvs, fill(pi, 10)], xlabel="n", legend=["fv", "pi"])